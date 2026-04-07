"""
Phase 1: ClusPro + Dual Prompt + Alpha Gating (NO VAPS, NO DHNO)

Baseline에서 변경:
- 단일 soft_att_obj → Primitive(frozen) + Contextual(learnable)
- Alpha Gating: entropy + confidence + f_global → per-image α
- per-image text loop 없음 → batch_size 유지 가능
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import CustomTextEncoder
from model.nce_loss import ContrastiveLoss
from model.hsic import hsic_normalized


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck=64, dropout=0.0, adapter_scalar="0.1"):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, d_model)
        self.dropout = dropout
        self.scale = float(adapter_scalar)
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        down = self.non_linear_func(self.down_proj(x))
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down) * self.scale
        return (up + residual) if add_residual else up


class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        return F.dropout(F.relu(self.bn1_fc(self.fc1(x))), training=self.training)


class ClusProDual(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()

        clip_arch = config.clip_arch if hasattr(config, 'clip_arch') and config.clip_arch else config.clip_model
        self.clip = load_clip(name=clip_arch, context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.offset = offset
        self.enable_pos_emb = True

        dtype = self.clip.dtype or torch.float16
        self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, dtype)

        for p in self.parameters():
            p.requires_grad = False

        output_dim = self.clip.visual.output_dim

        # ---- Visual Adapters (ClusPro) ----
        num_blocks = self.clip.visual.transformer.layers
        vision_width = self.clip.visual.transformer.width
        adapter_dim = getattr(config, 'adapter_dim', 64)
        adapter_dropout = getattr(config, 'adapter_dropout', 0.1)
        self.visual_adapters = nn.ModuleList([
            Adapter(vision_width, adapter_dim, adapter_dropout)
            for _ in range(2 * num_blocks)
        ])

        # ---- Disentanglers (ClusPro) ----
        self.attr_disentangler = Disentangler(output_dim)
        self.obj_disentangler = Disentangler(output_dim)
        self.attr_proj = Disentangler(output_dim)
        self.obj_proj = Disentangler(output_dim)

        # ---- Prototype Memory (ClusPro) ----
        self.cluster_num = getattr(config, 'cluster_num', 5)
        self.momentum = getattr(config, 'proto_momentum', 0.99)

        for i in range(len(self.attributes)):
            self.register_buffer(f"attr_queue{i}", torch.randn(self.cluster_num, output_dim))
        for i in range(len(self.classes)):
            self.register_buffer(f"obj_queue{i}", torch.randn(self.cluster_num, output_dim))

        self.nceloss = ContrastiveLoss()
        self.attr_dropout = nn.Dropout(getattr(config, 'attr_dropout', 0.3))

        # ---- Dual Soft Prompt (DPC) ----
        self.token_ids, base_soft_emb, self.comp_ctx, self.attr_ctx, self.obj_ctx = \
            self._construct_soft_prompt()

        self.soft_att_obj_prim = nn.Parameter(base_soft_emb.clone())
        self.soft_att_obj_ctx = nn.Parameter(base_soft_emb.clone())
        self.comp_ctx = nn.Parameter(self.comp_ctx)
        self.attr_ctx = nn.Parameter(self.attr_ctx)
        self.obj_ctx = nn.Parameter(self.obj_ctx)

        # Freeze primitive
        freeze_primitive = getattr(config, 'freeze_primitive', True)
        if freeze_primitive:
            self.soft_att_obj_prim.requires_grad = False
            print("[ClusProDual] Primitive soft embeddings frozen.")

        # ---- Alpha Gating (DPC-Alpha) ----
        self.gating_param = nn.Parameter(torch.tensor(0.0))
        self.alpha_predictor = nn.Sequential(
            nn.Linear(2 + output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # ---- Loss/Inference weights ----
        self.pair_loss_weight = getattr(config, 'pair_loss_weight', 1.0)
        self.attr_loss_weight = getattr(config, 'attr_loss_weight', 1.0)
        self.obj_loss_weight = getattr(config, 'obj_loss_weight', 1.0)
        self.contrastive_weight = getattr(config, 'contrastive_weight', 0.1)
        self.hsic_weight = getattr(config, 'hsic_weight', 0.1)
        self.pair_inf_w = getattr(config, 'pair_inference_weight', 1.0)
        self.attr_inf_w = getattr(config, 'attr_inference_weight', 1.0)
        self.obj_inf_w = getattr(config, 'obj_inference_weight', 1.0)

    # ==================================================================
    # Image Encoding (ClusPro — identical to baseline)
    # ==================================================================

    def encode_image(self, x):
        x = self.clip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.clip.visual.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        num_blocks = self.clip.visual.transformer.layers
        for i in range(num_blocks):
            block = self.clip.visual.transformer.resblocks[i]
            adapt_x = self.visual_adapters[i](x, add_residual=False)
            residual = x
            x = block.attention(block.ln_1(x))
            x = x + adapt_x + residual
            adapt_x = self.visual_adapters[i + num_blocks](x, add_residual=False)
            residual = x
            x = block.mlp(block.ln_2(x))
            x = x + adapt_x + residual

        x = x.permute(1, 0, 2)
        x = self.clip.visual.ln_post(x)
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        return x[:, 0, :], x

    # ==================================================================
    # Soft Prompt
    # ==================================================================

    def _construct_soft_prompt(self):
        prompt_template = self.config.prompt_template
        ctx_init = self.config.ctx_init

        token_ids = self.tokenizer(
            prompt_template, context_length=self.config.context_length
        ).cuda()

        tokenized = torch.cat([
            self.tokenizer(tok, context_length=self.config.context_length)
            for tok in self.attributes + self.classes
        ])
        orig_emb = self.clip.token_embedding(tokenized.cuda())
        soft_emb = torch.zeros(
            len(self.attributes) + len(self.classes), orig_emb.size(-1)
        )
        for idx, rep in enumerate(orig_emb):
            eos_idx = tokenized[idx].argmax()
            soft_emb[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init, context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx = embedding[0, 1:1+n_ctx[0], :].to(self.clip.dtype)
        attr_ctx = embedding[1, 1:1+n_ctx[1], :].to(self.clip.dtype)
        obj_ctx = embedding[2, 1:1+n_ctx[2], :].to(self.clip.dtype)

        return token_ids, soft_emb, comp_ctx, attr_ctx, obj_ctx

    def _construct_token_tensors(self, pair_idx, soft_att_obj):
        """Build token tensors using given soft embeddings."""
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        num_elements = [len(pair_idx), self.offset, len(self.classes)]
        token_tensor = []

        for i in range(self.token_ids.shape[0]):
            ids = self.token_ids[i].repeat(num_elements[i], 1)
            token_tensor.append(
                self.clip.token_embedding(ids.cuda()).type(self.clip.dtype)
            )

        eos_idx = [int(self.token_ids[i].argmax()) for i in range(self.token_ids.shape[0])]
        embs = self.attr_dropout(soft_att_obj)

        token_tensor[0][:, eos_idx[0]-2, :] = embs[attr_idx].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0]-1, :] = embs[obj_idx + self.offset].type(self.clip.dtype)
        token_tensor[0][:, 1:len(self.comp_ctx)+1, :] = self.comp_ctx.type(self.clip.dtype)

        token_tensor[1][:, eos_idx[1]-1, :] = embs[:self.offset].type(self.clip.dtype)
        token_tensor[1][:, 1:len(self.attr_ctx)+1, :] = self.attr_ctx.type(self.clip.dtype)

        token_tensor[2][:, eos_idx[2]-1, :] = embs[self.offset:].type(self.clip.dtype)
        token_tensor[2][:, 1:len(self.obj_ctx)+1, :] = self.obj_ctx.type(self.clip.dtype)

        return token_tensor

    # ==================================================================
    # Prototype (ClusPro — identical to baseline)
    # ==================================================================

    @torch.no_grad()
    @torch.autocast(device_type='cuda', enabled=False)
    def _update_prototypes(self, batch_attr, batch_obj, attr_idx, obj_idx):
        batch_attr_f = batch_attr.float()
        batch_obj_f = batch_obj.float()

        for k in range(len(self.attributes)):
            mask = (attr_idx == k)
            if mask.sum() == 0:
                continue
            feats_k = batch_attr_f[mask]
            queue_k = getattr(self, f"attr_queue{k}")
            queue_f = queue_k.float()
            sim = l2_normalize(feats_k) @ l2_normalize(queue_f).t()
            assign = F.softmax(sim / 0.5, dim=-1)
            new_proto = assign.t() @ feats_k
            counts = assign.sum(dim=0)
            valid = counts > 0
            if valid.any():
                new_proto[valid] = l2_normalize(new_proto[valid])
                queue_f[valid] = queue_f[valid] * self.momentum + new_proto[valid] * (1 - self.momentum)
            queue_k.copy_(l2_normalize(queue_f).to(queue_k.dtype))

        for k in range(len(self.classes)):
            mask = (obj_idx == k)
            if mask.sum() == 0:
                continue
            feats_k = batch_obj_f[mask]
            queue_k = getattr(self, f"obj_queue{k}")
            queue_f = queue_k.float()
            sim = l2_normalize(feats_k) @ l2_normalize(queue_f).t()
            assign = F.softmax(sim / 0.5, dim=-1)
            new_proto = assign.t() @ feats_k
            counts = assign.sum(dim=0)
            valid = counts > 0
            if valid.any():
                new_proto[valid] = l2_normalize(new_proto[valid])
                queue_f[valid] = queue_f[valid] * self.momentum + new_proto[valid] * (1 - self.momentum)
            queue_k.copy_(l2_normalize(queue_f).to(queue_k.dtype))

    def _get_cluster_labels(self, batch_feat, idx_labels, primitives, prim_type):
        B = batch_feat.shape[0]
        labels = torch.zeros(B, dtype=torch.long, device=batch_feat.device)
        for k in range(len(primitives)):
            mask = (idx_labels == k)
            if mask.sum() == 0:
                continue
            queue_k = getattr(self, f"{prim_type}_queue{k}")
            sim = l2_normalize(batch_feat[mask]) @ l2_normalize(queue_k).t()
            cluster_idx = sim.argmax(dim=-1)
            labels[mask] = cluster_idx + k * self.cluster_num
        return labels

    def _gather_all_prototypes(self, prim_type, primitives):
        protos = []
        for k in range(len(primitives)):
            protos.append(getattr(self, f"{prim_type}_queue{k}"))
        return torch.stack(protos, dim=0)

    # ==================================================================
    # Alpha Gating
    # ==================================================================

    def _compute_alpha(self, logits_prim, f_global_norm):
        logits_f32 = logits_prim.float()
        log_probs = F.log_softmax(logits_f32, dim=-1)
        entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1, keepdim=True)
        max_entropy = torch.log(torch.tensor(float(logits_prim.size(-1)), device=logits_prim.device))
        norm_entropy = entropy / max_entropy

        max_logit = logits_f32.max(dim=-1, keepdim=True)[0]
        logit_scale = self.clip.logit_scale.exp()
        norm_max_logit = max_logit / logit_scale

        alpha_input = torch.cat([norm_entropy, norm_max_logit, f_global_norm.float()], dim=-1)
        alpha_raw = self.gating_param.float() + self.alpha_predictor(alpha_input)
        alpha = torch.sigmoid(alpha_raw).to(self.clip.dtype)
        return torch.clamp(alpha, min=0.05, max=0.95)

    # ==================================================================
    # Forward
    # ==================================================================

    def train_forward(self, batch, idx):
        batch_img = batch[0].cuda()
        attr_idx, obj_idx = batch[1], batch[2]

        f_global, _ = self.encode_image(batch_img.type(self.clip.dtype))
        B = f_global.shape[0]

        # Disentangle
        f_attr = self.attr_disentangler(f_global)
        f_obj = self.obj_disentangler(f_global)

        # Prototype update
        self._update_prototypes(f_attr, f_obj, attr_idx, obj_idx)

        # Cluster labels + contrastive
        attr_labels = self._get_cluster_labels(f_attr, attr_idx, self.attributes, "attr")
        obj_labels = self._get_cluster_labels(f_obj, obj_idx, self.classes, "obj")

        attr_protos = self._gather_all_prototypes("attr", self.attributes)
        obj_protos = self._gather_all_prototypes("obj", self.classes)

        f_attr_proj = self.attr_proj(f_attr)
        f_obj_proj = self.obj_proj(f_obj)

        all_protos_flat = torch.cat([
            self.attr_proj(attr_protos.reshape(-1, attr_protos.shape[-1])).reshape(-1, attr_protos.shape[-1]),
            self.obj_proj(obj_protos.reshape(-1, obj_protos.shape[-1])).reshape(-1, obj_protos.shape[-1]),
        ], dim=0)

        loss_contrastive = (
            self.nceloss(f_attr_proj, all_protos_flat.unsqueeze(0).expand(B, -1, -1), attr_labels) +
            self.nceloss(f_obj_proj, all_protos_flat.unsqueeze(0).expand(B, -1, -1), obj_labels)
        )

        # HSIC
        attr_proto_sel = torch.stack([
            getattr(self, f"attr_queue{int(attr_idx[i])}")[attr_labels[i] % self.cluster_num]
            for i in range(B)
        ])
        obj_proto_sel = torch.stack([
            getattr(self, f"obj_queue{int(obj_idx[i])}")[obj_labels[i] % self.cluster_num]
            for i in range(B)
        ])
        loss_hsic = (
            hsic_normalized(f_attr_proj, obj_proto_sel.detach()) +
            0.5 * hsic_normalized(f_obj_proj, attr_proto_sel.detach())
        )

        # ---- Dual Text Encoding (NO per-image loop) ----
        tokens_prim = self._construct_token_tensors(idx, self.soft_att_obj_prim)
        tokens_ctx = self._construct_token_tensors(idx, self.soft_att_obj_ctx)

        img_feats = [f_global, f_attr_proj, f_obj_proj]
        norm_img = [f / f.norm(dim=-1, keepdim=True) for f in img_feats]
        logit_scale = self.clip.logit_scale.exp()

        logits_prim = []
        logits_ctx = []
        for i in range(self.token_ids.shape[0]):
            feat_p, _ = self.text_encoder(self.token_ids[i], tokens_prim[i], enable_pos_emb=self.enable_pos_emb)
            feat_p = feat_p / feat_p.norm(dim=-1, keepdim=True)
            logits_prim.append(logit_scale * norm_img[i] @ feat_p.t())

            feat_c, _ = self.text_encoder(self.token_ids[i], tokens_ctx[i], enable_pos_emb=self.enable_pos_emb)
            feat_c = feat_c / feat_c.norm(dim=-1, keepdim=True)
            logits_ctx.append(logit_scale * norm_img[i] @ feat_c.t())

        # Alpha gating on comp branch only
        alpha = self._compute_alpha(logits_prim[0], norm_img[0])
        comp_logits = alpha * logits_prim[0] + (1.0 - alpha) * logits_ctx[0]

        # attr/obj: use contextual
        attr_logits = logits_ctx[1]
        obj_logits = logits_ctx[2]

        return comp_logits, attr_logits, obj_logits, loss_contrastive, loss_hsic

    def val_forward(self, batch, idx):
        batch_img = batch[0].cuda()
        f_global, _ = self.encode_image(batch_img.type(self.clip.dtype))

        f_attr = self.attr_disentangler(f_global)
        f_obj = self.obj_disentangler(f_global)
        f_attr_proj = self.attr_proj(f_attr)
        f_obj_proj = self.obj_proj(f_obj)

        tokens_prim = self._construct_token_tensors(idx, self.soft_att_obj_prim)
        tokens_ctx = self._construct_token_tensors(idx, self.soft_att_obj_ctx)

        img_feats = [f_global, f_attr_proj, f_obj_proj]
        norm_img = [f / f.norm(dim=-1, keepdim=True) for f in img_feats]
        logit_scale = self.clip.logit_scale.exp()

        logits_prim = []
        logits_ctx = []
        for i in range(self.token_ids.shape[0]):
            feat_p, _ = self.text_encoder(self.token_ids[i], tokens_prim[i], enable_pos_emb=self.enable_pos_emb)
            feat_p = feat_p / feat_p.norm(dim=-1, keepdim=True)
            logits_prim.append(logit_scale * norm_img[i] @ feat_p.t())

            feat_c, _ = self.text_encoder(self.token_ids[i], tokens_ctx[i], enable_pos_emb=self.enable_pos_emb)
            feat_c = feat_c / feat_c.norm(dim=-1, keepdim=True)
            logits_ctx.append(logit_scale * norm_img[i] @ feat_c.t())

        alpha = self._compute_alpha(logits_prim[0], norm_img[0])
        comp_logits = alpha * logits_prim[0] + (1.0 - alpha) * logits_ctx[0]

        return comp_logits, logits_ctx[1], logits_ctx[2]

    # ==================================================================
    # Loss & Inference (same as baseline)
    # ==================================================================

    def loss_calu(self, predict, target):
        loss_fn = nn.CrossEntropyLoss()
        batch_attr, batch_obj, batch_target = target[1], target[2], target[3]
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()

        if self.training:
            comp_logits, attr_logits, obj_logits, loss_contras, loss_hsic = predict
        else:
            comp_logits, attr_logits, obj_logits = predict

        loss = (
            self.pair_loss_weight * loss_fn(comp_logits, batch_target) +
            self.attr_loss_weight * loss_fn(attr_logits, batch_attr) +
            self.obj_loss_weight * loss_fn(obj_logits, batch_obj)
        )

        if self.training:
            loss = loss + self.contrastive_weight * loss_contras + self.hsic_weight * loss_hsic

        return loss

    def logit_infer(self, predict, pairs):
        comp_logits, attr_logits, obj_logits = predict
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)

        for i in range(comp_logits.shape[-1]):
            w_attr = 1 if self.attr_inf_w == 0 else attr_pred[:, pairs[i][0]] * self.attr_inf_w
            w_obj = 1 if self.obj_inf_w == 0 else obj_pred[:, pairs[i][1]] * self.obj_inf_w
            comp_logits[:, i] = comp_logits[:, i] * self.pair_inf_w + w_attr * w_obj

        return comp_logits

    def forward(self, batch, idx):
        if self.training:
            return self.train_forward(batch, idx)
        else:
            with torch.no_grad():
                return self.val_forward(batch, idx)
