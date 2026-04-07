"""
AdaptDPC: Visual Adapter + DPC (Dual Prompt Composition)

역할 분리 설계:
- Vision쪽: Visual Adapter만 (feature quality 개선, compositional reasoning X)
- Text쪽:  DPC가 compositional reasoning 전담
  - Primitive branch: frozen general soft embeddings
  - Contextual branch: learnable composition-specific soft embeddings
  - VAPS: f_local → per-image prompt shift (vision→text bridge)
  - Alpha Gating: entropy + confidence + f_global → per-image α
- Loss: CE(comp/attr/obj) + DHNO (no contrastive, no HSIC)

제거된 것 (ClusPro에서):
- Disentangler (attr/obj 분리를 vision에서 안 함)
- Prototype Clustering / Memory Banks
- Contrastive Loss, HSIC Loss
→ Vision쪽 gradient가 CE loss에만 집중하여 깨끗한 feature 생성
→ DPC의 compositional reasoning과 역할 충돌 없음
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import CustomTextEncoder


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class Adapter(nn.Module):
    """LoRA-style adapter for ViT blocks — domain feature improvement only."""
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


class AdaptDPC(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()

        # ---- CLIP backbone ----
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

        # ---- Freeze CLIP ----
        for p in self.parameters():
            p.requires_grad = False

        # ---- Dimensions ----
        output_dim = self.clip.visual.output_dim
        num_blocks = self.clip.visual.transformer.layers
        vision_width = self.clip.visual.transformer.width

        # ==============================================================
        # [Vision] Visual Adapters only — no disentangler, no prototype
        # ==============================================================

        adapter_dim = getattr(config, 'adapter_dim', 64)
        adapter_dropout = getattr(config, 'adapter_dropout', 0.1)
        self.visual_adapters = nn.ModuleList([
            Adapter(vision_width, adapter_dim, adapter_dropout)
            for _ in range(2 * num_blocks)
        ])

        self.attr_dropout = nn.Dropout(getattr(config, 'attr_dropout', 0.3))

        # ==============================================================
        # [Text / DPC] Dual Prompt — compositional reasoning 전담
        # ==============================================================

        self.token_ids, base_soft_emb, self.comp_ctx, self.attr_ctx, self.obj_ctx = \
            self._construct_soft_prompt()

        # Primitive: general knowledge (frozen)
        self.soft_att_obj_prim = nn.Parameter(base_soft_emb.clone())
        # Contextual: composition-specific knowledge (learnable)
        self.soft_att_obj_ctx = nn.Parameter(base_soft_emb.clone())
        # Context prefix tokens (learnable)
        self.comp_ctx = nn.Parameter(self.comp_ctx)
        self.attr_ctx = nn.Parameter(self.attr_ctx)
        self.obj_ctx = nn.Parameter(self.obj_ctx)

        # Dynamic Alpha Gating
        self.gating_param = nn.Parameter(torch.tensor(0.0))
        self.alpha_predictor = nn.Sequential(
            nn.Linear(2 + output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # ==============================================================
        # [MSCI] f_local from mid-layer ViT block
        # ==============================================================

        self.feature_layer = getattr(config, 'feature_layer', num_blocks // 2)

        # ==============================================================
        # [VAPS] Visual-Adaptive Prompt Shifting
        # f_local → shift vector → contextual prompt에 per-image 적용
        # Vision→Text의 유일한 bridge
        # ==============================================================

        self.prompt_shifter = nn.Sequential(
            nn.Linear(vision_width, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

        # ==============================================================
        # Freeze primitive soft embeddings
        # ==============================================================

        freeze_primitive = getattr(config, 'freeze_primitive', True)
        if freeze_primitive:
            self.soft_att_obj_prim.requires_grad = False
            print("[AdaptDPC] Primitive soft embeddings frozen.")

        # ==============================================================
        # [DHNO] Dual Hard Negative Optimization
        # ==============================================================

        self.dhno_lambda = getattr(config, 'dhno_lambda', 0.3)
        self.dhno_margin = getattr(config, 'dhno_margin', 1.0)

        # ---- Loss weights ----
        self.pair_loss_weight = getattr(config, 'pair_loss_weight', 1.0)
        self.attr_loss_weight = getattr(config, 'attr_loss_weight', 1.0)
        self.obj_loss_weight = getattr(config, 'obj_loss_weight', 1.0)

        # ---- Inference weights ----
        self.pair_inf_w = getattr(config, 'pair_inference_weight', 1.0)
        self.attr_inf_w = getattr(config, 'attr_inference_weight', 1.0)
        self.obj_inf_w = getattr(config, 'obj_inference_weight', 1.0)

    # ==================================================================
    # Image Encoding (Adapter + MSCI)
    # ==================================================================

    def encode_image(self, x):
        """Encode image with adapter. Captures f_local at mid-layer for VAPS."""
        x = self.clip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.clip.visual.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        num_blocks = self.clip.visual.transformer.layers
        f_local = None
        for i in range(num_blocks):
            block = self.clip.visual.transformer.resblocks[i]
            # MHA + adapter
            adapt_x = self.visual_adapters[i](x, add_residual=False)
            residual = x
            x = block.attention(block.ln_1(x))
            x = x + adapt_x + residual
            # FFN + adapter
            adapt_x = self.visual_adapters[i + num_blocks](x, add_residual=False)
            residual = x
            x = block.mlp(block.ln_2(x))
            x = x + adapt_x + residual

            # [MSCI] Capture f_local at mid-layer
            if i == self.feature_layer:
                f_local = x[0].detach()  # CLS token: [B, vision_width]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.visual.ln_post(x)
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        f_global = x[:, 0, :]  # [B, output_dim]
        return f_global, f_local

    # ==================================================================
    # Soft Prompt Construction
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
        """Build token tensors for comp/attr/obj using given soft embeddings."""
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

        # comp: [prefix] ... [attr] [obj]
        token_tensor[0][:, eos_idx[0]-2, :] = embs[attr_idx].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0]-1, :] = embs[obj_idx + self.offset].type(self.clip.dtype)
        token_tensor[0][:, 1:len(self.comp_ctx)+1, :] = self.comp_ctx.type(self.clip.dtype)

        # attr-only
        token_tensor[1][:, eos_idx[1]-1, :] = embs[:self.offset].type(self.clip.dtype)
        token_tensor[1][:, 1:len(self.attr_ctx)+1, :] = self.attr_ctx.type(self.clip.dtype)

        # obj-only
        token_tensor[2][:, eos_idx[2]-1, :] = embs[self.offset:].type(self.clip.dtype)
        token_tensor[2][:, 1:len(self.obj_ctx)+1, :] = self.obj_ctx.type(self.clip.dtype)

        return token_tensor

    # ==================================================================
    # Alpha Gating
    # ==================================================================

    def _compute_alpha(self, logits_prim, f_global_norm):
        """Per-image gating between primitive and contextual branches."""
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
    # Text Encoding
    # ==================================================================

    def _encode_text_prim(self, pair_idx):
        """Encode primitive (static) text branch."""
        tokens_prim = self._construct_token_tensors(pair_idx, self.soft_att_obj_prim)
        text_feats = []
        for i in range(self.token_ids.shape[0]):
            feat, _ = self.text_encoder(
                self.token_ids[i], tokens_prim[i], enable_pos_emb=self.enable_pos_emb
            )
            text_feats.append(feat / feat.norm(dim=-1, keepdim=True))
        return text_feats

    def _encode_text_ctx_shifted(self, pair_idx, shift):
        """[VAPS] Encode contextual text branch with per-image prompt shifting.

        Args:
            pair_idx: [N_pairs, 2]
            shift:    [B, output_dim] — per-image shift from prompt_shifter(f_local)

        Returns:
            shifted_comp_feats: list of [N_pairs, D] per image
            ctx_attr_feat: [N_attr, D]
            ctx_obj_feat: [N_obj, D]
        """
        # Static contextual features for attr-only and obj-only branches
        tokens_ctx = self._construct_token_tensors(pair_idx, self.soft_att_obj_ctx)
        static_feats = []
        for i in range(self.token_ids.shape[0]):
            feat, _ = self.text_encoder(
                self.token_ids[i], tokens_ctx[i], enable_pos_emb=self.enable_pos_emb
            )
            static_feats.append(feat / feat.norm(dim=-1, keepdim=True))

        # For comp branch: apply per-image shift
        attr_idx = pair_idx[:, 0]
        obj_idx = pair_idx[:, 1]
        eos_idx = int(self.token_ids[0].argmax())
        B = shift.shape[0]

        # Base comp token tensor
        base_ids = self.token_ids[0].repeat(len(pair_idx), 1)
        base_tensor = self.clip.token_embedding(base_ids.cuda()).type(self.clip.dtype)

        embs = self.attr_dropout(self.soft_att_obj_ctx)
        base_tensor[:, eos_idx - 2, :] = embs[attr_idx].type(self.clip.dtype)
        base_tensor[:, eos_idx - 1, :] = embs[obj_idx + self.offset].type(self.clip.dtype)
        base_tensor[:, 1:len(self.comp_ctx)+1, :] = self.comp_ctx.type(self.clip.dtype)

        # Per-image: shift the attr/obj slots
        shift_typed = shift.to(self.clip.dtype)  # [B, D]
        shifted_comp_feats = []
        for b in range(B):
            shifted = base_tensor.clone()
            shifted[:, eos_idx - 2, :] += shift_typed[b]
            shifted[:, eos_idx - 1, :] += shift_typed[b]
            feat, _ = self.text_encoder(
                self.token_ids[0], shifted, enable_pos_emb=self.enable_pos_emb
            )
            shifted_comp_feats.append(feat / feat.norm(dim=-1, keepdim=True))

        return shifted_comp_feats, static_feats[1], static_feats[2]

    # ==================================================================
    # [DHNO] Dual Hard Negative Optimization
    # ==================================================================

    def _compute_dhno_loss(self, comp_logits, batch_target, train_pairs):
        """Triplet margin loss on contextual branch with hard negatives."""
        B = len(batch_target)
        N = train_pairs.shape[0]
        attrs = train_pairs[:, 0]
        objs = train_pairs[:, 1]

        mask_same_obj = (objs.unsqueeze(1) == objs.unsqueeze(0))
        mask_same_attr = (attrs.unsqueeze(1) == attrs.unsqueeze(0))
        eye = torch.eye(N, device=comp_logits.device).bool()
        mask_same_obj = mask_same_obj & (~eye)
        mask_same_attr = mask_same_attr & (~eye)

        cur_mask_obj = mask_same_obj[batch_target]
        cur_mask_attr = mask_same_attr[batch_target]

        pos_scores = comp_logits[torch.arange(B, device=comp_logits.device), batch_target]

        obj_logits = comp_logits.clone()
        obj_logits[~cur_mask_obj] = -1e9
        hard_neg_obj, _ = obj_logits.max(dim=-1)

        attr_logits = comp_logits.clone()
        attr_logits[~cur_mask_attr] = -1e9
        hard_neg_attr, _ = attr_logits.max(dim=-1)

        loss_obj = torch.clamp(self.dhno_margin + hard_neg_obj - pos_scores, min=0.0)
        loss_attr = torch.clamp(self.dhno_margin + hard_neg_attr - pos_scores, min=0.0)

        valid_obj = (hard_neg_obj > -1e8)
        valid_attr = (hard_neg_attr > -1e8)

        dhno_loss = (loss_obj[valid_obj].sum() + loss_attr[valid_attr].sum()) / B
        return dhno_loss

    # ==================================================================
    # Forward
    # ==================================================================

    def train_forward(self, batch, idx):
        batch_img = batch[0].cuda()

        # ---- 1. Encode image (adapter only, no disentangler) ----
        f_global, f_local = self.encode_image(batch_img.type(self.clip.dtype))
        B = f_global.shape[0]

        # ---- 2. f_global을 comp/attr/obj 모두에 사용 ----
        f_global_norm = f_global / f_global.norm(dim=-1, keepdim=True)
        logit_scale = self.clip.logit_scale.exp()

        # ---- 3. Primitive branch (frozen, general) ----
        text_prim = self._encode_text_prim(idx)
        logits_prim_comp = logit_scale * f_global_norm @ text_prim[0].t()
        logits_prim_attr = logit_scale * f_global_norm @ text_prim[1].t()
        logits_prim_obj = logit_scale * f_global_norm @ text_prim[2].t()

        # ---- 4. Contextual branch with VAPS ----
        if f_local is not None:
            shift = self.prompt_shifter(f_local.float())
        else:
            shift = torch.zeros(B, f_global.shape[-1], device=f_global.device)

        shifted_comp_feats, ctx_attr_feat, ctx_obj_feat = \
            self._encode_text_ctx_shifted(idx, shift)

        # Per-image comp logits
        ctx_comp_logits_list = []
        for b in range(B):
            logit_b = logit_scale * f_global_norm[b] @ shifted_comp_feats[b].t()
            ctx_comp_logits_list.append(logit_b)
        logits_ctx_comp = torch.stack(ctx_comp_logits_list, dim=0)

        logits_ctx_attr = logit_scale * f_global_norm @ ctx_attr_feat.t()
        logits_ctx_obj = logit_scale * f_global_norm @ ctx_obj_feat.t()

        # ---- 5. Alpha Gating ----
        alpha = self._compute_alpha(logits_prim_comp, f_global_norm)

        comp_logits = alpha * logits_prim_comp + (1.0 - alpha) * logits_ctx_comp
        attr_logits = logits_ctx_attr
        obj_logits = logits_ctx_obj

        # ---- 6. DHNO on contextual branch ----
        loss_dhno = self._compute_dhno_loss(logits_ctx_comp, batch[3].cuda(), idx)

        return comp_logits, attr_logits, obj_logits, loss_dhno

    def val_forward(self, batch, idx):
        batch_img = batch[0].cuda()

        f_global, f_local = self.encode_image(batch_img.type(self.clip.dtype))
        B = f_global.shape[0]

        f_global_norm = f_global / f_global.norm(dim=-1, keepdim=True)
        logit_scale = self.clip.logit_scale.exp()

        # Primitive branch
        text_prim = self._encode_text_prim(idx)
        logits_prim_comp = logit_scale * f_global_norm @ text_prim[0].t()

        # Contextual branch with VAPS
        if f_local is not None:
            shift = self.prompt_shifter(f_local.float())
        else:
            shift = torch.zeros(B, f_global.shape[-1], device=f_global.device)

        shifted_comp_feats, ctx_attr_feat, ctx_obj_feat = \
            self._encode_text_ctx_shifted(idx, shift)

        ctx_comp_logits_list = []
        for b in range(B):
            logit_b = logit_scale * f_global_norm[b] @ shifted_comp_feats[b].t()
            ctx_comp_logits_list.append(logit_b)
        logits_ctx_comp = torch.stack(ctx_comp_logits_list, dim=0)

        # Alpha gating
        alpha = self._compute_alpha(logits_prim_comp, f_global_norm)
        comp_logits = alpha * logits_prim_comp + (1.0 - alpha) * logits_ctx_comp
        attr_logits = logit_scale * f_global_norm @ ctx_attr_feat.t()
        obj_logits = logit_scale * f_global_norm @ ctx_obj_feat.t()

        return comp_logits, attr_logits, obj_logits

    # ==================================================================
    # Loss & Inference
    # ==================================================================

    def loss_calu(self, predict, target):
        loss_fn = nn.CrossEntropyLoss()
        batch_attr = target[1].cuda()
        batch_obj = target[2].cuda()
        batch_target = target[3].cuda()

        if self.training:
            comp_logits, attr_logits, obj_logits, loss_dhno = predict
        else:
            comp_logits, attr_logits, obj_logits = predict

        loss = (
            self.pair_loss_weight * loss_fn(comp_logits, batch_target) +
            self.attr_loss_weight * loss_fn(attr_logits, batch_attr) +
            self.obj_loss_weight * loss_fn(obj_logits, batch_obj)
        )

        if self.training:
            loss = loss + self.dhno_lambda * loss_dhno

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
