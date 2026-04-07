"""
ClusDPC: ClusPro + DPC + MSCI + VAPS Fusion Model

ClusPro의 비전 강화 (Visual Adapter + Disentangler + Prototype Clustering)
+ DPC의 텍스트 강화 (Dual Soft Prompt + Dynamic Alpha Gating)
+ MSCI (Multi-Stage CLIP Image features: f_global + f_local)
+ VAPS (Visual-Adaptive Prompt Shifting: f_local → prompt shift)
+ DHNO (Dual Hard Negative Optimization)
= 양쪽 modality를 모두 적응시키는 4중 적응 CZSL 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import CustomTextEncoder
from model.nce_loss import ContrastiveLoss
from model.otgcc import local_assign
from model.hsic import hsic_normalized


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


# =====================================================================
# Building Blocks (from ClusPro)
# =====================================================================

class Adapter(nn.Module):
    """LoRA-style adapter for ViT blocks."""
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
    """Disentangles global image features into attr/obj subspaces."""
    def __init__(self, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        return F.dropout(F.relu(self.bn1_fc(self.fc1(x))), training=self.training)


# =====================================================================
# ClusDPC Model
# =====================================================================

class ClusDPC(nn.Module):
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
        output_dim = self.clip.visual.output_dim  # e.g. 512 for B/16, 768 for L/14

        # ==============================================================
        # [ClusPro Components] Vision-side enhancement
        # ==============================================================

        # 1. Visual Adapters (one per MHA + one per FFN, per block)
        num_blocks = self.clip.visual.transformer.layers
        vision_width = self.clip.visual.transformer.width
        adapter_dim = getattr(config, 'adapter_dim', 64)
        adapter_dropout = getattr(config, 'adapter_dropout', 0.1)
        self.visual_adapters = nn.ModuleList([
            Adapter(vision_width, adapter_dim, adapter_dropout)
            for _ in range(2 * num_blocks)
        ])

        # 2. Attr/Obj Disentanglers
        self.attr_disentangler = Disentangler(output_dim)
        self.obj_disentangler = Disentangler(output_dim)
        # Post-disentangle projectors (for contrastive space)
        self.attr_proj = Disentangler(output_dim)
        self.obj_proj = Disentangler(output_dim)

        # 3. Prototype memory banks (K prototypes per primitive)
        self.cluster_num = getattr(config, 'cluster_num', 5)
        self.momentum = getattr(config, 'proto_momentum', 0.99)
        self.queue_len = getattr(config, 'queue_len', 5)

        for i in range(len(self.attributes)):
            self.register_buffer(f"attr_queue{i}", torch.randn(self.cluster_num, output_dim))
        for i in range(len(self.classes)):
            self.register_buffer(f"obj_queue{i}", torch.randn(self.cluster_num, output_dim))

        # Feature queues for momentum update
        n_attr_clusters = len(self.attributes) * self.cluster_num
        n_obj_clusters = len(self.classes) * self.cluster_num
        self.register_buffer("attr_feat_queue",
            F.normalize(torch.randn(n_attr_clusters, self.queue_len, output_dim), dim=-1))
        self.register_buffer("attr_feat_ptr",
            torch.zeros(n_attr_clusters, dtype=torch.long))
        self.register_buffer("obj_feat_queue",
            F.normalize(torch.randn(n_obj_clusters, self.queue_len, output_dim), dim=-1))
        self.register_buffer("obj_feat_ptr",
            torch.zeros(n_obj_clusters, dtype=torch.long))

        # 4. Contrastive loss
        self.nceloss = ContrastiveLoss()
        self.attr_dropout = nn.Dropout(getattr(config, 'attr_dropout', 0.3))

        # ==============================================================
        # [DPC Components] Text-side enhancement
        # ==============================================================

        # 5. Dual Soft Embeddings: Primitive (generalist) + Contextual (specialist)
        #    Shared across comp/attr/obj prompts
        self.token_ids, base_soft_emb, self.comp_ctx, self.attr_ctx, self.obj_ctx = \
            self._construct_soft_prompt()

        self.soft_att_obj_prim = nn.Parameter(base_soft_emb.clone())
        self.soft_att_obj_ctx = nn.Parameter(base_soft_emb.clone())
        self.comp_ctx = nn.Parameter(self.comp_ctx)
        self.attr_ctx = nn.Parameter(self.attr_ctx)
        self.obj_ctx = nn.Parameter(self.obj_ctx)

        # 6. Dynamic Alpha Gating
        #    Uses entropy + confidence + disentangled features
        self.gating_param = nn.Parameter(torch.tensor(0.0))
        self.alpha_predictor = nn.Sequential(
            nn.Linear(2 + output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # ---- Loss weights ----
        self.pair_loss_weight = getattr(config, 'pair_loss_weight', 1.0)
        self.attr_loss_weight = getattr(config, 'attr_loss_weight', 1.0)
        self.obj_loss_weight = getattr(config, 'obj_loss_weight', 1.0)
        self.contrastive_weight = getattr(config, 'contrastive_weight', 0.1)
        self.hsic_weight = getattr(config, 'hsic_weight', 0.1)

        # ==============================================================
        # [MSCI] Multi-Stage CLIP Image features
        # ==============================================================

        # 7. f_local: mid-layer ViT feature via manual capture in encode_image
        self.feature_layer = getattr(config, 'feature_layer', num_blocks // 2)
        self._local_feat_cache = {}

        # ==============================================================
        # [VAPS] Visual-Adaptive Prompt Shifting
        # ==============================================================

        # 8. Prompt Shifter: f_local (vision_width) → shift (output_dim)
        #    Shifts contextual branch token embeddings per-image
        self.prompt_shifter = nn.Sequential(
            nn.Linear(vision_width, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

        # 9. Freeze primitive soft embeddings (VAPS 3a)
        freeze_primitive = getattr(config, 'freeze_primitive', True)
        if freeze_primitive:
            self.soft_att_obj_prim.requires_grad = False
            print("[ClusDPC] Primitive soft embeddings frozen (VAPS).")

        # ==============================================================
        # [DHNO] Dual Hard Negative Optimization
        # ==============================================================

        # 10. DHNO loss parameters
        self.dhno_lambda = getattr(config, 'dhno_lambda', 0.3)
        self.dhno_margin = getattr(config, 'dhno_margin', 1.0)

        # ---- Inference fusion weights ----
        self.pair_inf_w = getattr(config, 'pair_inference_weight', 1.0)
        self.attr_inf_w = getattr(config, 'attr_inference_weight', 1.0)
        self.obj_inf_w = getattr(config, 'obj_inference_weight', 1.0)

    # ==================================================================
    # CLIP Encoding with Adapter
    # ==================================================================

    def encode_image(self, x):
        """Encode image with adapter + MSCI (captures f_local at mid-layer)."""
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

            # [MSCI] Capture f_local at mid-layer (CLS token)
            if i == self.feature_layer:
                f_local = x[0].detach()  # x is LND, [0] = CLS token → [B, D_vision]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.visual.ln_post(x)
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        # f_local stays in vision_width space (pre-projection) for prompt_shifter
        f_global = x[:, 0, :]  # [B, output_dim]
        self._local_feat_cache['local'] = f_local  # [B, vision_width]

        return f_global, x  # cls_token, all_patches

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

        # comp: [attr] [obj]
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
    # Prototype Update (ClusPro style)
    # ==================================================================

    @torch.no_grad()
    @torch.autocast(device_type='cuda', enabled=False)
    def _update_prototypes(self, batch_attr, batch_obj, attr_idx, obj_idx):
        """Momentum-update cluster prototypes based on current batch."""
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
        """Get cluster assignment labels for contrastive loss."""
        B = batch_feat.shape[0]
        labels = torch.zeros(B, dtype=torch.long, device=batch_feat.device)
        for k in range(len(primitives)):
            mask = (idx_labels == k)
            if mask.sum() == 0:
                continue
            queue_k = getattr(self, f"{prim_type}_queue{k}")  # [K, D]
            sim = l2_normalize(batch_feat[mask]) @ l2_normalize(queue_k).t()  # [n_k, K]
            cluster_idx = sim.argmax(dim=-1)  # [n_k]
            labels[mask] = cluster_idx + k * self.cluster_num
        return labels

    def _gather_all_prototypes(self, prim_type, primitives):
        """Gather all prototypes: [N_prim, K, D]."""
        protos = []
        for k in range(len(primitives)):
            protos.append(getattr(self, f"{prim_type}_queue{k}"))
        return torch.stack(protos, dim=0)  # [N_prim, K, D]

    # ==================================================================
    # Dynamic Alpha Gating (DPC-Alpha style)
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
    # Text Encoding Helper
    # ==================================================================

    def _encode_text_prim(self, pair_idx):
        """Encode primitive (static) text branch — shared across all images."""
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

        The shift vector modifies attr/obj token positions in the comp prompt,
        making each image's text representation unique.

        Args:
            pair_idx: [N_pairs, 2]
            shift:    [B, output_dim] — per-image shift from prompt_shifter(f_local)

        Returns:
            For comp branch: [B, N_pairs] logits (per-image, shifted)
            For attr/obj branches: standard (no per-image shift, just contextual)
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
        # Build shifted token tensors for each image
        attr_idx = pair_idx[:, 0]
        obj_idx = pair_idx[:, 1]
        eos_idx = int(self.token_ids[0].argmax())
        B = shift.shape[0]
        N_pairs = len(pair_idx)

        # Base comp token tensor
        base_ids = self.token_ids[0].repeat(N_pairs, 1)
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
            shifted[:, eos_idx - 2, :] += shift_typed[b]  # shift attr slot
            shifted[:, eos_idx - 1, :] += shift_typed[b]  # shift obj slot
            feat, _ = self.text_encoder(
                self.token_ids[0], shifted, enable_pos_emb=self.enable_pos_emb
            )
            shifted_comp_feats.append(feat / feat.norm(dim=-1, keepdim=True))

        return shifted_comp_feats, static_feats[1], static_feats[2]

    # ==================================================================
    # [DHNO] Dual Hard Negative Optimization
    # ==================================================================

    def _compute_dhno_loss(self, comp_logits, batch_target, train_pairs):
        """DHNO contrastive loss on the dynamic (contextual) comp branch.

        For each sample, finds hard negatives that share the same object
        or same attribute, and applies triplet margin loss.

        Args:
            comp_logits:  [B, N_pairs]
            batch_target: [B] — pair index ground truth
            train_pairs:  [N_pairs, 2]
        """
        B = len(batch_target)
        N = train_pairs.shape[0]
        attrs = train_pairs[:, 0]
        objs = train_pairs[:, 1]

        # Pre-compute masks: [N, N]
        mask_same_obj = (objs.unsqueeze(1) == objs.unsqueeze(0))
        mask_same_attr = (attrs.unsqueeze(1) == attrs.unsqueeze(0))
        eye = torch.eye(N, device=comp_logits.device).bool()
        mask_same_obj = mask_same_obj & (~eye)
        mask_same_attr = mask_same_attr & (~eye)

        # Current batch masks: [B, N]
        cur_mask_obj = mask_same_obj[batch_target]
        cur_mask_attr = mask_same_attr[batch_target]

        # Positive scores: [B]
        pos_scores = comp_logits[torch.arange(B, device=comp_logits.device), batch_target]

        # Hard neg (same obj): max logit among same-obj pairs
        obj_logits = comp_logits.clone()
        obj_logits[~cur_mask_obj] = -1e9
        hard_neg_obj, _ = obj_logits.max(dim=-1)

        # Hard neg (same attr)
        attr_logits = comp_logits.clone()
        attr_logits[~cur_mask_attr] = -1e9
        hard_neg_attr, _ = attr_logits.max(dim=-1)

        # Triplet margin loss
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
        attr_idx, obj_idx = batch[1], batch[2]

        # ---- 1. Encode image with adapter + MSCI ----
        f_global, f_patches = self.encode_image(batch_img.type(self.clip.dtype))
        f_local = self._local_feat_cache.get('local', None)
        B = f_global.shape[0]

        # ---- 2. Disentangle (ClusPro) ----
        f_attr = self.attr_disentangler(f_global)
        f_obj = self.obj_disentangler(f_global)

        # ---- 3. Update prototypes (ClusPro) ----
        self._update_prototypes(f_attr, f_obj, attr_idx, obj_idx)

        # ---- 4. Cluster labels for contrastive ----
        attr_labels = self._get_cluster_labels(f_attr, attr_idx, self.attributes, "attr")
        obj_labels = self._get_cluster_labels(f_obj, obj_idx, self.classes, "obj")

        # ---- 5. Contrastive loss (ClusPro) ----
        attr_protos = self._gather_all_prototypes("attr", self.attributes)
        obj_protos = self._gather_all_prototypes("obj", self.classes)

        f_attr_proj = self.attr_proj(f_attr)
        f_obj_proj = self.obj_proj(f_obj)

        all_protos_flat = torch.cat([
            self.attr_proj(attr_protos.reshape(-1, attr_protos.shape[-1])).reshape(-1, attr_protos.shape[-1]),
            self.obj_proj(obj_protos.reshape(-1, obj_protos.shape[-1])).reshape(-1, obj_protos.shape[-1]),
        ], dim=0)

        attr_all_protos = all_protos_flat.unsqueeze(0).expand(B, -1, -1)
        obj_all_protos = all_protos_flat.unsqueeze(0).expand(B, -1, -1)

        loss_contrastive = (
            self.nceloss(f_attr_proj, attr_all_protos, attr_labels) +
            self.nceloss(f_obj_proj, obj_all_protos, obj_labels)
        )

        # ---- 6. HSIC decorrelation (ClusPro) ----
        attr_proto_selected = torch.stack([
            getattr(self, f"attr_queue{int(attr_idx[i])}")[
                attr_labels[i] % self.cluster_num
            ] for i in range(B)
        ])
        obj_proto_selected = torch.stack([
            getattr(self, f"obj_queue{int(obj_idx[i])}")[
                obj_labels[i] % self.cluster_num
            ] for i in range(B)
        ])

        loss_hsic = (
            hsic_normalized(f_attr_proj, obj_proto_selected.detach()) +
            0.5 * hsic_normalized(f_obj_proj, attr_proto_selected.detach())
        )

        # ---- 7. Primitive branch (DPC — static, frozen) ----
        text_prim = self._encode_text_prim(idx)

        img_feats = [f_global, f_attr_proj, f_obj_proj]
        norm_img = [f / f.norm(dim=-1, keepdim=True) for f in img_feats]
        logit_scale = self.clip.logit_scale.exp()

        logits_prim_comp = logit_scale * norm_img[0] @ text_prim[0].t()
        logits_prim_attr = logit_scale * norm_img[1] @ text_prim[1].t()
        logits_prim_obj = logit_scale * norm_img[2] @ text_prim[2].t()

        # ---- 8. Contextual branch with VAPS (dynamic, shifted) ----
        if f_local is not None:
            shift = self.prompt_shifter(f_local.float())  # [B, output_dim]
        else:
            shift = torch.zeros(B, f_global.shape[-1], device=f_global.device)

        shifted_comp_feats, ctx_attr_feat, ctx_obj_feat = \
            self._encode_text_ctx_shifted(idx, shift)

        # Per-image comp logits for contextual branch
        ctx_comp_logits_list = []
        for b in range(B):
            logit_b = logit_scale * norm_img[0][b] @ shifted_comp_feats[b].t()
            ctx_comp_logits_list.append(logit_b)
        logits_ctx_comp = torch.stack(ctx_comp_logits_list, dim=0)  # [B, N_pairs]

        logits_ctx_attr = logit_scale * norm_img[1] @ ctx_attr_feat.t()
        logits_ctx_obj = logit_scale * norm_img[2] @ ctx_obj_feat.t()

        # ---- 9. Dynamic Alpha Gating (DPC-Alpha) ----
        alpha = self._compute_alpha(logits_prim_comp, norm_img[0])  # [B, 1]

        comp_logits = alpha * logits_prim_comp + (1.0 - alpha) * logits_ctx_comp
        attr_logits = logits_ctx_attr
        obj_logits = logits_ctx_obj

        # ---- 10. DHNO loss on dynamic branch ----
        loss_dhno = self._compute_dhno_loss(logits_ctx_comp, batch[3].cuda(), idx)

        return comp_logits, attr_logits, obj_logits, loss_contrastive, loss_hsic, loss_dhno

    def val_forward(self, batch, idx):
        batch_img = batch[0].cuda()

        # ---- 1. Encode image ----
        f_global, _ = self.encode_image(batch_img.type(self.clip.dtype))
        f_local = self._local_feat_cache.get('local', None)
        B = f_global.shape[0]

        f_attr = self.attr_disentangler(f_global)
        f_obj = self.obj_disentangler(f_global)
        f_attr_proj = self.attr_proj(f_attr)
        f_obj_proj = self.obj_proj(f_obj)

        img_feats = [f_global, f_attr_proj, f_obj_proj]
        norm_img = [f / f.norm(dim=-1, keepdim=True) for f in img_feats]
        logit_scale = self.clip.logit_scale.exp()

        # ---- 2. Primitive branch ----
        text_prim = self._encode_text_prim(idx)
        logits_prim_comp = logit_scale * norm_img[0] @ text_prim[0].t()

        # ---- 3. Contextual branch with VAPS ----
        if f_local is not None:
            shift = self.prompt_shifter(f_local.float())
        else:
            shift = torch.zeros(B, f_global.shape[-1], device=f_global.device)

        shifted_comp_feats, ctx_attr_feat, ctx_obj_feat = \
            self._encode_text_ctx_shifted(idx, shift)

        ctx_comp_logits_list = []
        for b in range(B):
            logit_b = logit_scale * norm_img[0][b] @ shifted_comp_feats[b].t()
            ctx_comp_logits_list.append(logit_b)
        logits_ctx_comp = torch.stack(ctx_comp_logits_list, dim=0)

        # ---- 4. Alpha gating ----
        alpha = self._compute_alpha(logits_prim_comp, norm_img[0])
        comp_logits = alpha * logits_prim_comp + (1.0 - alpha) * logits_ctx_comp
        attr_logits = logit_scale * norm_img[1] @ ctx_attr_feat.t()
        obj_logits = logit_scale * norm_img[2] @ ctx_obj_feat.t()

        return comp_logits, attr_logits, obj_logits

    # ==================================================================
    # Loss & Inference
    # ==================================================================

    def loss_calu(self, predict, target):
        loss_fn = nn.CrossEntropyLoss()
        batch_attr, batch_obj, batch_target = target[1], target[2], target[3]
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()

        if self.training:
            comp_logits, attr_logits, obj_logits, loss_contras, loss_hsic, loss_dhno = predict
        else:
            comp_logits, attr_logits, obj_logits = predict

        loss = (
            self.pair_loss_weight * loss_fn(comp_logits, batch_target) +
            self.attr_loss_weight * loss_fn(attr_logits, batch_attr) +
            self.obj_loss_weight * loss_fn(obj_logits, batch_obj)
        )

        if self.training:
            loss = (loss
                    + self.contrastive_weight * loss_contras
                    + self.hsic_weight * loss_hsic
                    + self.dhno_lambda * loss_dhno)

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
