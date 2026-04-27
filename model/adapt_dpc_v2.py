"""
AdaptDPC v2: Visual Adapter + Lightweight Disentanglement + DPC

v1 대비 변경:
- Disentangler 추가 (Linear+BN+ReLU, lightweight)
  → f_attr, f_obj를 명시적으로 분리
  → attr logit은 f_attr로, obj logit은 f_obj로 계산
  → comp logit은 f_global로 계산 (그대로)
- Cosine decorrelation 추가 (HSIC 대체)
  → f_attr과 f_obj의 cosine similarity 절대값 최소화
  → RBF 커널 HSIC 대비 소배치(B=4)에서도 안정적
- Alpha Gating을 comp/attr/obj 세 branch 모두에 적용
  → unseen 조합에서 primitive의 범용적 attr/obj 의미 활용 가능
  → MLP가 [B, 3] 출력 → comp/attr/obj별 독립적 gating

v1에서 유지:
- Visual Adapter (feature quality 개선)
- DPC (dual prompt) — compositional reasoning 전담
- VAPS (f_local → per-image prompt shift)
- DHNO (contextual branch hard negative)

ClusPro 대비 제거된 것:
- Prototype Clustering / Memory Banks
- Prototype Contrastive Loss (NCE)
→ vision쪽 gradient 압력 최소화
→ DPC와의 역할 충돌 방지
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


def cosine_decorrelation(f_attr, f_obj):
    """Cosine-based decorrelation loss.

    배치 내 각 샘플에서 f_attr과 f_obj의 cosine similarity 절대값 평균.
    RBF 커널 기반 HSIC와 달리 소배치(B=4)에서도 안정적.
    0이면 완전 독립, 1이면 완전 상관.
    """
    f_a = F.normalize(f_attr.float(), p=2, dim=-1)
    f_o = F.normalize(f_obj.float(), p=2, dim=-1)
    cos_sim = (f_a * f_o).sum(dim=-1)  # [B]
    return cos_sim.abs().mean()


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
    """Lightweight attr/obj feature separator."""
    def __init__(self, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        return F.dropout(F.relu(self.bn1_fc(self.fc1(x))), training=self.training)


class AdaptDPCv2(nn.Module):
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
        self.use_vaps = getattr(config, 'use_vaps', True)

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
        # [Vision] Adapter + Lightweight Disentanglement
        # ==============================================================

        adapter_dim = getattr(config, 'adapter_dim', 64)
        adapter_dropout = getattr(config, 'adapter_dropout', 0.1)
        self.visual_adapters = nn.ModuleList([
            Adapter(vision_width, adapter_dim, adapter_dropout)
            for _ in range(2 * num_blocks)
        ])

        # Disentanglers: f_global → f_attr, f_obj
        self.attr_disentangler = Disentangler(output_dim)
        self.obj_disentangler = Disentangler(output_dim)

        self.attr_dropout = nn.Dropout(getattr(config, 'attr_dropout', 0.3))

        # HSIC weight
        self.hsic_weight = getattr(config, 'hsic_weight', 0.1)

        # ==============================================================
        # [Text / DPC] Dual Prompt — compositional reasoning 전담
        # ==============================================================

        self.token_ids, base_soft_emb, self.comp_ctx, self.attr_ctx, self.obj_ctx = \
            self._construct_soft_prompt()

        self.soft_att_obj_prim = nn.Parameter(base_soft_emb.clone())
        self.soft_att_obj_ctx = nn.Parameter(base_soft_emb.clone())
        self.comp_ctx = nn.Parameter(self.comp_ctx)
        self.attr_ctx = nn.Parameter(self.attr_ctx)
        self.obj_ctx = nn.Parameter(self.obj_ctx)

        # Dynamic Alpha Gating (comp/attr/obj 각각)
        self.gating_param_comp = nn.Parameter(torch.tensor(0.0))
        self.gating_param_attr = nn.Parameter(torch.tensor(0.0))
        self.gating_param_obj = nn.Parameter(torch.tensor(0.0))
        self.alpha_predictor = nn.Sequential(
            nn.Linear(2 + output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # comp, attr, obj 각각의 alpha
        )

        # ==============================================================
        # [MSCI] f_local from mid-layer ViT block
        # ==============================================================

        self.feature_layer = getattr(config, 'feature_layer', num_blocks // 2)

        # ==============================================================
        # [VAPS] Visual-Adaptive Prompt Shifting
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
            print("[AdaptDPC-v2] Primitive soft embeddings frozen.")

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

            # [MSCI] Capture f_local at mid-layer (only when VAPS enabled)
            if self.use_vaps and i == self.feature_layer:
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

        token_tensor[0][:, eos_idx[0]-2, :] = embs[attr_idx].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0]-1, :] = embs[obj_idx + self.offset].type(self.clip.dtype)
        token_tensor[0][:, 1:len(self.comp_ctx)+1, :] = self.comp_ctx.type(self.clip.dtype)

        token_tensor[1][:, eos_idx[1]-1, :] = embs[:self.offset].type(self.clip.dtype)
        token_tensor[1][:, 1:len(self.attr_ctx)+1, :] = self.attr_ctx.type(self.clip.dtype)

        token_tensor[2][:, eos_idx[2]-1, :] = embs[self.offset:].type(self.clip.dtype)
        token_tensor[2][:, 1:len(self.obj_ctx)+1, :] = self.obj_ctx.type(self.clip.dtype)

        return token_tensor

    # ==================================================================
    # Alpha Gating
    # ==================================================================

    def _compute_alpha(self, logits_prim, f_global_norm):
        """Per-image gating for comp/attr/obj branches.

        Returns:
            alpha_comp: [B, 1]
            alpha_attr: [B, 1]
            alpha_obj:  [B, 1]
        """
        logits_f32 = logits_prim.float()
        log_probs = F.log_softmax(logits_f32, dim=-1)
        entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1, keepdim=True)
        max_entropy = torch.log(torch.tensor(float(logits_prim.size(-1)), device=logits_prim.device))
        norm_entropy = entropy / max_entropy

        max_logit = logits_f32.max(dim=-1, keepdim=True)[0]
        logit_scale = self.clip.logit_scale.exp()
        norm_max_logit = max_logit / logit_scale

        alpha_input = torch.cat([norm_entropy, norm_max_logit, f_global_norm.float()], dim=-1)
        alpha_raw = self.alpha_predictor(alpha_input)  # [B, 3]

        gating_biases = torch.stack([
            self.gating_param_comp, self.gating_param_attr, self.gating_param_obj
        ])  # [3]
        alpha_raw = alpha_raw + gating_biases.float()

        alpha = torch.sigmoid(alpha_raw).to(self.clip.dtype)  # [B, 3]
        alpha = torch.clamp(alpha, min=0.05, max=0.95)

        return alpha[:, 0:1], alpha[:, 1:2], alpha[:, 2:3]

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

    def _encode_text_ctx_static(self, pair_idx):
        """Encode contextual text without VAPS (no per-image shift)."""
        embs = self.attr_dropout(self.soft_att_obj_ctx)
        tokens_ctx = self._construct_token_tensors(pair_idx, embs)
        text_feats = []
        for i in range(self.token_ids.shape[0]):
            feat, _ = self.text_encoder(
                self.token_ids[i], tokens_ctx[i], enable_pos_emb=self.enable_pos_emb
            )
            text_feats.append(feat / feat.norm(dim=-1, keepdim=True))
        return text_feats

    def _encode_text_ctx_shifted(self, pair_idx, shift):
        """[VAPS] Encode contextual text branch with per-image prompt shifting.

        Batched: B개 shifted tensor를 [B*N_pairs, seq, D]로 합쳐서
        text_encoder를 1회만 호출 (기존 B회 → 1회).
        """
        # Static contextual features for attr-only and obj-only branches
        tokens_ctx = self._construct_token_tensors(pair_idx, self.soft_att_obj_ctx)
        static_feats = []
        for i in range(self.token_ids.shape[0]):
            feat, _ = self.text_encoder(
                self.token_ids[i], tokens_ctx[i], enable_pos_emb=self.enable_pos_emb
            )
            static_feats.append(feat / feat.norm(dim=-1, keepdim=True))

        # For comp branch: batched per-image shift
        attr_idx = pair_idx[:, 0]
        obj_idx = pair_idx[:, 1]
        eos_idx = int(self.token_ids[0].argmax())
        B = shift.shape[0]
        N = len(pair_idx)

        base_ids = self.token_ids[0].repeat(N, 1)
        base_tensor = self.clip.token_embedding(base_ids.cuda()).type(self.clip.dtype)

        embs = self.attr_dropout(self.soft_att_obj_ctx)
        base_tensor[:, eos_idx - 2, :] = embs[attr_idx].type(self.clip.dtype)
        base_tensor[:, eos_idx - 1, :] = embs[obj_idx + self.offset].type(self.clip.dtype)
        base_tensor[:, 1:len(self.comp_ctx)+1, :] = self.comp_ctx.type(self.clip.dtype)

        # [B, N, seq, D] — 각 이미지별 shifted token tensor
        shift_typed = shift.to(self.clip.dtype)  # [B, D]
        batched = base_tensor.unsqueeze(0).expand(B, -1, -1, -1).clone()  # [B, N, seq, D]
        batched[:, :, eos_idx - 2, :] += shift_typed[:, None, :]  # broadcast shift
        batched[:, :, eos_idx - 1, :] += shift_typed[:, None, :]

        # Reshape to [B*N, seq, D], chunk to avoid OOM
        batched_flat = batched.reshape(B * N, batched.shape[2], batched.shape[3])
        token_ids_flat = self.token_ids[0].repeat(B * N, 1)

        # 청크 단위로 text encoder 호출 (N개씩 = 이미지 1장 분량)
        chunk_size = N
        feat_chunks = []
        for start in range(0, B * N, chunk_size):
            end = start + chunk_size
            feat_chunk, _ = self.text_encoder(
                token_ids_flat[start:end],
                batched_flat[start:end],
                enable_pos_emb=self.enable_pos_emb
            )
            feat_chunks.append(feat_chunk)
        feat_flat = torch.cat(feat_chunks, dim=0)
        feat_flat = feat_flat / feat_flat.norm(dim=-1, keepdim=True)

        # Reshape back to [B, N, D]
        shifted_comp_feats = feat_flat.reshape(B, N, -1)

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

        # ---- 1. Encode image ----
        f_global, f_local = self.encode_image(batch_img.type(self.clip.dtype))
        B = f_global.shape[0]

        # ---- 2. Disentangle (lightweight) ----
        f_attr = self.attr_disentangler(f_global)
        f_obj = self.obj_disentangler(f_global)

        # ---- 3. Cosine decorrelation (소배치 안정적) ----
        loss_decorr = cosine_decorrelation(f_attr, f_obj)

        # ---- 4. Normalize features ----
        f_global_norm = f_global / f_global.norm(dim=-1, keepdim=True)
        f_attr_norm = f_attr / f_attr.norm(dim=-1, keepdim=True)
        f_obj_norm = f_obj / f_obj.norm(dim=-1, keepdim=True)
        logit_scale = self.clip.logit_scale.exp()

        # ---- 5. Primitive branch (frozen, general) ----
        text_prim = self._encode_text_prim(idx)
        logits_prim_comp = logit_scale * f_global_norm @ text_prim[0].t()
        logits_prim_attr = logit_scale * f_attr_norm @ text_prim[1].t()
        logits_prim_obj = logit_scale * f_obj_norm @ text_prim[2].t()

        # ---- 6. Contextual branch ----
        if self.use_vaps:
            if f_local is not None:
                shift = self.prompt_shifter(f_local.float())
            else:
                shift = torch.zeros(B, f_global.shape[-1], device=f_global.device)
            shifted_comp_feats, ctx_attr_feat, ctx_obj_feat = \
                self._encode_text_ctx_shifted(idx, shift)
            logits_ctx_comp = logit_scale * torch.bmm(
                f_global_norm.unsqueeze(1), shifted_comp_feats.transpose(1, 2)
            ).squeeze(1)  # [B, N]
        else:
            text_ctx = self._encode_text_ctx_static(idx)
            logits_ctx_comp = logit_scale * f_global_norm @ text_ctx[0].t()
            ctx_attr_feat = text_ctx[1]
            ctx_obj_feat = text_ctx[2]

        logits_ctx_attr = logit_scale * f_attr_norm @ ctx_attr_feat.t()
        logits_ctx_obj = logit_scale * f_obj_norm @ ctx_obj_feat.t()

        # ---- 7. Alpha Gating (comp/attr/obj 각각) ----
        alpha_comp, alpha_attr, alpha_obj = self._compute_alpha(logits_prim_comp, f_global_norm)
        comp_logits = alpha_comp * logits_prim_comp + (1.0 - alpha_comp) * logits_ctx_comp
        attr_logits = alpha_attr * logits_prim_attr + (1.0 - alpha_attr) * logits_ctx_attr
        obj_logits = alpha_obj * logits_prim_obj + (1.0 - alpha_obj) * logits_ctx_obj

        # ---- 8. DHNO on contextual comp branch ----
        loss_dhno = self._compute_dhno_loss(logits_ctx_comp, batch[3].cuda(), idx)

        return comp_logits, attr_logits, obj_logits, loss_decorr, loss_dhno

    def val_forward(self, batch, idx):
        batch_img = batch[0].cuda()

        f_global, f_local = self.encode_image(batch_img.type(self.clip.dtype))
        B = f_global.shape[0]

        # Disentangle
        f_attr = self.attr_disentangler(f_global)
        f_obj = self.obj_disentangler(f_global)

        f_global_norm = f_global / f_global.norm(dim=-1, keepdim=True)
        f_attr_norm = f_attr / f_attr.norm(dim=-1, keepdim=True)
        f_obj_norm = f_obj / f_obj.norm(dim=-1, keepdim=True)
        logit_scale = self.clip.logit_scale.exp()

        # Primitive branch
        text_prim = self._encode_text_prim(idx)
        logits_prim_comp = logit_scale * f_global_norm @ text_prim[0].t()
        logits_prim_attr = logit_scale * f_attr_norm @ text_prim[1].t()
        logits_prim_obj = logit_scale * f_obj_norm @ text_prim[2].t()

        # Contextual branch
        if self.use_vaps:
            if f_local is not None:
                shift = self.prompt_shifter(f_local.float())
            else:
                shift = torch.zeros(B, f_global.shape[-1], device=f_global.device)
            shifted_comp_feats, ctx_attr_feat, ctx_obj_feat = \
                self._encode_text_ctx_shifted(idx, shift)
            logits_ctx_comp = logit_scale * torch.bmm(
                f_global_norm.unsqueeze(1), shifted_comp_feats.transpose(1, 2)
            ).squeeze(1)
        else:
            text_ctx = self._encode_text_ctx_static(idx)
            logits_ctx_comp = logit_scale * f_global_norm @ text_ctx[0].t()
            ctx_attr_feat = text_ctx[1]
            ctx_obj_feat = text_ctx[2]

        logits_ctx_attr = logit_scale * f_attr_norm @ ctx_attr_feat.t()
        logits_ctx_obj = logit_scale * f_obj_norm @ ctx_obj_feat.t()

        # Alpha gating (comp/attr/obj 각각)
        alpha_comp, alpha_attr, alpha_obj = self._compute_alpha(logits_prim_comp, f_global_norm)
        comp_logits = alpha_comp * logits_prim_comp + (1.0 - alpha_comp) * logits_ctx_comp
        attr_logits = alpha_attr * logits_prim_attr + (1.0 - alpha_attr) * logits_ctx_attr
        obj_logits = alpha_obj * logits_prim_obj + (1.0 - alpha_obj) * logits_ctx_obj

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
            comp_logits, attr_logits, obj_logits, loss_decorr, loss_dhno = predict
        else:
            comp_logits, attr_logits, obj_logits = predict

        loss = (
            self.pair_loss_weight * loss_fn(comp_logits, batch_target) +
            self.attr_loss_weight * loss_fn(attr_logits, batch_attr) +
            self.obj_loss_weight * loss_fn(obj_logits, batch_obj)
        )

        if self.training:
            loss = loss + self.hsic_weight * loss_decorr + self.dhno_lambda * loss_dhno

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
