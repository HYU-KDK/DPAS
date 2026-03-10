import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dpc_alpha import DPCAlphaInterface


class DPCNextInterface(DPCAlphaInterface):
    """
    DPC-Next: SOTA CZSL architecture.

    Key innovations over DPC-Alpha:
    1. Multi-stage visual feature extraction:
       - f_global: final CLIP layer features (holistic/semantic)
       - f_local:  mid-layer features (texture/fine-grained local)
    2. Dynamic Prompt Shifting (Adapter):
       - f_local drives a lightweight MLP that shifts contextual prompt embeddings
       - P_dynamic = P_anchor + Adapter(f_local)
       - Each image gets its own personalized text prompt!
    3. Advanced Alpha Predictor:
       - Combines entropy, confidence (max logit), AND global visual features
       - Much richer signal than pure entropy-based gating
    """

    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_embeddings_primitive,
        soft_embeddings_contextual,
        class_token_ids,
        device="cuda:0",
        enable_pos_emb=True,
        attr_dropout=0.0,
        feature_layer=12,  # Mid-layer index (6 or 12 for ViT-L/14)
    ):
        super().__init__(
            clip_model,
            config,
            offset,
            soft_embeddings_primitive,
            soft_embeddings_contextual,
            class_token_ids,
            device=device,
            enable_pos_emb=enable_pos_emb,
            attr_dropout=attr_dropout,
        )

        self.feature_layer = feature_layer
        self._local_feat_cache = {}

        # -------------------------------------------------------
        # Figure out dimensions from the CLIP model
        # -------------------------------------------------------
        # vision_width: internal ViT transformer width (e.g. 1024 for L/14)
        # embed_dim:    the joint image-text embedding size output by encode_image/encode_text
        #               (e.g. 768 for ViT-L/14 after the visual projection)
        vision_width = clip_model.visual.class_embedding.shape[0]   # 1024 (internal)
        embed_dim    = clip_model.text_projection.shape[1]           # 768  (output)

        # -------------------------------------------------------
        # 1. Prompt Shifter: f_local (internal) → shift vector (embed_dim)
        #    f_local comes from the mid-ViT hook → dimension = vision_width (1024)
        # -------------------------------------------------------
        self.prompt_shifter = nn.Sequential(
            nn.Linear(vision_width, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        ).to(device)

        # -------------------------------------------------------
        # 2. Advanced Alpha Predictor: [entropy, conf, f_global] → α
        #    f_global comes from encode_image → dimension = embed_dim (768)
        # -------------------------------------------------------
        self.advanced_alpha_predictor = nn.Sequential(
            nn.Linear(2 + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)

        # -------------------------------------------------------
        # Register forward hook on target ViT block
        # -------------------------------------------------------
        self._register_local_hook()

    # ------------------------------------------------------------------
    # Hook plumbing
    # ------------------------------------------------------------------

    def _register_local_hook(self):
        """Attach a hook to capture the CLS token at a mid ViT block."""
        def hook_fn(module, inp, out):
            # ViT-L/14 (clip-pytorch): output shape is [Seq, Batch, D]
            # CLS token is at position 0
            self._local_feat_cache['local'] = out[0].detach()  # [Batch, D]

        try:
            block = self.clip_model.visual.transformer.resblocks[self.feature_layer]
            self._hook_handle = block.register_forward_hook(hook_fn)
            print(f"[DPC-Next] Registered visual hook on ViT block {self.feature_layer}.")
        except Exception as e:
            self._hook_handle = None
            print(f"[DPC-Next] Warning: Could not register visual hook: {e}")

    # ------------------------------------------------------------------
    # Dynamic token construction
    # ------------------------------------------------------------------

    def _build_dynamic_token_tensors(self, pair_idx, shift):
        """
        Build per-image contextual token tensors shifted by `shift`.

        Args:
            pair_idx   : LongTensor [N_pairs, 2] — all training pairs
            shift      : FloatTensor [Batch, D_embed] — per-image shift vectors

        Returns:
            token_tensors_ctx : Tensor [Batch, N_pairs, Seq, D_embed]
                                — one prompt set per image
        """
        attr_idx = pair_idx[:, 0]   # [N_pairs]
        obj_idx  = pair_idx[:, 1]   # [N_pairs]

        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)   # [N_pairs, Seq]
        base_tensors = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)                                # [N_pairs, Seq, D]

        eos_idx = int(self.token_ids[0].argmax())

        # Static contextual embeddings for each pair
        ctx_embs = self.attr_dropout(self.soft_embeddings_contextual)
        attr_embs = ctx_embs[attr_idx]                         # [N_pairs, D]
        obj_embs  = ctx_embs[obj_idx + self.offset]            # [N_pairs, D]

        # base_tensors: [N_pairs, Seq, D]
        base_tensors[:, eos_idx - 2, :] = attr_embs
        base_tensors[:, eos_idx - 1, :] = obj_embs
        # → base_tensors now has static ctx prompts: [N_pairs, Seq, D]

        # Expand for batch: [Batch, N_pairs, Seq, D]
        batch_size = shift.shape[0]
        token_tensors = base_tensors.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()

        # Apply per-image shift to the two slot positions
        shift_typed = shift.to(self.clip_model.dtype)           # [Batch, D]
        # shift_typed[:, None, :] → [Batch, 1, D] broadcasts over N_pairs
        token_tensors[:, :, eos_idx - 2, :] += shift_typed[:, None, :]
        token_tensors[:, :, eos_idx - 1, :] += shift_typed[:, None, :]

        return token_tensors   # [Batch, N_pairs, Seq, D]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, imgs, pair_idx):
        """
        Args:
            imgs     : [Batch, C, H, W]  — raw images (NOT pre-encoded features!)
            pair_idx : [N_pairs, 2]       — all training/eval pairs
        """
        imgs = imgs.to(self.device)

        # ---- 1. Multi-stage visual feature extraction ----
        # Calling encode_image triggers the hook → _local_feat_cache['local']
        f_global = self.clip_model.encode_image(imgs)   # [Batch, D_vision]
        f_local  = self._local_feat_cache.get('local', f_global)  # [Batch, D_vision]

        f_global_norm = f_global / f_global.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        # ---- 2. Primitive Branch (static CSP anchor) ----
        token_tensor_prim, _ = super().construct_token_tensors(pair_idx)  # [N_pairs, Seq, D]
        prim_text_feat = self.text_encoder(
            self.token_ids, token_tensor_prim, enable_pos_emb=self.enable_pos_emb
        )
        prim_text_feat = prim_text_feat / prim_text_feat.norm(dim=-1, keepdim=True)
        # logits_prim: [Batch, N_pairs]
        logits_prim = logit_scale * f_global_norm @ prim_text_feat.t()

        # ---- 3. Contextual Branch with Dynamic Prompt Shifting ----
        # Compute per-image shift from local features
        shift = self.prompt_shifter(f_local.float())                 # [Batch, D_embed]

        # Build per-image token tensors: [Batch, N_pairs, Seq, D_embed]
        dyn_token_tensors = self._build_dynamic_token_tensors(pair_idx, shift)

        # Encode each image's personalized prompts
        # Process in a loop over batch to stay memory-efficient
        batch_size = imgs.shape[0]
        ctx_logits_list = []
        for b in range(batch_size):
            ctx_feat_b = self.text_encoder(
                self.token_ids, dyn_token_tensors[b], enable_pos_emb=self.enable_pos_emb
            )                                                        # [N_pairs, D_text]
            ctx_feat_b = ctx_feat_b / ctx_feat_b.norm(dim=-1, keepdim=True)
            logit_b = logit_scale * f_global_norm[b] @ ctx_feat_b.t()  # [N_pairs]
            ctx_logits_list.append(logit_b)
        logits_ctx = torch.stack(ctx_logits_list, dim=0)            # [Batch, N_pairs]

        # ---- 4. Advanced Alpha (entropy + confidence + visual) ----
        logits_prim_f32 = logits_prim.float()
        log_probs = F.log_softmax(logits_prim_f32, dim=-1)
        entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1, keepdim=True)  # [Batch,1]
        max_entropy = torch.log(torch.tensor(float(logits_prim.size(-1)), device=self.device))
        norm_entropy = entropy / max_entropy

        max_logit = logits_prim_f32.max(dim=-1, keepdim=True)[0]
        norm_max_logit = max_logit / logit_scale

        alpha_input = torch.cat(
            [norm_entropy, norm_max_logit, f_global_norm.float()], dim=-1
        )                                                            # [Batch, 2+D_vision]
        alpha_raw = self.gating_param.float() + self.advanced_alpha_predictor.float()(alpha_input)
        alpha = torch.sigmoid(alpha_raw).to(self.clip_model.dtype)
        alpha = torch.clamp(alpha, min=0.05, max=0.95)              # [Batch, 1]

        # ---- 5. Fusion ----
        final_logits = alpha * logits_prim + (1.0 - alpha) * logits_ctx
        return final_logits / self.temp


# -------------------------------------------------------------------------
# Factory function
# -------------------------------------------------------------------------

def get_dpc_next(train_dataset, config, device):
    from models.dpc import dpc_init

    csp_init_path  = getattr(config, 'csp_init_path',   None)
    freeze_prim    = getattr(config, 'freeze_primitive', False)
    feature_layer  = getattr(config, 'dpc_next_feature_layer', 12)

    clip_model, soft_prim, soft_ctx, token_ids, offset = dpc_init(
        train_dataset, config, device, csp_checkpoint_path=csp_init_path
    )

    interface = DPCNextInterface(
        clip_model, config, offset,
        soft_prim, soft_ctx, token_ids,
        device=device,
        attr_dropout=config.attr_dropout,
        feature_layer=feature_layer,
    )

    params = []
    if freeze_prim:
        print("Freezing Primitive soft embeddings (CSP Anchor).")
        soft_prim.requires_grad = False
        params.append({'params': [soft_ctx]})
    else:
        params.append({'params': [soft_prim, soft_ctx]})

    # Gating + Alpha Predictor + Prompt Shifter — higher LR
    params.append({
        'params': (
            [interface.gating_param]
            + list(interface.advanced_alpha_predictor.parameters())
            + list(interface.prompt_shifter.parameters())
        ),
        'lr': 0.001,
    })

    optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    return interface, optimizer
