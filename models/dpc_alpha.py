import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dpc import DPCInterface

class DPCAlphaInterface(DPCInterface):
    """
    DPC Variant with Entropy-based Dynamic Gating.
    Adjusts alpha based on the uncertainty (entropy) of the Primitive branch.
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
        
        # Small MLP to predict alpha from entropy and confidence (max logit)
        self.alpha_predictor = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ).to(device)
        
        # Logit temperature scaling (default 1.0)
        self.temp = getattr(config, 'temp', 1.0)

    def forward(self, batch_img, idx):
        # batch_img: Normalized image features or raw images
        # Ensure device
        batch_img = batch_img.to(self.device)
        
        # 1. Get both token tensors
        token_tensor_prim, token_tensor_ctx = self.construct_token_tensors(idx)

        # 2. Encode both prompts
        # Primitive Features
        prim_text_features = self.text_encoder(
            self.token_ids,
            token_tensor_prim,
            enable_pos_emb=self.enable_pos_emb,
        )
        prim_text_features = prim_text_features / prim_text_features.norm(dim=-1, keepdim=True)

        # Contextual Features
        ctx_text_features = self.text_encoder(
            self.token_ids,
            token_tensor_ctx,
            enable_pos_emb=self.enable_pos_emb,
        )
        ctx_text_features = ctx_text_features / ctx_text_features.norm(dim=-1, keepdim=True)

        # 3. Image Feature Normalization (if not already)
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        # 4. Compute Primitive Logits
        logits_prim = logit_scale * normalized_img @ prim_text_features.t()

        # 5. Energy/Entropy-based Dynamic Alpha
        # Perform gating calculations in Float32 for stability
        logits_prim_f32 = logits_prim.float()
        log_probs_prim = F.log_softmax(logits_prim_f32, dim=-1)
        entropy = -torch.sum(torch.exp(log_probs_prim) * log_probs_prim, dim=-1, keepdim=True)
        
        # Normalize entropy by log(N_pairs) to range [0, 1] approx
        num_classes = logits_prim.size(-1)
        max_entropy = torch.log(torch.tensor(float(num_classes), device=self.device))
        normalized_entropy = entropy / max_entropy

        # Dynamic Gating: Alpha = sigmoid(base + MLP(entropy, max_logit))
        # Add max logit (confidence) as second feature
        max_logit = torch.max(logits_prim_f32, dim=-1, keepdim=True)[0]
        # Normalize max_logit roughly (it depends on logit_scale (~30-100))
        # normalized_max_logit = max_logit / logit_scale
        
        alpha_input_tensor = torch.cat([normalized_entropy, max_logit / logit_scale], dim=-1)
        alpha_input = self.gating_param.float() + self.alpha_predictor.float()(alpha_input_tensor)
        alpha = torch.sigmoid(alpha_input).to(self.clip_model.dtype)
        
        # Alpha Smoothing: Clamp alpha between [0.05, 0.95] to ensure both branches contribute
        alpha = torch.clamp(alpha, min=0.05, max=0.95)

        # 6. Compute Contextual Logits
        logits_ctx = logit_scale * normalized_img @ ctx_text_features.t()

        # 7. Final Score Fusion
        final_logits = alpha * logits_prim + (1.0 - alpha) * logits_ctx
        
        # 8. Temperature Scaling
        final_logits = final_logits / self.temp
        
        return final_logits

def get_dpc_alpha(train_dataset, config, device):
    from models.dpc import dpc_init
    
    csp_init_path = getattr(config, 'csp_init_path', None)
    freeze_primitive = getattr(config, 'freeze_primitive', False)

    (
        clip_model,
        soft_embedding_prim,
        soft_embedding_ctx,
        class_token_ids,
        offset
    ) = dpc_init(train_dataset, config, device, csp_checkpoint_path=csp_init_path)

    interface = DPCAlphaInterface(
        clip_model,
        config,
        offset,
        soft_embedding_prim,
        soft_embedding_ctx,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout
    )
    
    params_to_optimize = []
    
    if freeze_primitive:
        print("Freezing Primitive soft embeddings (CSP Anchor).")
        soft_embedding_prim.requires_grad = False
        params_to_optimize.append({'params': [soft_embedding_ctx]})
    else:
        params_to_optimize.append({'params': [soft_embedding_prim, soft_embedding_ctx]})
        
    # Always optimize gating and predictor
    params_to_optimize.append(
        {'params': [interface.gating_param] + list(interface.alpha_predictor.parameters()), 'lr': 0.001}
    )

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    return interface, optimizer
