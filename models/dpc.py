import torch
import torch.nn as nn
import os
import clip
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load

class DPCInterface(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_embeddings_primitive,  # Tuned Prompt (Shared)
        soft_embeddings_contextual, # Parallel Prompt (Context-specific)
        class_token_ids,
        device="cuda:0",
        enable_pos_emb=True,
        attr_dropout=0.0,
    ):
        super().__init__(
            clip_model,
            config,
            class_token_ids,
            soft_embeddings_primitive, # Default to primitive for compatibility
            device=device,
            enable_pos_emb=enable_pos_emb,
        )

        self.offset = offset
        self.attr_dropout = nn.Dropout(attr_dropout)
        
        # Dual Prompts
        self.soft_embeddings_primitive = soft_embeddings_primitive
        self.soft_embeddings_contextual = soft_embeddings_contextual
        
        # Gating Factor (Alpha) - Learnable parameter, initialized to 0.5
        # We use a sigmoid to keep it between 0 and 1
        self.gating_param = nn.Parameter(torch.tensor(0.0).to(device)) 

    @property
    def alpha(self):
        return torch.sigmoid(self.gating_param)

    def construct_token_tensors(self, pair_idx):
        """
        Constructs token tensors for both Primitive (Tuned) and Contextual (Parallel) prompts.
        Returns a tuple or a combined tensor? 
        The prompt construction is: [SOS] [ATTR] [OBJ] [EOS]
        
        We need to generate TWO sets of token tensors:
        1. Primitive: Uses soft_embeddings_primitive
        2. Contextual: Uses soft_embeddings_contextual
        """
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        
        # Base token tensor from CLIP (static tokens)
        base_token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())
        
        # Apply Dropout
        primitive_embs = self.attr_dropout(self.soft_embeddings_primitive)
        contextual_embs = self.attr_dropout(self.soft_embeddings_contextual)

        # --- Construct Primitive Token Tensor ---
        token_tensor_prim = base_token_tensor.clone()
        token_tensor_prim[:, eos_idx - 2, :] = primitive_embs[attr_idx].type(self.clip_model.dtype)
        token_tensor_prim[:, eos_idx - 1, :] = primitive_embs[obj_idx + self.offset].type(self.clip_model.dtype)

        # --- Construct Contextual Token Tensor ---
        token_tensor_ctx = base_token_tensor.clone()
        # Contextual prompt might need specific logic, but for now we assume it follows the same structure
        # but uses the contextual embeddings which are optimized for Seen pairs
        token_tensor_ctx[:, eos_idx - 2, :] = contextual_embs[attr_idx].type(self.clip_model.dtype)
        token_tensor_ctx[:, eos_idx - 1, :] = contextual_embs[obj_idx + self.offset].type(self.clip_model.dtype)

        return token_tensor_prim, token_tensor_ctx

    def forward(self, batch_img, idx):
        # idx is train_pairs indices
        batch_img = batch_img.to(self.device)
        
        # Get both token tensors
        token_tensor_prim, token_tensor_ctx = self.construct_token_tensors(idx)

        # Encode both prompts
        # 1. Primitive Features
        prim_text_features = self.text_encoder(
            self.token_ids,
            token_tensor_prim,
            enable_pos_emb=self.enable_pos_emb,
        )
        prim_text_features = prim_text_features / prim_text_features.norm(dim=-1, keepdim=True)

        # 2. Contextual Features
        ctx_text_features = self.text_encoder(
            self.token_ids,
            token_tensor_ctx,
            enable_pos_emb=self.enable_pos_emb,
        )
        ctx_text_features = ctx_text_features / ctx_text_features.norm(dim=-1, keepdim=True)

        # Encode Image
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)

        # Compute Logits for both
        logit_scale = self.clip_model.logit_scale.exp()
        
        logits_prim = logit_scale * normalized_img @ prim_text_features.t()
        logits_ctx = logit_scale * normalized_img @ ctx_text_features.t()

        # --- Gating Mechanism ---
        # Final Score = alpha * Primitive + (1 - alpha) * Contextual
        # In the simplest version, alpha is a scalar. 
        # For more complex gating (entropy-based), we need the logits themselves.
        
        # Using simple learnable scalar gating for now as a starting point
        alpha = self.alpha
        final_logits = alpha * logits_prim + (1 - alpha) * logits_ctx
        
        return final_logits


def dpc_init(train_dataset, config, device, prompt_template="a photo of X X", csp_checkpoint_path=None):
    # Re-use csp_init logic but create TWO sets of embeddings
    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    
    # cleaning
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]

    if csp_checkpoint_path and os.path.exists(csp_checkpoint_path):
        print(f"Initializing DPC from CSP checkpoint: {csp_checkpoint_path}")
        checkpoint = torch.load(csp_checkpoint_path, map_location=device)
        # Check if it's a standard CSP checkpoint or a DPC one
        if 'soft_embeddings' in checkpoint:
            soft_embedding_prim = checkpoint['soft_embeddings'].clone().detach()
        elif 'soft_embeddings_primitive' in checkpoint:
            soft_embedding_prim = checkpoint['soft_embeddings_primitive'].clone().detach()
        else:
            raise KeyError(f"Could not find valid embeddings in {csp_checkpoint_path}")
    else:
        tokenized = torch.cat(
            [
                clip.tokenize(tok, context_length=config.context_length)
                for tok in attributes + classes
            ]
        )

        orig_token_embedding = clip_model.token_embedding(tokenized.to(device))

        # Initialize Soft Embeddings (Primitive) - Copy of CSP initialization
        soft_embedding_prim = torch.zeros(
            (len(attributes) + len(classes), orig_token_embedding.size(-1)),
            device=device
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_embedding_prim[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
    
    soft_embedding_prim = nn.Parameter(soft_embedding_prim)

    # Initialize Soft Embeddings (Contextual) - Start from same point as Primitive
    soft_embedding_ctx = soft_embedding_prim.clone().detach()
    soft_embedding_ctx.requires_grad = True
    soft_embedding_ctx = nn.Parameter(soft_embedding_ctx)

    class_token_ids = clip.tokenize(
        [prompt_template],
        context_length=config.context_length,
    )
    offset = len(attributes)

    return (
        clip_model,
        soft_embedding_prim,
        soft_embedding_ctx,
        class_token_ids,
        offset
    )

def get_dpc(train_dataset, config, device):
    (
        clip_model,
        soft_embedding_prim,
        soft_embedding_ctx,
        class_token_ids,
        offset
    ) = dpc_init(train_dataset, config, device)

    interface = DPCInterface(
        clip_model,
        config,
        offset,
        soft_embedding_prim,
        soft_embedding_ctx,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout
    )
    
    # Optimize both sets of embeddings and the gating parameter
    optimizer = torch.optim.Adam(
        [
            {'params': [soft_embedding_prim, soft_embedding_ctx]},
            {'params': [interface.gating_param], 'lr': 0.001} # Gating might need different LR
        ],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    return interface, optimizer
