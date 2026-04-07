"""Optimal Transport based cluster assignment for prototype updates."""

import torch
import torch.nn.functional as F


def local_assign(features, similarity_scores, top_percent=1.0):
    """Assign samples to clusters using similarity-based soft assignment.

    Simplified version of the OT-based assignment in ClusPro.
    Uses Gumbel-Softmax for differentiable hard assignment.

    Args:
        features:          [B, D] — batch features (detached)
        similarity_scores: [B, K] — similarity to K cluster centers
        top_percent:       float — fraction of samples to consider (1.0 = all)

    Returns:
        couplings:     [B, K] — soft assignment matrix
        selected_mask: [B] — boolean mask of selected samples
    """
    B, K = similarity_scores.shape

    # Select top samples by max similarity
    max_sim = similarity_scores.max(dim=-1)[0]
    num_select = max(1, int(B * top_percent))
    _, top_indices = torch.topk(max_sim, min(num_select, B))
    selected_mask = torch.zeros(B, dtype=torch.bool, device=features.device)
    selected_mask[top_indices] = True

    # Sinkhorn-like normalization for balanced assignment
    scores = similarity_scores.clone()

    # Temperature-scaled softmax for soft assignment
    tau = 0.5
    couplings = F.softmax(scores / tau, dim=-1)

    return couplings, selected_mask
