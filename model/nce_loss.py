"""NCE / Contrastive Loss for prototype-based learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Prototype-based contrastive loss.
    Pulls samples toward their assigned prototype cluster
    and pushes away from other clusters.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, prototypes, labels):
        """
        Args:
            features:   [B, D] — disentangled image features
            prototypes: [B, N_proto, D] — all prototypes (per-batch expanded)
            labels:     [B] — cluster assignment index for each sample
        """
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)

        # similarity: [B, N_proto]
        sim = torch.einsum('bd,bnd->bn', features, prototypes) / self.temperature

        loss = F.cross_entropy(sim, labels)
        return loss


class ContrastiveLoss_ppc(nn.Module):
    """Prototype-prototype contrastive loss.
    Ensures prototypes within the same primitive are separated.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, prototypes, labels):
        """
        Args:
            prototypes: [N_proto, D]
            labels:     [N_proto] — primitive index for each prototype
        """
        prototypes = F.normalize(prototypes, dim=-1)
        sim = prototypes @ prototypes.t() / self.temperature

        # mask: same primitive = positive
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)

        # log-sum-exp trick for stability
        exp_sim = torch.exp(sim)
        exp_sim.fill_diagonal_(0)

        pos = (exp_sim * mask).sum(dim=-1)
        neg = exp_sim.sum(dim=-1)

        loss = -torch.log(pos / (neg + 1e-8) + 1e-8)
        loss = loss[mask.sum(dim=-1) > 0].mean()

        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=prototypes.device)
