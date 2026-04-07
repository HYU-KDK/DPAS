"""HSIC (Hilbert-Schmidt Independence Criterion) for decorrelation learning.
Ensures attr and obj disentangled features are independent.
"""

import torch


def rbf_kernel(X, sigma=None):
    """Compute RBF (Gaussian) kernel matrix."""
    pairwise_sq = torch.cdist(X, X, p=2).pow(2)
    if sigma is None:
        sigma = torch.median(pairwise_sq[pairwise_sq > 0]).clamp(min=1e-5)
    return torch.exp(-pairwise_sq / (2 * sigma))


def hsic_normalized(X, Y):
    """Compute normalized HSIC between two feature sets.

    Args:
        X: [B, D] — first feature set (e.g., attr features)
        Y: [B, D] — second feature set (e.g., obj prototypes)

    Returns:
        scalar — HSIC value (lower = more independent)
    """
    n = X.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=X.device)

    X = X.float()
    Y = Y.float()

    K = rbf_kernel(X)
    L = rbf_kernel(Y)

    # centering matrix H = I - 1/n * 11^T
    H = torch.eye(n, device=X.device) - 1.0 / n

    # HSIC = 1/(n-1)^2 * tr(KHLH)
    KH = K @ H
    LH = L @ H

    hsic = torch.trace(KH @ LH) / ((n - 1) ** 2)

    # normalize
    hsic_xx = torch.trace(KH @ KH) / ((n - 1) ** 2)
    hsic_yy = torch.trace(LH @ LH) / ((n - 1) ** 2)

    denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=1e-8)
    return hsic / denom
