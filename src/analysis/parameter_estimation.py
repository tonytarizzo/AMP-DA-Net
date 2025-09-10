
import torch
from ..codebooks.operations import normalize_codebook


# --------------------------------
# 1) Pi estimation
# --------------------------------

@torch.no_grad()
def compute_pi_distribution(dataset):
    """Compute normalized frequency distribution (pi) from dataset"""
    pi_counts = dataset.sum(dim=0)  # (n,)
    pi = pi_counts.sum().clamp_min(1e-12)
    return (pi_counts / pi).to(torch.float32).contiguous()

@torch.no_grad()
def pi_from_indices(idx: torch.Tensor, n: int, device) -> torch.Tensor:
    """Form π̂ by normalised counts from ANY index pool."""
    N = torch.bincount(idx.to(torch.long), minlength=n).to(device=device, dtype=torch.float32)
    return (N / N.sum().clamp_min(1e-12)).contiguous()


# --------------------------------
# 2) Active device estimation
# --------------------------------

@torch.no_grad()
def estimate_K_mf(Z_round: torch.Tensor, pi_round: torch.Tensor, C_nt: torch.Tensor, eps=1e-12) -> float:
    """
    Matched-filter K̂ along Gπ.
    Z_round: (S,d) received (noisy) vectors for the round
    pi_round: (n,) probability vector (from idx_mini)
    C_nt: (n,d) unit-norm rows
    """
    C_nt = normalize_codebook(C_nt)
    C_dn = C_nt.t().contiguous()
    gram = (C_nt @ C_nt.t()).contiguous()

    pi = (pi_round / pi_round.sum().clamp_min(eps)).flatten()
    g = gram @ pi                             # (n,)
    r_bar = (Z_round @ C_dn).mean(dim=0)      # (n,)

    denom = (g * g).sum().clamp_min(eps)
    K = float((r_bar * g).sum().item() / float(denom.item()))
    n = C_nt.size(0)
    return max(0.0, min(float(n), K))

# --------------------------------
# 3) Second moment estimation
# --------------------------------

@torch.no_grad()
def m2_from_idxmini_u_stat(C_nt: torch.Tensor, X_counts_round: torch.Tensor, K_a: int) -> float:
    eps = 1e-12
    if K_a <= 0: return float('nan')
    Y = X_counts_round @ C_nt
    e_tot = (Y*Y).sum(dim=1).mean().item()             # total energy (not per-dim)
    return e_tot / max(eps, K_a**2)


@torch.no_grad()
def m2_from_indices_counts(idx_round: torch.Tensor, K_used: float, C_nt: torch.Tensor, eps=1e-12) -> float:
    """
    U-statistic using counts from idx_mini over the whole round.
    idx_round: (S,) LongTensor of mini-dataset assignments for that round
    K_used:    the K you will use in the noise identity (ideally K̂ from MF)
    C_nt:      (n,d) unit-norm rows
    """
    C_nt = normalize_codebook(C_nt)
    n = C_nt.size(0)
    G = C_nt @ C_nt.t()
    diagG = torch.diagonal(G)

    N = torch.bincount(idx_round.to(torch.long), minlength=n).to(C_nt.device, dtype=torch.float32)
    M = float(N.sum().item())
    if M < 2:
        return float(diagG.mean().item())

    off = G - torch.diag(diagG)
    off_term  = torch.dot(N, off @ N)                # Σ_{i≠j} N_i N_j G_{ij}
    diag_term = torch.dot(N * (N - 1.0), diagG)      # Σ_i N_i (N_i-1) G_{ii}
    q_hat = float((off_term + diag_term) / (M * (M - 1.0) + eps))

    s_bar = float((N * diagG).sum().item() / max(eps, M))
    invK  = 1.0 / max(eps, float(K_used))
    return invK * s_bar + (1.0 - invK) * q_hat


@torch.no_grad()
def m2_from_indices(idx_round: torch.Tensor, K: float, C_nt: torch.Tensor) -> float:
    """
    Unbiased single-device m2 using U-statistics for q_full = E[G_{i_a,i_b}] over ordered a≠b.
    Returns m2_hat = (1/K) * s̄ + (1 - 1/K) * q̂_full, with:
      s̄ = (N · diag(G)) / M
      q̂_full = [ Nᵀ G N - Σ_i N_i G_ii ] / [ M (M - 1) ]
    """
    eps = 1e-12
    device = C_nt.device
    n, d = C_nt.shape
    G = (C_nt @ C_nt.t()).contiguous()
    diagG = torch.diagonal(G)

    N = torch.bincount(idx_round.to(torch.long), minlength=n).to(device=device, dtype=torch.float32)
    M = float(N.sum().item())
    if M <= 1:
        return float(diagG.mean().item())

    s_bar = float((N * diagG).sum().item() / M)
    quad  = float((N @ (G @ N)).item())
    self_diag = float((N * diagG).sum().item())
    q_hat = (quad - self_diag) / (M * (M - 1.0) + eps)

    invK = 1.0 / max(eps, float(K))
    return invK * s_bar + (1.0 - invK) * q_hat


@torch.no_grad()
def m2_true_from_pi(pi: torch.Tensor, K: float, C_nt: torch.Tensor) -> float:
    """Target m2 from full-round π and codebook (for logging/plots)."""
    G = C_nt @ C_nt.t()
    s = torch.diagonal(G)
    s_bar = float((pi * s).sum().item())
    q = float((pi @ (G @ pi)).item())
    return (K * s_bar + K * (K - 1.0) * q) / (K**2 + 1e-12)


# --------------------------------
# 4) Noise variance estimation
# --------------------------------

