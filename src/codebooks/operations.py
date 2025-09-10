
import torch
import torch.nn.functional as F


def normalize_codebook(C_nt: torch.Tensor) -> torch.Tensor:
    """Row L2-normalize codebook (n,d) → unit-norm rows."""
    return F.normalize(C_nt, p=2, dim=1)


def make_perm_popularity(pi_est: torch.Tensor):
    """
    Return (perm, inv_perm) that sorts codewords by descending pi_est.
    pi_est: (n,)
    """
    assert pi_est.dim() == 1
    perm = torch.argsort(pi_est, descending=True)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
    return perm, inv_perm


def make_perm_spectral(
    Q: torch.Tensor,
    idx_mini: torch.Tensor | None = None,
    pi_est: torch.Tensor | None = None,
    blend_pop: float = 0.0,           # 0 = no popularity blend, >0 adds tie-break by popularity
    trans_weight: float = 0.3,        # how much to mix transition co-occurrence (0..1)
):
    """
    Spectral seriation of codewords using a nonnegative similarity.
    - Q: (n, L) codebook (rows = codewords)
    - idx_mini: (S,) assignments from the mini-dataset (optional)
    - pi_est: (n,) popularity (optional)
    Returns (perm, inv_perm).
    """
    n = Q.size(0)
    device = Q.device
    eps = 1e-12

    # Cosine-based nonnegative similarity
    Qn = F.normalize(Q, p=2, dim=1)
    S = (Qn @ Qn.t()).abs()                # (n,n), in [0,1]
    S.fill_diagonal_(0)

    # Optional: add transition adjacency from mini-dataset (consecutive assignments)
    if idx_mini is not None and idx_mini.numel() >= 2:
        idx_mini = idx_mini.to(device)
        T = torch.zeros_like(S)
        i = idx_mini[:-1]
        j = idx_mini[1:]
        # increment both directions
        T.index_put_((i, j), torch.ones_like(i, dtype=S.dtype), accumulate=True)
        T.index_put_((j, i), torch.ones_like(i, dtype=S.dtype), accumulate=True)
        if T.max() > 0:
            T = T / (T.max() + eps)
            S = (1.0 - trans_weight) * S + trans_weight * T

    # Graph Laplacian and Fiedler vector
    d = S.sum(dim=1)
    L = torch.diag(d) - S
    # Use CPU eigensolver for stability on small problems
    w, v = torch.linalg.eigh(L.cpu())
    fiedler = v[:, 1].to(device)           # 2nd smallest eigenvector

    # Base permutation by Fiedler coordinate
    perm = torch.argsort(fiedler)

    # Optional: blend in popularity as a stable tie-break
    if (blend_pop > 0.0) and (pi_est is not None):
        pi_est = pi_est.to(device)
        # ranks (lower is earlier)
        rank_f = torch.empty_like(perm)
        rank_f[perm] = torch.arange(n, device=device)
        rank_p = torch.argsort(torch.argsort(-pi_est))  # rank of descending popularity
        score = rank_f.float() + blend_pop * rank_p.float()
        perm = torch.argsort(score)

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(n, device=device)
    return perm, inv_perm


def get_round_perm(mode: str, Q: torch.Tensor, idx_mini: torch.Tensor, pi_est: torch.Tensor):
    """
    Returns a 1D LongTensor 'perm' (or None) describing the reordering to apply to codewords this round.
    Also returns an inverse perm for remapping idx_mini
    Uses the *same* perm builders you pasted from dataset collection.
    """
    m = (mode or "none").lower()
    if m == "none":
        return None, None
    if m == "pop":
        perm, inv_perm = make_perm_popularity(pi_est)
        return perm, inv_perm
    if m == "spectral":
        perm, inv_perm = make_perm_spectral(Q, idx_mini=idx_mini, pi_est=None, blend_pop=0.0, trans_weight=0.3)
        return perm, inv_perm
    if m in {"spectral_pop", "spectral+pop"}:
        # spectral order + popularity tie-break (your spectral helper supports popularity blending)
        perm, inv_perm = make_perm_spectral(Q, idx_mini=idx_mini, pi_est=pi_est, blend_pop=1.0, trans_weight=0.3)
        return perm, inv_perm
    raise ValueError(f"Unknown codebook ordering mode: {mode!r}")
