
import math
import numpy as np
import torch
import torch.nn.functional as F

### Dataset Distribution Metrics ###
def kl_divergence(p, q, eps=1e-12):
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p / q).log()).sum().item()

def entropy(p, eps=1e-12):
    p = p.clamp(min=eps)
    return -(p * p.log()).sum().item()

def normalized_entropy(p):
    H = entropy(p)
    n = p.numel()
    return H / np.log(n)  # between 0 and 1

def gini_coefficient(p):
    # inequality measure for probability vector
    sorted_p, _ = torch.sort(p)
    n = p.numel()
    idx = torch.arange(1, n + 1, dtype=p.dtype, device=p.device)
    weights = (n - idx + 0.5)
    G = 1 - 2 * (weights * sorted_p).sum().item() / n
    return G

def topk_mass(p, fraction=0.05):
    k = max(1, int(fraction * p.numel()))
    topk, _ = torch.topk(p, k)
    return topk.sum().item()


### Quantisation Metrics ###
@torch.no_grad()
def _code_sim_abs_cosine(Q: torch.Tensor) -> torch.Tensor:
    # Q: (n, L)
    Qn = F.normalize(Q, p=2, dim=1, eps=1e-12)
    S = (Qn @ Qn.t()).abs()
    S.fill_diagonal_(0)  # keep display & metric fair (no self-sim)
    return S

@torch.no_grad()
def _bandedness_score(S: torch.Tensor) -> float:
    # Lower is better (more mass near diagonal)
    n = S.shape[0]
    if n <= 1 or float(S.sum()) <= 1e-12:
        return float('nan')
    idx = torch.arange(n, device=S.device, dtype=S.dtype)
    D = (idx[:, None] - idx[None, :]).abs()
    return float((S * D).sum().item() / (S.sum().item() + 1e-12))

def error_feedback_info(deltas, error_matrix):
    delta_l2 = deltas.norm(p=2).item()
    err_l2   = error_matrix.norm(p=2).item()
    avg_delta_abs = deltas.abs().mean().item()
    avg_err_abs   = error_matrix.abs().mean().item()
    ratio = err_l2 / (delta_l2 + 1e-12)
    print(f"Delta Norm = {delta_l2:.3e}, Error Norm = {err_l2:.3e}, Error / Delta = {ratio:.3f}")
    print(f"Average Delta = {avg_delta_abs:.3e}, Average Error = {avg_err_abs:.3e}")
    

### Channel Metrics ###
def compute_bits_per_dimension(num_codewords: int, block_size: int) -> float:
    """
    Compute bits per dimension = (log2 M) / Vb
    where M = num_codewords, Vb = block_size.
    """
    return math.log2(num_codewords) / block_size

def compute_ul_overhead(D, quant_len, ura_len, P=1024, complex=True):
    """ Compute the uplink overhead in terms of quantization and resource allocation, assumes packing in real setup. """
    splits = np.ceil(D/quant_len)
    if complex:
        symbols_per_split = ura_len
    else:
        symbols_per_split = np.ceil(ura_len/2)
    total_symbols = splits * symbols_per_split
    return np.ceil(total_symbols / P)
    