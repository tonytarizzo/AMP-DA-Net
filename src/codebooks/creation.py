
import torch
from .operations import normalize_codebook
from ..utils.channel import add_awgn_noise

def make_gaussian_codebook(n: int, d: int, device) -> torch.Tensor:
    return normalize_codebook(torch.randn(n, d, device=device, dtype=torch.float32))

def make_bernoulli_codebook(n: int, d: int, device) -> torch.Tensor:
    C = (2 * torch.randint(0, 2, (n, d), device=device) - 1).to(torch.float32)
    return normalize_codebook(C)

def compute_q_init(X_train, snr_db, dim, device='cuda'):
    """
    Least-squares 'pseudoinverse-style' initializer from training counts.
    Returns a (n, dim) row-normalized matrix suitable as a URA codebook seed.
    """
    _, n = X_train.shape
    C_temp = normalize_codebook(torch.randn(n, dim, device=device))
    Y_train = X_train @ C_temp
    if snr_db is not None:
        Y_train = add_awgn_noise(Y_train, snr_db)
    X = X_train.double()
    Y = Y_train.double()
    YtY = Y.T @ Y
    Q_init = (X.T @ Y) @ torch.linalg.pinv(YtY)
    return normalize_codebook(Q_init.float())