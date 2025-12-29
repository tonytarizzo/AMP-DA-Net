
import numpy as np
import torch
from sklearn.cluster import kmeans_plusplus, KMeans
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


def create_quantisation_codebook(bs_blocks, args):
    """
    Create quantization codebook from base station blocks using k-means.
    
    Supports three curvature modes:
      - 'none': Standard k-means++ (with optional Lloyd iterations via kmeans_max_iter)
      - 'diagonal_hessian_proxy': Diagonal Hessian proxy scaling before k-means
      - 'data_whitening': SVD-based ZCA whitening before k-means
    
    Args:
        bs_blocks: (S, L) tensor of base station message blocks
        args: Namespace with n, message_split, seed, kmeans_max_iter, curv_mode, curv_max_samples, curv_eps
    
    Returns:
        Q: (n, L) quantization codebook tensor on same device as bs_blocks
    """
    if bs_blocks.is_cuda:
        X_cpu = bs_blocks.detach().cpu().contiguous()
    else:
        X_cpu = bs_blocks.contiguous()
    X = X_cpu.view(-1, args.message_split)
    X_np = X.numpy()
    
    N, L = X_np.shape
    n_clusters = int(getattr(args, 'n', 256))
    max_iter = int(getattr(args, 'kmeans_max_iter', 0))
    curv_mode = str(getattr(args, 'curv_mode', 'none')).lower()
    curv_max_samples = int(getattr(args, 'curv_max_samples', N))
    curv_eps = float(getattr(args, 'curv_eps', 1e-6))
    seed = int(getattr(args, 'seed', 0))
    
    if curv_mode == 'none':
        # Standard k-means++ (with optional Lloyd iterations)
        centers = _kmeans_standard(X_np, n_clusters, max_iter, seed)
    elif curv_mode == 'diagonal_hessian_proxy':
        # Diagonal Hessian proxy scaling before k-means
        centers = _kmeans_diagonal_hessian_proxy(X_np, n_clusters, max_iter, seed, curv_max_samples, curv_eps)
    elif curv_mode == 'data_whitening':
        # SVD-based ZCA whitening before k-means
        centers = _kmeans_data_whitening(X_np, n_clusters, max_iter, seed, curv_max_samples, curv_eps)
    else:
        raise ValueError(f"Unknown curv_mode: {curv_mode!r}. Must be one of ['none', 'diagonal_hessian_proxy', 'data_whitening']")
    
    target_device = bs_blocks.device if bs_blocks.is_cuda else (getattr(args, 'device', 'cpu'))
    Q = torch.tensor(centers, dtype=torch.float32, device=target_device)
    return Q


def _kmeans_standard(X_np: np.ndarray, n_clusters: int, max_iter: int, seed: int) -> np.ndarray:
    """Standard k-means++ with optional Lloyd iterations."""
    if max_iter == 0:
        # Just get initial seeds from k-means++
        centers, _ = kmeans_plusplus(X_np, n_clusters=n_clusters, random_state=seed)
    else:
        # Use KMeans with specified iterations
        km = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            max_iter=max_iter,
            random_state=seed,
            verbose=0,
        )
        km.fit(X_np)
        centers = km.cluster_centers_.astype(np.float32)
    return centers


def _kmeans_diagonal_hessian_proxy(X_np: np.ndarray, n_clusters: int, max_iter: int, seed: int, curv_max_samples: int, curv_eps: float) -> np.ndarray:
    """Diagonal Hessian proxy: K-means++ in diagonally-scaled space."""
    N, L = X_np.shape
    
    # Subsample if needed for statistics
    if curv_max_samples > 0 and N > curv_max_samples:
        indices = np.random.choice(N, size=curv_max_samples, replace=False)
        X_stats = X_np[indices]
    else:
        X_stats = X_np
    
    # Center the data
    mu = X_stats.mean(axis=0)  # (L,)
    
    # Compute per-dimension spread (MAD or std) for robustness
    X_centered = X_stats - mu  # (n_samples, L)
    
    # Use MAD (Median Absolute Deviation) for robustness, fallback to std
    medians = np.median(X_centered, axis=0)  # (L,)
    mad = np.median(np.abs(X_centered - medians), axis=0)  # (L,)
    # Scale MAD to match std for Gaussian: MAD ≈ 0.6745 * std, so std ≈ MAD / 0.6745
    std_approx = mad / 0.6745
    # Fallback to std where MAD is zero
    std_fallback = np.std(X_centered, axis=0)
    sigma_i = np.where(mad > 1e-10, std_approx, std_fallback)  # (L,)
    
    # Diagonal Hessian proxy: H_i ∝ 1/(σ_i² + ε)
    hessian_diag = 1.0 / (sigma_i ** 2 + curv_eps)  # (L,)
    
    # Diagonal whitening: W = diag(√H)
    W_diag = np.sqrt(hessian_diag)  # (L,)
    
    # Apply diagonal scaling: Ỹ = (Y - μ) ⊙ diag(W)
    X_full_centered = X_np - mu  # (N, L)
    X_scaled = X_full_centered * W_diag  # (N, L) element-wise
    
    # Run K-means in scaled space
    if max_iter == 0:
        centers_scaled, _ = kmeans_plusplus(X_scaled, n_clusters=n_clusters, random_state=seed)
    else:
        km = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            max_iter=max_iter,
            random_state=seed,
            verbose=0,
        )
        km.fit(X_scaled)
        centers_scaled = km.cluster_centers_.astype(np.float32)
    
    # Inverse transform: C = C̃ / W + μ
    W_inv = 1.0 / (W_diag + curv_eps)  # (L,)
    centers = centers_scaled * W_inv + mu  # (n, L)
    
    return centers.astype(np.float32)


def _kmeans_data_whitening(X_np: np.ndarray, n_clusters: int, max_iter: int, seed: int, curv_max_samples: int, curv_eps: float) -> np.ndarray:
    """Whitening transform: K-means++ in Mahalanobis space using SVD-based ZCA whitening."""
    N, L = X_np.shape
    
    # Subsample if needed for covariance estimation
    if curv_max_samples > 0 and N > curv_max_samples:
        indices = np.random.choice(N, size=curv_max_samples, replace=False)
        X_cov = X_np[indices]
    else:
        X_cov = X_np
    
    # Center the data
    mu = X_cov.mean(axis=0, keepdims=True)  # (1, L)
    X_centered = X_cov - mu
    
    # SVD-based whitening: X_centered = U S V^T
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    n_samples = X_cov.shape[0]
    
    # Whitening matrix: W = Σ^(-1/2) = Vt^T @ diag(√(N-1)/s) @ Vt (with ridge)
    n_samples_float = float(max(n_samples - 1, 1))
    
    # Regularize singular values: s_reg = s + ε·max(s)
    s_reg = s + curv_eps * np.max(s)
    scale_whiten = np.sqrt(n_samples_float) / s_reg  # (rank,)
    
    # Construct whitening transform
    rank = Vt.shape[0]
    V_eig = Vt.T  # (L, rank) - eigenvectors in columns
    
    # Construct W: (L, L) whitening matrix
    if rank < L:
        # Rank-deficient case: pad null space
        W = V_eig @ np.diag(scale_whiten) @ V_eig.T + (np.eye(L) - V_eig @ V_eig.T)
    else:
        # Full rank
        W = V_eig @ np.diag(scale_whiten) @ V_eig.T  # (L, L)
    
    # Apply whitening to full dataset
    X_full_centered = X_np - mu  # (N, L)
    X_whitened = X_full_centered @ W.T  # (N, L)
    
    # Run K-means in whitened space
    if max_iter == 0:
        centers_whitened, _ = kmeans_plusplus(X_whitened, n_clusters=n_clusters, random_state=seed)
    else:
        km = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            max_iter=max_iter,
            random_state=seed,
            verbose=0,
        )
        km.fit(X_whitened)
        centers_whitened = km.cluster_centers_.astype(np.float32)
    
    # Inverse transform: C = C̃ @ W_inv + μ
    scale_inv = s_reg / np.sqrt(n_samples_float)  # (rank,)
    if rank < L:
        W_inv = V_eig @ np.diag(scale_inv) @ V_eig.T + (np.eye(L) - V_eig @ V_eig.T)
    else:
        W_inv = V_eig @ np.diag(scale_inv) @ V_eig.T  # (L, L)
    centers = centers_whitened @ W_inv + mu  # (n, L)
    
    return centers.astype(np.float32)