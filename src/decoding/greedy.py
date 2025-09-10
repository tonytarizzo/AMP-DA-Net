
import math
import numpy as np
import torch

def omp(y, C, K_a):
    """Simple Orthogonal Matching Pursuit (OMP) implementation. Complex case not supported."""
    l = 0
    x_recovered = np.zeros(C.shape[1])
    indices = np.zeros(K_a, dtype=int)

    while l < K_a:
        inner_product = C.T @ y
        indices[l] = np.argmax(np.abs(inner_product))

        C_sub = C[:, indices[:l+1]]
        x_ls = np.linalg.pinv(C_sub) @ y
        x_recovered[indices[:l+1]] = x_ls

        y = y - np.dot(C, x_recovered)
        l += 1
    return x_recovered

def ista(y, C, K_a=None, sigma=None, tau=None, lambda_val=None,
         max_iters=1000, tol=1e-6, debias=True, alpha=None, safety=0.99,
         complex_valued=False):
    """
    Solve 0.5||y - Cx||^2 + lambda||x||_1 by ISTA.
    Provide exactly one of {tau, lambda_val, sigma}. If none given, uses sigma via MAD on first step.
    If K_a is given, does an optional top-K projection at the end (oracle-K variant).
    If complex_valued=True, uses complex soft-thresholding on magnitudes.
    """
    _, n = C.shape

    # Step size
    if alpha is None:
        # Estimate ||C||_2^2 via power iteration (faster than exact SVD on big problems)
        s = power_iteration_snorm(C)
        L = (s ** 2)
        alpha = safety / (L + 1e-12)

    # Initial x and helpers
    x = np.zeros(n, dtype=y.dtype)
    def soft_thresh(z, t):
        if complex_valued:
            mag = np.abs(z)
            scale = np.maximum(0.0, 1.0 - t / (mag + 1e-12))
            return scale * z
        else:
            return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

    # Initial threshold selection
    if tau is None:
        if lambda_val is not None:
            tau = alpha * lambda_val
        else:
            # Estimate sigma if not given: MAD on first proxy
            if sigma is None:
                z0 = C.T @ y  # proxy when x=0 and alpha≈1/L
                sigma = np.median(np.abs(z0)) / 0.6745 + 1e-12
            lambda_val = sigma * np.sqrt(2.0 * np.log(n))
            tau = alpha * lambda_val

    prev = x.copy()
    for k in range(max_iters):
        # Gradient step
        r = C @ x - y
        z = x - alpha * (C.T @ r)
        x = soft_thresh(z, tau)

        # Stopping
        if np.linalg.norm(x - prev) <= tol * (np.linalg.norm(prev) + 1e-12):
            break
        prev = x.copy()

    # Optional oracle-K projection
    if K_a is not None:
        if K_a < n:
            idx = np.argpartition(np.abs(x), -K_a)[-K_a:]
            mask = np.zeros_like(x, dtype=bool)
            mask[idx] = True
            x[~mask] = 0.0

    # Debias on support (LS refit) for better amplitudes
    if debias:
        supp = np.flatnonzero(np.abs(x) > 0)
        if supp.size:
            Cs = C[:, supp]
            # Solve least squares: min ||y - Cs xs||_2
            xs, *_ = np.linalg.lstsq(Cs, y, rcond=None)
            x[:] = 0
            x[supp] = xs

    return x

def power_iteration_snorm(C, iters=50):
    # Fast estimate of ||C||_2
    v = np.random.randn(C.shape[1])
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(iters):
        v = C.T @ (C @ v)
        n = np.linalg.norm(v) + 1e-12
        v /= n
    s = np.linalg.norm(C @ v)
    return s  # spectral norm ||C||_2


def amp_da(
    y, C, T=50, damp=0.3, K_max=16, device=None):
    """
    MD-AirComp AMP-DA (single block, no fading, real tensors):
    Note that this implementation assumes real inputs, not complex
      - alpha init via λ * max_c rho(c) (AirComp)
      - x init via alpha_0 * (M/2)
      - uniform nonzero prior over {1..M}
      - EM updates for alpha and sigma2
    """
    if device is None:
        device = y.device
    y = y.to(device); C = C.to(device)
    B, d = y.shape
    n = C.shape[1]
    assert C.shape[0] == d, "C must be (d, n)"
    eps = 1e-12

    # --- alpha init ---
    def _alpha0_aircomp(d, n):
        lam = d / float(n)
        c_grid = torch.linspace(0.01, 10.0, 1024, device=device, dtype=y.dtype)
        phi = torch.exp(-0.5 * c_grid**2) / math.sqrt(2.0 * math.pi)
        Phi_neg = 0.5 * (1.0 - torch.erf(c_grid / math.sqrt(2.0)))
        R = (1.0 + c_grid**2) * Phi_neg - c_grid * phi
        rho = (1.0 - (2.0 / lam) * R) / (1.0 + c_grid**2 - 2.0 * R)
        rho_max = torch.clamp(rho.max(), min=0.0).item()
        a0 = lam * rho_max
        return float(max(min(a0, 1.0 - 1e-10), 1e-10))

    a0 = _alpha0_aircomp(d, n)
    alpha = torch.full((n,), a0, device=device, dtype=y.dtype)

    # --- alphabet {0..M} and M ---
    M = int(K_max)
    alphabet = torch.arange(0.0, M + 1, device=device, dtype=y.dtype)

    # --- messages ---
    x0_scalar = float(alpha.mean().item() * (M / 2.0))
    x_hat   = torch.full((B, n), x0_scalar, device=device, dtype=y.dtype)
    var_hat = torch.ones(B, n, device=device, dtype=y.dtype)
    Z       = y.clone()
    V       = torch.ones(B, d, device=device, dtype=y.dtype)

    # --- σ² init ---
    sigma2 = torch.tensor(100.0, device=device, dtype=y.dtype)  # AirComp default

    # --- precompute ---
    C2  = C.pow(2)                 # (d, n)
    C2T = C2.t().contiguous()      # (n, d)

    # --- iterations ---
    prev_x = x_hat.clone()
    prev_mse = None

    for t in range(1, T + 1):
        # Output stage
        V_new = torch.matmul(var_hat, C2T)               # (B, d)
        Z_tmp = torch.matmul(x_hat, C.t())               # (B, d)
        denom_out = (sigma2 + V).clamp_min(eps)
        Z_new = Z_tmp - ((y - Z) / denom_out) * V_new

        # Damping
        Z = damp * Z + (1 - damp) * Z_new
        V = damp * V + (1 - damp) * V_new

        # Pseudo-channel
        inv  = (1.0 / (sigma2 + V).clamp_min(eps))       # (B, d)
        var1 = torch.matmul(inv, C2).clamp_min(eps)      # (B, n)
        Vi   = 1.0 / var1                                # (B, n)
        tmp  = ((y - Z) * inv)                           # (B, d)
        var2 = torch.matmul(tmp, C)                      # (B, n)
        Ri   = var2 / var1 + x_hat                       # (B, n)

        # Discrete posterior (uniform nonzero)
        R_exp  = Ri.unsqueeze(-1)                        # (B,n,1)
        Vi_exp = Vi.unsqueeze(-1).clamp_min(eps)         # (B,n,1)
        m_exp  = alphabet.view(1, 1, -1)                 # (1,1,M+1)
        ll = - (R_exp - m_exp)**2 / (2.0 * Vi_exp)

        alphaB = alpha.unsqueeze(0)                      # (1,n)
        logw = torch.empty_like(ll)
        logw[:, :, 0]  = torch.log1p(-alphaB)           # m=0
        if M > 0:
            logw[:, :, 1:] = torch.log(alphaB).unsqueeze(-1) - math.log(M)

        log_post = ll + logw
        post = torch.softmax(log_post, dim=2)           # normalize over m

        # Moments
        m1 = torch.sum(post * m_exp, dim=2)             # (B,n)
        m2 = torch.sum(post * (m_exp**2), dim=2)        # (B,n)
        x_hat_new   = m1
        var_hat_new = (m2 - m1**2).clamp_min(eps)

        # α update (average over batch)
        alpha_post = post[:, :, 1:].sum(dim=2).clamp(1e-10, 1-1e-10)
        alpha = alpha_post.mean(dim=0)

        # σ² update (AirComp EM)
        denom2  = (1.0 + V / (sigma2 + eps))
        s_term1 = ((y - Z)**2) / (denom2**2)
        s_term2 = sigma2 * V / (V + sigma2 + eps)
        sigma2 = max(1e-12, float((s_term1 + s_term2).mean().item()))

        # Assign
        x_hat, var_hat = x_hat_new, var_hat_new

        # Stopping rule
        y_hat = torch.matmul(x_hat, C.t())
        mse = (y - y_hat).pow(2).mean().item()
        if (t > 15) and (prev_mse is not None) and (mse >= prev_mse):
            x_hat = prev_x.clone()
            break
        prev_mse = mse
        prev_x = x_hat.clone()

        # NaN/Inf guard
        if (torch.isnan(x_hat).any() or torch.isinf(x_hat).any()
            or torch.isnan(Z).any() or torch.isnan(V).any()):
            raise RuntimeError("NaN/Inf in amp_dm_aircomp; check scaling/prior.")

    stats = {
        "alpha": alpha,                     # (n,)
        "sigma2": float(sigma2),            # scalar
        "iters": t,
        "K_support": float(alpha.sum().item()),  # expected #nonzero codewords (NOT what we use for K)
        "K_count_per_split": x_hat.sum(dim=1).detach().cpu()  # tensor shape (B=S,)
    }
    return x_hat, stats