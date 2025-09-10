
import numpy as np

def greedy_rounding(x, K_a):
    """
    Rounds the real vector x >= 0 to a non-negative integer vector 
    whose entries sum to K_a in a way that approximately minimises
    the L2 distance between x and the rounded result.

    Returns
    -------
    x_int : ndarray, shape (n,)
        Integer-rounded version of x with sum(x_int) = K_a.
    """
    x_clipped = np.clip(x, 0, None)

    x_floor = np.floor(x_clipped)
    sum_floor = int(np.sum(x_floor))
    
    x_int = x_floor.copy()

    diff = K_a - sum_floor

    if diff == 0:
        return x_int.astype(int)
    elif diff > 0:
        frac = x_clipped - x_floor
        idx_desc = np.argsort(-frac)

        for i in idx_desc:
            if diff <= 0:
                break
            x_int[i] += 1
            diff -= 1
    else:
        surplus = -diff
        frac = x_clipped - x_floor
        idx_asc = np.argsort(frac)

        for i in idx_asc:
            if surplus <= 0:
                break
            if x_int[i] > 0:
                x_int[i] -= 1
                surplus -= 1
    
    return x_int.astype(int)

def top_k_nonneg(x, K):
    """
    Non-negative projection then keep the K largest entries (by value).
    x: (n,) numpy array
    """
    x = np.maximum(x, 0.0)
    if K >= x.size:
        return x
    idx = np.argpartition(x, -K)[-K:]   # indices of top-K (unordered)
    out = np.zeros_like(x)
    out[idx] = x[idx]
    return out

def l2_refit_on_support(y, A, x, ridge=0.0, nonneg=False):
    """
    Least-squares debias on the current support of x.
    y: (d,)   measurement
    A: (d,n)  sensing/design matrix
    x: (n,)   current estimate (support = nonzeros)
    ridge:    small Tikhonov term (e.g. 1e-6) if ill-conditioned
    nonneg:   clip negatives after refit
    """
    S = np.flatnonzero(x)
    if S.size == 0:
        return np.zeros_like(x)

    As = A[:, S]            # (d, |S|)
    if ridge and ridge > 0.0:
        # (As^T As + λI)^{-1} As^T y
        AtA = As.T @ As
        rhs = As.T @ y
        xs = np.linalg.solve(AtA + ridge * np.eye(S.size, dtype=AtA.dtype), rhs)
    else:
        xs, *_ = np.linalg.lstsq(As, y, rcond=None)

    if nonneg:
        xs = np.maximum(xs, 0.0)

    out = np.zeros_like(x)
    out[S] = xs
    return out