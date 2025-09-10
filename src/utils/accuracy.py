import numpy as np

def compute_accuracy(x, x_recovered):
    """
    Compute an accuracy measure based on the normalized L1 error between the true and recovered vectors.
    
    Accuracy is defined as:
        1 - (sum(|x - x_recovered|) / sum(x))
    
    Returns a fraction between 0 and 1 (multiply by 100 for a percentage).
    """
    x = np.array(x, dtype=float)
    x_recovered = np.array(x_recovered, dtype=float)
    
    total = np.sum(x)
    if total == 0:
        return 1.0  # or handle as a special case if needed
    
    error = np.sum(np.abs(x - x_recovered))
    return max(0.0, 1.0 - error / total)

def compute_pupe(x, x_recovered):
    x = np.array(x, dtype=int)
    x_recovered  = np.array(np.round(x_recovered), dtype=int)
    K_a = np.sum(x)
    if K_a == 0:
        raise ValueError("x has zero sum—no active users to compute PUPE")
    total_l1 = np.abs(x - x_recovered).sum()
    return min(total_l1 / K_a, 1.0)

def compute_nmse(x, x_recovered):
    """
    Compute the Normalized Mean Squared Error (NMSE) between the true vector x and the recovered vector.
    Lower NMSE indicates better recovery.
    
    NMSE = ||x - x_recovered||^2 / ||x||^2
    """
    return np.linalg.norm(x - x_recovered)**2 / (np.linalg.norm(x)**2 + 1e-8)

def compute_rsnr(x, x_recovered):
    """
    Compute the Reconstruction Signal-to-Noise Ratio (RSNR) in dB.
    
    RSNR = 10 * log10( ||x||^2 / ||x - x_recovered||^2 )
    Higher RSNR means better recovery.
    """
    x = np.array(x, dtype=float)
    x_recovered = np.array(x_recovered, dtype=float)

    signal_power = np.linalg.norm(x)**2
    error_power = np.linalg.norm(x - x_recovered)**2 + 1e-8  # small epsilon to avoid log(0)
    
    return 10 * np.log10(signal_power / error_power)

def compute_accuracy_batch(X, X_rec):
    """
    Normalised L1 accuracy for a batch.
    accuracy_i = 1 -  Σ|x_i - x̂_i| / Σ x_i   (clipped to [0,1])
    """
    X      = np.asarray(X,      dtype=float)
    X_rec  = np.asarray(X_rec,  dtype=float)

    l1_err   = np.abs(X - X_rec).sum(axis=1)          # (batch,)
    totals   = X.sum(axis=1) + 1e-12                  # avoid /0
    acc      = 1.0 - l1_err / totals
    return np.clip(acc, 0.0, 1.0)

def compute_pupe_batch(X, X_rec):
    """
    Per-User Probability of Error (PUPE) for a batch.
    PUPE_i = min( Σ|x_i - round(x̂_i)| / K_a , 1 )
    """
    X       = np.asarray(X,     dtype=int)
    X_rec_r = np.asarray(np.round(X_rec), dtype=int)

    l1_err  = np.abs(X - X_rec_r).sum(axis=1)         # (batch,)
    K_a     = X.sum(axis=1)                           # active users
    pupe    = np.where(K_a > 0, l1_err / K_a, np.nan) # NaN if K_a==0
    return np.clip(pupe, 0.0, 1.0)

def compute_nmse_batch(X, X_rec):
    """
    Normalised MSE for a batch.
    NMSE_i = ‖x_i - x̂_i‖² / (‖x_i‖² + ε)
    """
    X     = np.asarray(X,     dtype=float)
    X_rec = np.asarray(X_rec, dtype=float)

    num   = np.sum((X - X_rec) ** 2, axis=1)
    den   = np.sum(X ** 2, axis=1) + 1e-8
    return num / den

def compute_rsnr_batch(X, X_rec):
    """
    Reconstruction SNR (dB) for a batch.
    RSNR_i = 10·log10( ‖x_i‖² / ‖x_i - x̂_i‖² )
    """
    X     = np.asarray(X,     dtype=float)
    X_rec = np.asarray(X_rec, dtype=float)

    signal_pwr = np.sum(X ** 2, axis=1)
    error_pwr  = np.sum((X - X_rec) ** 2, axis=1) + 1e-8
    return 10.0 * np.log10(signal_pwr / error_pwr)