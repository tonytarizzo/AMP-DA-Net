
import torch


def add_awgn_noise(signal, snr_db):
    if snr_db is None:
        return signal
    snr_lin = 10 ** (snr_db / 10.0)
    P_row = signal.pow(2).mean(dim=1, keepdim=True)         # (S,1)
    noise_std = (P_row / snr_lin).sqrt()                    # (S,1)
    return signal + torch.randn_like(signal) * noise_std


def bernoulli_encode_with_noise(X, n, d, snr_db, device, dtype=torch.float32): # Helper to make bernoulli and add noise in one go
    A = (2*torch.randint(0, 2, (d, n), device=device) - 1).to(dtype)
    A = torch.nn.functional.normalize(A, p=2, dim=0)   # unit-norm columns
    C = A.t().contiguous()                             # (n, d) "codebook"
    Z = X @ C
    if snr_db is not None:
        Z = add_awgn_noise(Z, snr_db)
    return Z, A


def make_snr_sampler(args, stream="train", seed=None):
    """
    Simple persistent SNR sampler.
    - 'fixed'  → always returns args.snr_db
    - 'range'  → uniform in [snr_min_db, snr_max_db]
    A separate deterministic RNG stream per {train,valid,eval,test}.
    """
    mode = str(getattr(args, "snr_mode", "fixed")).lower()
    if mode == "fixed":
        const = float(args.snr_db)
        return lambda: const
    lo = float(getattr(args, "snr_min_db", 0.0))
    hi = float(getattr(args, "snr_max_db", 20.0))
    if seed is None:
        base = int(getattr(args, "seed", 0))
        bump = {"train": 0, "valid": 12345, "eval": 22222, "test": 33333}.get(stream, 44444)
        seed = base + bump
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    def _next():
        return float(torch.empty((), device="cpu").uniform_(lo, hi, generator=g).item())
    return _next


def _parse_snr_list(s: str) -> list[float]:  # Helper for choosing snrs to be tested
    s = (s or '').strip()
    if not s:
        return []
    vals = []
    for tok in s.replace(';', ',').split(','):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals