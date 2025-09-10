
import torch

class EarlyStopNaN(RuntimeError):
    """Raised to stop training cleanly when NaN/Inf is detected."""
    pass

def _ensure_finite(t: torch.Tensor, name: str):
    if not torch.is_tensor(t):
        return
    if not torch.isfinite(t).all():
        raise EarlyStopNaN(f"Non-finite values detected in {name}.")
