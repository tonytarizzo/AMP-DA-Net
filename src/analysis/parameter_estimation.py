
import torch

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
