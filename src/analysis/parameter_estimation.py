
import torch

@torch.no_grad()
def compute_pi_distribution(dataset):
    """Compute normalized frequency distribution (pi) from dataset"""
    pi_counts = dataset.sum(dim=0)  # (n,)
    pi = pi_counts.sum().clamp_min(1e-12)
    return (pi_counts / pi).to(torch.float32).contiguous()
