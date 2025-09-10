
import torch
import torch.nn as nn
import torch.nn.functional as F

    
class SimAMPNetLossWithSparsity(nn.Module):
    def __init__(self, lambda_sparse=0.001, lambda_w=0.001, lambda_k=0.01):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_w = lambda_w
        self.lambda_k = lambda_k

    def forward(self, model, x_gt, x_pred, K_a, K_final):
        # MSE reconstruction loss
        rec_loss = F.mse_loss(x_pred, x_gt, reduction='mean')

        # L1 sparsity (normalized by target scale)
        target_scale = torch.mean(torch.abs(x_gt)).clamp_min(1e-10)
        l1_norm = torch.mean(torch.abs(x_pred))
        sparsity = l1_norm / target_scale

        # W regularization only if model has W
        w_reg = 0.0
        if hasattr(model, "W"):
            I = torch.eye(model.W.shape[0], device=model.W.device, dtype=model.W.dtype)
            w_reg = torch.norm(model.W.conj().T @ model.W - I, p='fro') ** 2

        # K loss
        K_a_t      = torch.as_tensor(float(K_a), device=x_pred.device, dtype=x_pred.dtype)
        K_final_t  = torch.as_tensor(float(K_final), device=x_pred.device, dtype=x_pred.dtype)
        k_loss = F.mse_loss(K_final_t, K_a_t, reduction='mean')

        return rec_loss + self.lambda_sparse * sparsity + self.lambda_w * w_reg + self.lambda_k * k_loss