
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from .metrics import _code_sim_abs_cosine, _bandedness_score


def plot_codebook_ordering(Q_before, pi_before, idx_before, perm, args, tag: str = ""):
    assert perm.ndim == 1 and perm.dtype == torch.long, "perm must be 1D LongTensor"
    assert perm.numel() == Q_before.size(0), "perm length must match #codes"
    assert perm.device == Q_before.device, "perm and Q_before must be on same device"

    vis_dir = os.path.join(args.save_dir, "code_order_vis")
    os.makedirs(vis_dir, exist_ok=True)

    # Similarities
    S_before = _code_sim_abs_cosine(Q_before)
    S_after  = S_before[perm][:, perm]

    # Scores
    band_before = _bandedness_score(S_before)
    band_after  = _bandedness_score(S_after)
    delta = band_after - band_before

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axes[0].imshow(S_before.detach().cpu().numpy(), vmin=0.0, vmax=1.0, aspect='auto')
    axes[0].set_title(f"Similarity (before)\nband={band_before:.3f}")
    axes[0].set_xlabel("Codeword idx"); axes[0].set_ylabel("Codeword idx")
    im1 = axes[1].imshow(S_after.detach().cpu().numpy(), vmin=0.0, vmax=1.0, aspect='auto')
    axes[1].set_title(f"Similarity (after)\nband={band_after:.3f} (Δ={delta:+.3f})")
    axes[1].set_xlabel("Codeword idx (reordered)")
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("|cosine|")
    plt.savefig(os.path.join(vis_dir, f"codebook_ordering_heatmap{('_' + tag) if tag else ''}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return band_before, band_after


def plot_dataset_collection_history(history, save_dir):
    rounds = list(range(1, len(history['test_acc']) + 1))
    os.makedirs(save_dir, exist_ok=True)
    
    # Figure 1: Test Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(rounds, history['test_acc'], linewidth=2, linestyle='-', color='tab:blue')
    plt.xlabel('Global Round')
    plt.ylabel('Test Accuracy')
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Test Accuracy over Rounds')
    acc_path = os.path.join(save_dir, 'test_accuracy.png')
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close() 
    print(f"Test accuracy plot saved to {acc_path}")


def plot_pretraining(train_losses, val_losses, epochs, lr, batch_size, train_size, num_layers, save_dir="results"):
    os.makedirs(f"{save_dir}/loss_curves", exist_ok=True)

    # 1) AMP-Net Curve
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='AMP Train Loss', linewidth=2)
    plt.plot(val_losses, label='AMP Val Loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("AMP Loss")
    plt.title(f"AMP-Net Loss Curves\nEpochs: {epochs}, LR: {lr}, Batch: {batch_size}, TrainSize: {train_size}, Layers: {num_layers}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    amp_path = (f"{save_dir}/loss_curves/amp_layers{num_layers}_epochs{epochs}_bs{batch_size}.png")
    plt.tight_layout()
    plt.savefig(amp_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_estimation_diagnostics(estimation_logs, save_dir):
    os.makedirs(os.path.join(save_dir, "eval_plots"), exist_ok=True)
    out_dir = os.path.join(save_dir, "eval_plots")

    K_true  = np.array([float(log["K_true"])     for log in estimation_logs], dtype=float)
    K_est   = np.array([float(log["K_est"])      for log in estimation_logs], dtype=float)
    SNR_t   = np.array([float(log["snr_true"])   for log in estimation_logs], dtype=float)
    SNR_est = np.array([float(log["snr_est"])    for log in estimation_logs], dtype=float)
    m2_t    = np.array([float(log["m2_true"])    for log in estimation_logs], dtype=float)
    m2_est  = np.array([float(log["m2_est"])     for log in estimation_logs], dtype=float)

    # 1) K_true vs K_est
    plt.figure(figsize=(6,6))
    plt.scatter(K_true, K_est, s=18)
    lo = min(K_true.min(), K_est.min())
    hi = max(K_true.max(), K_est.max())
    plt.plot([lo, hi], [lo, hi], linestyle='--')
    plt.xlabel("K true")
    plt.ylabel("K estimated")
    plt.title("Per-round K: true vs estimated")
    plt.grid(True, alpha=0.3)
    k_path = os.path.join(out_dir, "K_true_vs_K_est.png")
    plt.tight_layout()
    plt.savefig(k_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2) SNR_true vs SNR_est
    plt.figure(figsize=(6,6))
    plt.scatter(SNR_t, SNR_est, s=18)
    lo = min(SNR_t.min(), SNR_est.min())
    hi = max(SNR_t.max(), SNR_est.max())
    plt.plot([lo, hi], [lo, hi], linestyle='--')
    plt.xlabel("SNR true (dB)")
    plt.ylabel("SNR estimated (dB)")
    plt.title("Per-round SNR: true vs estimated")
    plt.grid(True, alpha=0.3)
    snr_path = os.path.join(out_dir, "SNR_true_vs_SNR_est.png")
    plt.tight_layout()
    plt.savefig(snr_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3) m2_true vs m2_est (scatter)
    plt.figure(figsize=(6,6))
    plt.scatter(m2_t, m2_est, s=18)
    lo = float(min(m2_t.min(), m2_est.min()))
    hi = float(max(m2_t.max(), m2_est.max()))
    plt.plot([lo, hi], [lo, hi], linestyle='--')
    plt.xlabel("m₂ true")
    plt.ylabel("m₂ estimated")
    plt.title("Per-round m₂: true vs estimated")
    plt.grid(True, alpha=0.3)
    m2_scatter_path = os.path.join(out_dir, "m2_true_vs_m2_est.png")
    plt.tight_layout()
    plt.savefig(m2_scatter_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 4) m2 traces by round index
    r_idx = np.arange(len(estimation_logs))
    plt.figure(figsize=(10,6))
    plt.plot(r_idx, m2_t, label='m₂ true', linewidth=2)
    plt.plot(r_idx, m2_est, label='m₂ est', linewidth=2)
    plt.xlabel("Round")
    plt.ylabel("m₂")
    plt.title("Per-round m₂ (true vs est)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    m2_trend_path = os.path.join(out_dir, "m2_true_est_traces.png")
    plt.tight_layout()
    plt.savefig(m2_trend_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_traces(estimation_logs, save_dir):
    out_dir = os.path.join(save_dir, "eval_plots")
    os.makedirs(out_dir, exist_ok=True)

    # K Ribbon Plot
    plt.figure(figsize=(10, 6))
    k_traces = []
    for log in estimation_logs:
        k_tr = np.asarray(log["K_trace"], dtype=float)
        k_traces.append(k_tr)
        plt.plot(np.arange(len(k_tr)), k_tr, alpha=0.35)  # faint per-round curve

    plt.xlabel("Layer (0=init)")
    plt.ylabel("K (float)")
    plt.title("K trajectory across AMP layers (per-round + ribbon)")
    plt.grid(True, alpha=0.3)

    if k_traces:
        T1 = min(len(t) for t in k_traces)          # robust to tiny length mismatches
        M  = np.stack([t[:T1] for t in k_traces], axis=0)  # (R, T1)
        x  = np.arange(T1)
        q25 = np.quantile(M, 0.25, axis=0)
        q75 = np.quantile(M, 0.75, axis=0)
        mean = M.mean(axis=0)
        plt.fill_between(x, q25, q75, alpha=0.2, label="IQR across rounds")
        plt.plot(x, mean, linewidth=2.5, label="Mean across rounds")
        if len(estimation_logs) <= 8:
            plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "K_trace_ribbon.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Sigma2 Ribbon Plot
    plt.figure(figsize=(10, 6))
    s_traces = []
    for log in estimation_logs:
        if "sigma_trace" not in log:
            continue
        s_tr = np.asarray(log["sigma_trace"], dtype=float)
        s_traces.append(s_tr)
        plt.plot(np.arange(len(s_tr)), s_tr, alpha=0.35)  # faint per-round curve

    plt.xlabel("Layer (0=init)")
    plt.ylabel("sigma^2 (mean)")
    plt.title("Noise variance trajectory across AMP layers (per-round + ribbon)")
    plt.grid(True, alpha=0.3)

    if s_traces:
        T1 = min(len(t) for t in s_traces)
        M  = np.stack([t[:T1] for t in s_traces], axis=0)
        x  = np.arange(T1)
        q25 = np.quantile(M, 0.25, axis=0)
        q75 = np.quantile(M, 0.75, axis=0)
        mean = M.mean(axis=0)
        plt.fill_between(x, q25, q75, alpha=0.2, label="IQR across rounds")
        plt.plot(x, mean, linewidth=2.5, label="Mean across rounds")
        if len(estimation_logs) <= 8:
            plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sigma_trace_ribbon.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_inference_history(history, save_dir, snr_true_db: float):
    rounds = list(range(1, len(history['test_acc']) + 1))
    os.makedirs(save_dir, exist_ok=True)

    def _save(fig, name):
        p = os.path.join(save_dir, name); plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close(); print(f"Saved {p}")

    plt.figure(figsize=(10,6))
    plt.plot(rounds, history['test_acc'], linewidth=2)
    plt.xlabel('Global Round'); plt.ylabel('Test Accuracy'); plt.ylim(0.0, 1.0); plt.grid(True, ls='--', alpha=0.5)
    plt.title('Test Accuracy over Rounds'); _save(plt, 'test_accuracy.png')

    plt.figure(figsize=(10,6))
    plt.plot(rounds, history['K_a'],   ls='--', label='True Kₐ')
    plt.plot(rounds, history['K_est'], ls='-',  label='Estimated K̂')
    plt.xlabel('Global Round'); plt.ylabel('Active Devices'); plt.title('True vs Estimated K'); plt.legend(); plt.grid(True, ls='--', alpha=0.5)
    _save(plt, 'ka_vs_kest.png')

    plt.figure(figsize=(6,6))
    Ka = np.asarray(history['K_a'], dtype=float); Ke = np.asarray(history['K_est'], dtype=float)
    plt.scatter(Ka, Ke, s=18); lo, hi = float(min(Ka.min(),Ke.min())), float(max(Ka.max(),Ke.max()))
    plt.plot([lo,hi],[lo,hi],'--'); plt.xlabel("K true"); plt.ylabel("K estimated"); plt.title("Per-round K: true vs est"); plt.grid(True, alpha=0.3)
    _save(plt, 'K_true_vs_K_est.png')

    plt.figure(figsize=(10,6))
    plt.plot(rounds, history['ampnet_acc'], marker='o')
    plt.xlabel('Global Round'); plt.ylabel('AMP-DM-Net Accuracy'); plt.ylim(0.0,1.0); plt.grid(True, ls='--', alpha=0.5)
    _save(plt, 'ampnet_accuracy.png')

    plt.figure(figsize=(10,6))
    plt.plot(rounds, history['ampnet_pupe'], marker='d')
    plt.xlabel('Global Round'); plt.ylabel('PUPE'); plt.ylim(0.0,1.0); plt.grid(True, ls='--', alpha=0.5)
    _save(plt, 'ampnet_pupe.png')

    plt.figure(figsize=(10,6))
    plt.plot(rounds, history['snr_final'], marker='o', label='Estimated')
    plt.axhline(y=snr_true_db, ls='--', label='True')
    plt.xlabel('Global Round'); plt.ylabel('SNR (dB)'); plt.grid(True, ls='--', alpha=0.5); plt.title('Estimated SNR per Round'); plt.legend()
    _save(plt, 'snr_final_db.png')

    plt.figure(figsize=(10,6))
    plt.plot(rounds, history['mean_local_loss'], marker='^')
    plt.xlabel('Global Round'); plt.ylabel('Mean Local Loss'); plt.grid(True, ls='--', alpha=0.5); plt.title('Mean Local Training Loss per Round')
    _save(plt, 'mean_local_loss.png')