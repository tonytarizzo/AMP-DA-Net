
import argparse
import os
import copy
import math
import time 
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import kmeans_plusplus

from src.analysis import (
    error_feedback_info,
    compute_bits_per_dimension,
    compute_ul_overhead,
    plot_inference_history,
    save_json,
    append_jsonl,
    save_round_history_csv,
    run_manifest,
)
from src.codebooks import get_round_perm
from src.datasets import load_cifar_datasets
from src.decoding import (
    AMPNet1DEnhanced,
    SimAMPNetLossWithSparsity,
    greedy_rounding, 
    l2_refit_on_support, 
    top_k_nonneg,
)
from src.federated import update_model_inplace
from src.models import build_global_model
from src.paths import load_pretrained_pair
from src.utils import (
    compute_accuracy_batch,
    compute_pupe,
    add_awgn_noise,
)


class WirelessCompressor(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.grad_accumulation_size = int(args.grad_accumulation_size)
        self.post_rounding = bool(args.post_rounding)

        # 1) Load pretrained AMPNet & URA codebook
        amp_state, meta, C_nt, dec_pth, cb_pt = load_pretrained_pair(args, device)
        print(f"Loaded decoder '{os.path.basename(dec_pth)}' "
              f"and codebook '{os.path.basename(cb_pt)}' "
              f"(init={meta.get('codebook_init','?')}, trainable={meta.get('codebook_trainable','?')}, "
              f"n={C_nt.size(0)}, d={C_nt.size(1)})")
        self.register_buffer("URA_codebook", C_nt)          # (n, dim), fixed
        self.n, self.dim = C_nt.shape

        # 2) Instantiate AMPNet with same options & load weights
        self.ampnet = AMPNet1DEnhanced(
            URA_codebook=C_nt,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            kernel_size=args.kernel_size,
            use_poisson=args.use_poisson,
            update_alpha=args.update_alpha,
            learn_sigma2=args.learn_sigma2,
            finetune_K=args.finetune_K,
            finetune_pi=args.finetune_pi,
            finetune_sigma2=args.finetune_sigma2,
            K_max=(None if args.K_max < 0 else int(args.K_max)),
            blend_init=args.blend_init,
            per_sample_sigma=args.per_sample_sigma,
            per_sample_alpha=args.per_sample_alpha,
        ).to(device)
        self.ampnet.load_state_dict(amp_state, strict=True)
        self.ampnet.eval()
        self.ampnet.set_init_const(100.0)

        # 3) Working quantizer codebook for this round (n x L), refreshed each round & criterion function
        self.register_buffer("Q_codebook", torch.zeros(args.n, args.message_split, device=device))
        self.criterion_ampnet = SimAMPNetLossWithSparsity().to(device)

    @property
    def C_syn(self):
        return self.URA_codebook
    
    def quantize(self, splits_flat, batch_size=2048):
        """Memory-efficient batched quantization"""
        total_splits = splits_flat.shape[0]
        indices_list = torch.empty(total_splits, dtype=torch.long, device=self.device)
        quant_words_list = torch.empty(total_splits, self.args.message_split, device=self.device)
        ura_words_list = torch.empty(total_splits, self.dim, device=self.device)

        for start_idx in range(0, total_splits, batch_size):
            end_idx = min(start_idx + batch_size, total_splits)
            batch_splits = splits_flat[start_idx:end_idx]
            dists = torch.cdist(batch_splits, self.Q_codebook, p=2)
            indices = torch.argmin(dists, dim=1)
            quant_words = self.Q_codebook[indices]
            ura_words = self.C_syn[indices]

            indices_list[start_idx:end_idx] = indices
            quant_words_list[start_idx:end_idx] = quant_words
            ura_words_list[start_idx:end_idx] = ura_words
        return indices_list, quant_words_list, ura_words_list

    def _aggregate_Ka(self, Ka_list, weights):
        if not Ka_list: return float("nan")
        return float(np.average(np.array(Ka_list), weights=np.array(weights)))

    def post_process_round(self, X_raw, K_int, y=None, ridge=1e-6):
        K_int = int(K_int)
        S, n = X_raw.shape
        out = []
        A_np = self.C_syn.t().detach().cpu().numpy()
        for i in range(S):
            xi = X_raw[i].detach().cpu().numpy()
            xk = top_k_nonneg(xi, K_int)
            if y is not None:
                yi = y[i].detach().cpu().numpy()
                xk = l2_refit_on_support(yi, A_np, xk, ridge=ridge, nonneg=True)
            xk = greedy_rounding(xk, K_int)
            out.append(torch.tensor(xk, device=X_raw.device, dtype=X_raw.dtype))
        return torch.stack(out, dim=0)

    def decode_round(self, y, S, target_vectors, pi_est, idx_mini, K_a, k_agg="mean", trim=0.0, z_clean=None):
        """
        Decodes one round with AMP-DA-Net using cached per-round parameters.
        Returns: recovered_flat, amp_loss (placeholder=0.0), K_est_final, acc, pupe, sigma2_final, snr_final_db
        """
        # 0) Init/caches
        device = y.device
        n = self.C_syn.size(0)
        self.ampnet.start_new_round(
            z_round=y,
            pi_round=pi_est,
            external_URA_codebook=self.C_syn
        )

        # 1) Run AMP-DA-Net across the round in chunks
        X_raw = torch.empty((S, n), device=device, dtype=y.dtype)
        Ka_vals, weights, sigma_last_vals = [], [], []
        loss_sum = 0.0
        sample_sum = 0

        for start in range(0, S, self.grad_accumulation_size):
            end = min(S, start + self.grad_accumulation_size)
            z_b = y[start:end]
            x_raw, k_raw, _, sigma_trace_b = self.ampnet(
                z_b, external_URA_codebook=self.C_syn, pi_round=pi_est, snr_db=None
            )

            X_raw[start:end] = x_raw.detach()
            target_vec_batch = target_vectors[start:end] 
            B = target_vec_batch.size(0)
            loss_b = self.criterion_ampnet(self, target_vec_batch, x_raw, K_a=K_a, K_final=float(k_raw))
            loss_sum += float(loss_b.item()) * B
            sample_sum += B

            Ka_vals.append(float(k_raw))
            sigma_last_vals.append(float(sigma_trace_b[-1]))
            weights.append(B)

        ampnet_loss = loss_sum / max(1, sample_sum)
        K_est_final  = self._aggregate_Ka(Ka_vals, weights)
        K_int_final  = int(round(max(0.0, min(float(n), K_est_final))))
        sigma2_final = float(np.average(np.array(sigma_last_vals), weights=np.array(weights)))

        # 2) Optional post rounding per split
        if self.post_rounding and K_int_final > 0:
            X_post = self.post_process_round(X_raw, K_int_final, y=y)
        else:
            X_post = X_raw

        # 3) Reconstruct aggregated update (still padded)
        recovered_flat = (X_post.float() @ self.Q_codebook).reshape(-1)

        # 4) Metrics (vs integer targets)
        Xp_np = X_post.detach().cpu().numpy()
        T_np  = target_vectors.detach().cpu().numpy()
        acc_vec  = compute_accuracy_batch(T_np, Xp_np)
        pupe_vec = [compute_pupe(T_np[i], Xp_np[i]) for i in range(len(Xp_np))]
        ampnet_acc  = float(np.mean(acc_vec))
        ampnet_pupe = float(np.mean(pupe_vec))

        # final SNR using clean power per-split if provided; else from y-energy minus σ²
        if z_clean is not None:
            P_sig_per = z_clean.pow(2).mean(dim=1)  # (S,)
            snr_lin_per = (P_sig_per / max(1e-12, sigma2_final)).detach().cpu().numpy()
            snr_final_db = 10.0 * math.log10(max(1e-12, float(np.mean(snr_lin_per))))
        else:
            Py = float(y.pow(2).mean().item())
            P_sig = max(0.0, Py - float(sigma2_final))  # unbiased if y = signal + AWGN
            snr_final_db = 10.0 * math.log10(max(1e-12, P_sig) / max(1e-12, float(sigma2_final)))

        return recovered_flat, K_est_final, K_int_final, ampnet_loss, ampnet_acc, ampnet_pupe, sigma2_final, snr_final_db

    def forward(self, deltas_w_feedback, args):
        """
        Round pipeline:
          (a) build Q via k-means++ on base-station mini dataset,
          (b) compute idx_mini + π̂, then reorder {Q, idx_mini, π̂} by args.code_order (NOT URA),
          (c) quantize all device splits with Q,
          (d) build targets + π_target, apply same permutation,
          (e) sum URA words, add noise, decode with AMP-DA-Net.
        """
        K_a, D = deltas_w_feedback.shape
        S, L = args.num_splits, args.message_split

        # 1) Update quantisation codebook & base station mini dataset
        bs_mini = deltas_w_feedback[0]
        pad = (-bs_mini.numel()) % L
        if pad > 0:
            bs_mini = torch.cat([bs_mini, bs_mini.new_zeros(pad)])
        bs_blocks = bs_mini.view(-1, L).cpu().numpy()
        centers, _ = kmeans_plusplus(bs_blocks, n_clusters=args.n, random_state=0)
        new_Q = torch.tensor(centers, device=self.device, dtype=self.Q_codebook.dtype)

        # 2) Estimate pi distribution & apply codebook ordering
        mini = torch.from_numpy(bs_blocks).to(self.device)
        dists_mini = torch.cdist(mini, new_Q, p=2)
        idx_mini = torch.argmin(dists_mini, dim=1)                  # (S,)
        pi_est = torch.bincount(idx_mini, minlength=args.n).float()
        pi_est = (pi_est / pi_est.sum().clamp_min(1e-12)).contiguous()

        # 2.5) Re-order quantisation codebook, idx_mini & pi distribution
        perm_used, inv_perm_used = get_round_perm(args.code_order, new_Q, idx_mini, pi_est)
        with torch.no_grad():
            self.Q_codebook.copy_(new_Q if perm_used is None else new_Q[perm_used])
        pi_est = pi_est if perm_used is None else pi_est[perm_used]
        idx_mini = idx_mini if inv_perm_used is None else inv_perm_used[idx_mini]

        # 3) Message splitting, padding and reshaping
        pad = (-D) % L
        if pad > 0:
            padding = deltas_w_feedback.new_zeros((K_a, pad))
            deltas_w_feedback = torch.cat([deltas_w_feedback, padding], dim=1)
        deltas_flat = deltas_w_feedback.view(-1, L)  # (K_a * S, L)

        # 4) Quantize & update error feedback
        idx_all, Qw_all, Uw_all = self.quantize(deltas_flat, batch_size=4096)
        new_err_flat = deltas_flat - Qw_all

        # 5) Compute pi target & target vectors for logging
        idx_mat = idx_all.view(K_a, S)                     # (K_a, S)
        target_vectors = F.one_hot(idx_mat, num_classes=args.n).sum(dim=0).float()  # (S, n)

        # 6) Sum URA words, add noise & decode
        z_clean = Uw_all.view(K_a, S, -1).sum(0)  # (S,d)
        y = add_awgn_noise(z_clean, args.snr_db)
        recovered, K_est, K_int, ampnet_loss, ampnet_acc, ampnet_pupe, sigma2_final, snr_final_db = \
            self.decode_round(
                y, S, target_vectors,
                pi_est=pi_est,
                idx_mini=idx_mini,
                K_a=K_a, k_agg="mean", trim=0.0, z_clean=z_clean
            )
        
        # 7) Remove padding & reshape quantisation error updates
        new_errors = new_err_flat.view(K_a, -1)[:, :D]
        return recovered[:D], new_errors, ampnet_loss, K_est, K_int, ampnet_acc, ampnet_pupe, sigma2_final, snr_final_db


def federated_training(
    global_model: torch.nn.Module,
    device_loaders: list[DataLoader],
    test_loader: DataLoader,
    args,
    device: torch.device,
):
    """
    Federated learning with wireless channel compression using AMP-DA-Net
    Args:
        global_model: the ResNetS model
        device_loaders: list of DataLoaders, one per device with on-fly transformations
        test_loader: DataLoader for CIFAR-10 test set
        args: parsed arguments containing all hyperparameters
        device: torch device (cuda/cpu)
    """
    # 0) Start timer, dispaly setup info & define transforms
    training_start = time.time()
    max_duration = 23 * 3600

    D = sum(p.numel() for p in global_model.parameters())
    args.num_splits = math.ceil(D / args.message_split)  # Number of message-splits per device
    print(f"Global model has {D} parameters.")
    UL_overhead = compute_ul_overhead(D, args.message_split, args.dim, P=1024, complex=False)
    print(f"UL communication overhead: {UL_overhead} time slots per round")
    bpd = compute_bits_per_dimension(args.n, args.message_split)
    print(f"Bits per dimension: {bpd:.3f} bits/dim")


    # 1) Global optimizer & history init
    momentum_buffer_list = [torch.zeros_like(p.data, device=device) for p in global_model.parameters()]
    exp_avgs = [torch.zeros_like(p.data, device=device) for p in global_model.parameters()]
    exp_avg_sqs = [torch.zeros_like(p.data, device=device) for p in global_model.parameters()]
    Q_errors = [torch.zeros((D,), device=device) for _ in range(args.K_t)]

    history = {
        'ampnet_loss': [], 'ampnet_acc': [], 'ampnet_pupe': [],
        'test_acc': [], 'K_a': [], 'K_est': [], 'sigma2_final': [], 'snr_final': [], 'mean_local_loss': [],
    }
    best_test_acc = -1.0; rounds_no_improve = 0

    # 3) Load pretrained WirelessCompressor
    compressor = WirelessCompressor(args, device).to(device)

    # 4) Run global rounds
    for rnd in range(1, args.total_rounds + 1):
        global_state = global_model.state_dict()
        par_before = [p.data.clone() for p in global_model.parameters()] # captures only the trainable parameters

        # 4a) Sample participants
        K_a = np.random.randint(args.min_p, args.max_p + 1)
        participants = np.random.choice(args.K_t, K_a, replace=False)
        print(f"Participants in round {rnd}: {participants}")

        # 4b) Local training & delta collection
        local_state_dicts, local_losses = [], []
        for i, dev_id in tqdm(enumerate(participants), desc=f"Round {rnd} — Local training", total=len(participants), ncols=80, leave=True):
            local_model = build_global_model(args).to(device)
            local_model.load_state_dict(global_state)
            opt = optim.SGD(local_model.parameters(), lr=args.local_lr)
            
            # 4c) Split local device's dataset into train/val/test
            full_subset = device_loaders[dev_id].dataset
            device_full_indices = np.array(full_subset.indices)
            np.random.shuffle(device_full_indices)
            n = len(device_full_indices)
            n_train = int(0.8 * n)

            train_global_idxs = device_full_indices[:n_train].tolist()
            base_dataset = device_loaders[dev_id].dataset.dataset
            train_subset = Subset(base_dataset, train_global_idxs)
            train_loader = DataLoader(
                train_subset,
                batch_size=args.local_batch_size,
                shuffle=True, 
                num_workers=2, 
                pin_memory=True
            )

            # 4d) Run local training
            local_model.train()
            total_samples = 0
            total_loss_sum = 0.0
            for _ in range(args.local_epochs):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    logits = local_model(xb)
                    loss = F.cross_entropy(logits, yb)
                    loss.backward()
                    opt.step()

                    batch_n = yb.size(0)
                    total_samples += batch_n
                    total_loss_sum += loss.item() * batch_n

            avg_loss = total_loss_sum / max(1, total_samples)
            print(f"Device {dev_id} sample-weighted avg loss: {avg_loss:.4f}")
            local_losses.append(avg_loss)
            local_state_dicts.append(local_model.state_dict())
            del local_model, opt

        # 4e) Compute per-device deltas explicitly
        param_names = [name for name, _ in global_model.named_parameters()]
        global_flat = torch.cat([global_state[name].view(-1) for name in param_names])

        local_deltas = []
        for local_state in local_state_dicts:
            local_flat = torch.cat([local_state[name].view(-1) for name in param_names])
            local_deltas.append(local_flat - global_flat)
        deltas = torch.stack(local_deltas, dim=0)  # Shape: (K_a, D)

        # 4f) Add quantisation error feedback
        error_list = [Q_errors[dev_id].detach() for dev_id in participants]
        error_matrix = torch.stack(error_list, dim=0)  # (K_a, D)
        deltas_w_feedback = deltas + error_matrix
        error_feedback_info(deltas, error_matrix)

        # 4g) Compress via wireless channel
        compressor.eval() 
        recovered_update, new_errors, amp_loss, K_est, K_int, amp_acc, amp_pupe, sigma2_est, snr_final_db = compressor(deltas_w_feedback, args=args)
        print(f"Compression done, K_true={K_a}, K_est={K_est:.3f} (K_int={K_int}), AMP-DA-Net Loss={amp_loss:.4f}, SNR≈{snr_final_db:.2f} dB")

        history['ampnet_loss'].append(amp_loss)
        history['ampnet_acc'].append(amp_acc)
        history['ampnet_pupe'].append(amp_pupe)
        history['mean_local_loss'].append(np.average(local_losses, weights=None))
        history['sigma2_final'].append(sigma2_est)
        history['snr_final'].append(snr_final_db)

        # 4h) Update quantisation errors per device
        for i, dev_id in enumerate(participants): 
            Q_errors[dev_id] = new_errors[i].detach()

        # 4i) Build averaged delta list
        avg_delta = []
        pointer = 0
        for p in par_before:
            n_param = p.numel()
            slice_ = recovered_update[pointer:pointer+n_param].view(p.size())
            avg_delta.append((slice_ / max(1, K_int)).detach())
            pointer += n_param
        assert pointer == recovered_update.numel(), "Mismatch in recovered update size!"

        # 4j) Updates BatchNorm statistics via averaging
        buffer_updates = {}
        for buf_name, _ in global_model.named_buffers():
            summed = sum(local_sd[buf_name] for local_sd in local_state_dicts)
            buffer_updates[buf_name] = summed / max(1, K_int)
        global_model.load_state_dict(buffer_updates, strict=False)

        # 4k) Update global model parameters via optimizer
        update_model_inplace(
            global_model,
            par_before,
            avg_delta,
            args,
            cur_iter=rnd - 1,
            momentum_buffer_list=momentum_buffer_list,
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
        )
        print("Global model parameters updated.")

        # 4l) Evaluate global model on test set
        global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = global_model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total

        # 4m) Update history & early stopping
        history['test_acc'].append(acc); history['K_a'].append(K_a); history['K_est'].append(K_est)
        print(f"Round {rnd}/{args.total_rounds} | K_true={K_a} K_est={K_est:.2f} (K_int={K_int}) | AMP-Acc={amp_acc:.3f} PUPE={amp_pupe:.3f} | TestAcc={acc:.4%}")

        if acc > best_test_acc:
            best_test_acc = acc
            best_global_model_state = copy.deepcopy(global_model.state_dict())
            best_rnd = rnd
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1
            if rounds_no_improve >= args.early_stopping_patience:
                print(f"Early stopping at round {rnd}. Best: {best_rnd} ({best_test_acc:.4%})")
                break
        if (time.time() - training_start) >= max_duration:
            print("⏱️ Time limit reached, stopping.")
            break
    return history, best_global_model_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wireless Federated Learning with AMP-DA-Net Compression")

    # === Dataset Parameters ===
    parser.add_argument('--data-dir',         type=str, default="runs/datasets", help="path to CIFAR-10 dataset")
    parser.add_argument('--batch-size-train', type=int, default=16, help="batch size for training")
    parser.add_argument('--batch-size-test',  type=int, default=16, help="batch size for testing")
    parser.add_argument('--frac-random',      type=float, default=0.2, help="fraction of random samples per device")
    
    # === Federated Learning Parameters ===
    parser.add_argument('--K-t',              type=int, default=40, help="total number of devices") # Make sure is a factor of 50,000
    parser.add_argument('--min-p',            type=int, default=2, help="minimum participants per round")
    parser.add_argument('--max-p',            type=int, default=2, help="maximum participants per round")
    parser.add_argument('--total-rounds',     type=int, default=5, help="total number of global rounds")
    parser.add_argument('--local-epochs',     type=int, default=3, help="number of local epochs per round")
    parser.add_argument('--local-batch-size', type=int, default=20, help="local training batch size")
    parser.add_argument('--local-lr',         type=float, default=0.01, help="local learning rate")
    
    # === Global Optimizer Parameters ===
    parser.add_argument('--optimizer',        type=str, default='fedavg', choices=['fedavg', 'fedavgm', 'fedadam'], help="global optimizer type")
    parser.add_argument('--global-lr',        type=float, default=1.0, help="global learning rate")
    parser.add_argument('--momentum',         type=float, default=0.0, help='SGD momentum')
    parser.add_argument('--beta1',            type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2',            type=float, default=0.99, help='Adam beta2')
    parser.add_argument('--eps',              type=float, default=0, help='Adam epsilon')
    
    # === Wireless Channel & Compression Parameters ===
    parser.add_argument('--message-split',    type=int, default=10, help="message split size & quantization dimension")
    parser.add_argument('--n',                type=int, default=256, help="number of codewords in codebooks")
    parser.add_argument('--dim',              type=int, default=64, help="encoding dimension")
    parser.add_argument('--snr-db',           type=float, default=20.0, help="runtime channel SNR used in FL")
    parser.add_argument('--snr-mode',         type=str, default='range', choices=['fixed','range'], help="used only for decoder loading. runtime channel still uses --snr-db")
    parser.add_argument('--snr-min-db',       type=float, default=0.0,  help="for slug when --snr-mode=range")
    parser.add_argument('--snr-max-db',       type=float, default=20.0, help="for slug when --snr-mode=range")

    parser.add_argument("--post-rounding",    action=argparse.BooleanOptionalAction, default=True, help="apply post-rounding (use --no-post-rounding to disable)")
    parser.add_argument('--grad-accumulation-size', type=int, default=128, help="gradient accumulation size for decoding")
    parser.add_argument('--code-order',       type=str, default="pop", choices=["none", "pop", "spectral", "spectral_pop"])

    # === AMPNet Parameters ===
    parser.add_argument('--num-layers',       type=int, default=10, help="number of AMPNet layers")
    parser.add_argument('--num-filters',      type=int, default=32, help="number of filters in AMPNet")
    parser.add_argument('--kernel-size',      type=int, default=3, help="kernel size in AMPNet")
    parser.add_argument('--use-poisson',      action=argparse.BooleanOptionalAction, default=True, help="Use spike+Poisson counts prior (use --no-use-poisson for uniform counts)")
    parser.add_argument('--update-alpha',     action=argparse.BooleanOptionalAction, default=True, help="Enable per-layer alpha prior-posterior mixing")
    parser.add_argument('--learn-sigma2',     action=argparse.BooleanOptionalAction, default=True, help="Enable σ² EM head for noise-variance updates.")
    parser.add_argument('--finetune-K',       action=argparse.BooleanOptionalAction, default=True, help="Enable learned refinement of K per layer.")
    parser.add_argument('--finetune-pi',      action=argparse.BooleanOptionalAction, default=True, help="Enable learned refinement of π per layer.")
    parser.add_argument('--finetune-sigma2',  action=argparse.BooleanOptionalAction, default=True, help="Blend σ² via learned step size (requires --learn-sigma2).")
    parser.add_argument('--per-sample-sigma', action=argparse.BooleanOptionalAction, default=True, help="Track per-sample σ² (B,1) instead of a scalar.")
    parser.add_argument('--per-sample-alpha', action=argparse.BooleanOptionalAction, default=False, help="Use per-sample alpha instead of batch-averaged alpha.")
    parser.add_argument('--K-max',            type=int, default=-1, help="Fixed discrete prior cap K_max (≤ n) -1 represents adaptive (2*Ka)")
    parser.add_argument('--blend-init',       type=float, default=0.85, help="Initial CNN blend weight in (0,1) for m1-CNN")
    
    # === Compressor Loading ===
    parser.add_argument('--ckpt-dir',         type=str, default="runs/checkpoints/pretrain")
    parser.add_argument('--decoder-slug',     type=str, default="", help="override auto slug")
    parser.add_argument('--decoder-path',     type=str, default="", help="explicit path to <slug>.pth")
    parser.add_argument('--codebook-path',    type=str, default="", help="explicit path to <slug>_cb.pt")
    parser.add_argument('--codebook-init',    type=str, default='gaussian', choices=['q_init', 'gaussian', 'bernoulli'], help="codebook initializer used during pretraining (for artifact selection)")
    parser.add_argument('--codebook-trainable', action=argparse.BooleanOptionalAction, default=True, help="whether URA codebook was trainable during pretraining (for artifact selection)")
    parser.add_argument('--within-round',     type=str, default="random", choices=["prefix", "contig_rand", "random"], help="Sampling strategy within each round")

    # === Global Model Selection ===
    parser.add_argument('--model',            type=str, default='resnet', choices=['resnet', 'cifarcnn', 'custom'])
    parser.add_argument('--custom-model',     type=str, default='', help="module.path:ClassName")
    parser.add_argument('--custom-kwargs',    type=str, default='{}')

    # === System Parameters ===
    parser.add_argument('--early-stopping-patience', type=int, default=20)
    parser.add_argument('--save-dir',         type=str, default="runs/results")
    parser.add_argument('--seed',             type=int, default=42)

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    if torch.cuda.is_available():
        args.device = torch.device('cuda') 
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        args.device = torch.device('cpu')
        print("GPU not available, using CPU")

    # Load dataset and create model
    device_loaders, test_loader = load_cifar_datasets(
        data_dir=args.data_dir,
        num_devices=args.K_t,
        frac_random=args.frac_random,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
    )
    global_model = build_global_model(args).to(args.device)
    torch.cuda.empty_cache()

    # Run federated training
    history, best_global_state = federated_training(global_model, device_loaders, test_loader, args, args.device)

    # Print loss & accuracy curves
    plot_path = os.path.join(args.save_dir, "inference_history_plots")
    plot_inference_history(history, plot_path, snr_true_db=args.snr_db)

    # Persist round-by-round history (JSON + CSV)
    summ_dir = os.path.join(args.save_dir, "summaries"); os.makedirs(summ_dir, exist_ok=True)
    save_json(history, os.path.join(summ_dir, "history.json"))
    save_round_history_csv(history, os.path.join(summ_dir, "history_rounds.csv"))

    # Final summary (useful for tables across runs)
    final_summary = dict(
        metric_source="federated",
        total_rounds=len(history["test_acc"]),
        final_test_acc=float(history["test_acc"][-1]) if history["test_acc"] else None,
        best_test_acc=float(max(history["test_acc"])) if history["test_acc"] else None,
        best_round=(int(np.argmax(history["test_acc"])) + 1) if history["test_acc"] else None,
        final_ampnet_acc=float(history["ampnet_acc"][-1]) if history["ampnet_acc"] else None,
        final_ampnet_pupe=float(history["ampnet_pupe"][-1]) if history["ampnet_pupe"] else None,
        final_K_est=float(history["K_est"][-1]) if history["K_est"] else None,
        final_sigma2=float(history["sigma2_final"][-1]) if history["sigma2_final"] else None,
        final_snr_db=float(history["snr_final"][-1]) if history["snr_final"] else None,
    )
    save_json(final_summary, os.path.join(summ_dir, "final_summary.json"))
    append_jsonl({**run_manifest(args), **final_summary}, os.path.join(summ_dir, "runs.jsonl"))

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(history, os.path.join(args.save_dir, "history.pth"))
    torch.save(best_global_state, os.path.join(args.save_dir, "final_global_model.pth"))
    print(f"Training completed. Results saved to {args.save_dir}")
