
import os
import copy
import math
import time
import warnings
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.cluster import kmeans_plusplus

from src.codebooks import make_perm_popularity, make_perm_spectral
from src.models import _model_factory_from_args
from src.federated import update_model_inplace
from src.paths import _norm_order_tag, make_dataset_slug
from src.utils import EarlyStopNaN, _ensure_finite
from src.analysis import (
    analyze_dataset, 
    plot_dataset_collection_history, 
    plot_codebook_ordering, 
    compute_pi_distribution,
    compute_bits_per_dimension, 
    compute_ul_overhead, 
    error_feedback_info
)


class TargetDatasetCollector:
    """ Currently saves:
        Target indices per split: X
        Per-round Kₐ: K_rounds
        Per-round true π: pi_targets
        Per-round estimated π: pi_estimates
        BS one-hot index vectors per split: device_idx
        Number of splits per round (S): round_len

        Now also saves:
        - meta: dict with dataset generation knobs (n, dim, message_split, code_order, K_t, p-range, frac_random, seed)
        - filename derived from a deterministic slug that encodes those knobs
    """
    def __init__(self, args, save_root: str | None = None):
        self.X_list = []
        self.K_rounds = []
        self.pi_targets = []
        self.pi_estimates = []
        self.device_idx_list = []
        self.round_len = None

        # meta captured once
        order_tag = _norm_order_tag(args.code_order)
        model_tag = getattr(args, 'model_tag', 'resnet')
        self.meta = {
            "kind": "ifed_sparse_vectors",
            "version": 1,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # model info
            "model": model_tag,
            "model_class": getattr(args, 'model_classname', None),
            "model_num_params": getattr(args, 'model_num_params', None),
            # hard-compat knobs
            "n": int(args.n),
            "dim": int(args.dim),
            "message_split": int(args.message_split),
            "code_order": order_tag,                   # "pop" | "spectral" | "spectral+pop" | "none"
            # generation context
            "K_t": int(args.K_t),
            "min_p": int(args.min_p),
            "max_p": int(args.max_p),
            "frac_random": float(args.frac_random),
            "seed": int(args.seed),
        }

        slug = make_dataset_slug(
            model=model_tag,
            order=order_tag, n=self.meta["n"], dim=self.meta["dim"],
            msg=self.meta["message_split"], Kt=self.meta["K_t"],
            pmin=self.meta["min_p"], pmax=self.meta["max_p"],
            frac_random=self.meta["frac_random"], seed=self.meta["seed"]
        )

        base = save_root or os.path.join(args.data_dir, "iFed_datasets")
        os.makedirs(base, exist_ok=True)
        self.out_path = os.path.join(base, f"{slug}.pth")
        print(f"[dataset] Will save to: {self.out_path}")

    def add(self, target_vectors, K_a, pi_target, pi_estimate, device_idx_round):
        self.X_list.append(target_vectors.detach().cpu())
        self.K_rounds.append(int(K_a))
        self.pi_targets.append(pi_target.detach().cpu())
        self.pi_estimates.append(pi_estimate.detach().cpu())
        self.device_idx_list.append(device_idx_round.detach().cpu())
        if self.round_len is None:
            self.round_len = int(target_vectors.shape[0])

    def save(self):
        X = torch.cat(self.X_list, dim=0) if self.X_list else torch.empty((0,0))
        K_rounds = torch.tensor(self.K_rounds, dtype=torch.long)
        pi_targets = torch.stack(self.pi_targets, dim=0) if self.pi_targets else torch.empty((0,0))
        pi_estimates = torch.stack(self.pi_estimates, dim=0) if self.pi_estimates else torch.empty((0,0))
        device_idx = torch.cat(self.device_idx_list, dim=0) if self.device_idx_list else torch.empty((0,), dtype=torch.long)

        bundle = {
            'X': X,
            'K_rounds': K_rounds,
            'pi_targets': pi_targets,
            'pi_estimates': pi_estimates,
            'device_idx': device_idx,
            'round_len': self.round_len,
            'meta': self.meta,
        }
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        torch.save(bundle, self.out_path)
        print(f"[dataset] Saved bundle with {len(self.K_rounds)} rounds → {self.out_path}")


def load_cifar_datasets(
    data_dir: str,
    num_devices: int = 100,
    frac_random: float = 0.2,
    batch_size_train: int = 32,
    batch_size_test: int = 128,
    shuffle_train: bool = True,
):
    # 0) Define exactly the AirComp transformation pipelines
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 1) Load full CIFAR10 via torchvision (handles download & pickles)
    full_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    full_test = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    # 2) Build the same non‑IID index splits (rand + global‑sort shard)
    N = len(full_train)  # 50000
    local_per_dev = N // num_devices
    rand_per_dev = int(local_per_dev * frac_random)
    shard_size = local_per_dev - rand_per_dev

    if N % num_devices != 0:
        warnings.warn(f"{N} not divisible by {num_devices}, dropping {N%num_devices}")

    # 2a) shuffle all indices, carve off the random chunk & reshape
    all_idxs = np.arange(N)
    np.random.shuffle(all_idxs)
    rand_idxs = all_idxs[: num_devices * rand_per_dev]
    rem_idxs = all_idxs[num_devices * rand_per_dev :]
    rand_blocks = rand_idxs.reshape(num_devices, rand_per_dev)

    # 2b) global sort of the remainder by label → shard_blocks
    rem_labels = np.array(full_train.targets)[rem_idxs]
    sorted_rem = rem_idxs[np.argsort(rem_labels)]
    shards = sorted_rem[: num_devices * shard_size]
    shard_blocks = shards.reshape(num_devices, shard_size)

    # 3) Create one DataLoader per device via Subset(full_train, idxs)
    device_loaders = []
    for dev in range(num_devices):
        idxs = np.concatenate([rand_blocks[dev], shard_blocks[dev]])
        subset = Subset(full_train, idxs.tolist())

        loader = DataLoader(
            subset,
            batch_size=batch_size_train,
            shuffle=shuffle_train,
            num_workers=2,
            pin_memory=True
        )
        device_loaders.append(loader)

    # 4) Single global test loader
    test_loader = DataLoader(
        full_test,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return device_loaders, test_loader


def quantize(splits_flat, args, L, Q_codebook, batch_size=256):
    """Memory-efficient batched quantization"""
    total_splits = splits_flat.shape[0]
    indices_list = torch.empty(total_splits, dtype=torch.long, device=args.device)
    quant_words_list = torch.empty(total_splits, L, device=args.device)
    for start_idx in range(0, total_splits, batch_size):
        end_idx = min(start_idx + batch_size, total_splits)
        batch_splits = splits_flat[start_idx:end_idx]
        dists = torch.cdist(batch_splits, Q_codebook, p=2)
        indices = torch.argmin(dists, dim=1)
        quant_words = Q_codebook[indices]
        quant_words_list[start_idx:end_idx] = quant_words
        indices_list[start_idx:end_idx] = indices
    return indices_list, quant_words_list


def quantize_process(deltas_w_feedback, args, K_a, dataset_collector):
    """ Quantizes the deltas with feedback and updates the error feedback."""
    # 0) Initialistaion
    K_a, D = deltas_w_feedback.shape
    S, L = args.num_splits, args.message_split

    # 1) Calculate k-means quantisation codebook
    bs_mini_dataset = deltas_w_feedback[0]
    pad = (-bs_mini_dataset.numel()) % L
    if pad > 0:
        bs_mini_dataset = torch.cat([bs_mini_dataset, bs_mini_dataset.new_zeros(pad)])
    bs_blocks = bs_mini_dataset.view(-1, L).cpu().numpy()

    # Additional Guard for Nans produced during training
    if not np.isfinite(bs_blocks).all():
        raise EarlyStopNaN("mini-dataset blocks (bs_blocks) contain NaN/Inf before k-means.")
    
    centers, _ = kmeans_plusplus(bs_blocks, n_clusters=args.n, random_state=0)
    new_Q = torch.tensor(centers, device=args.device)

    # 2) Estimate pi distribution
    mini = torch.from_numpy(bs_blocks).to(args.device)
    dists_mini = torch.cdist(mini, new_Q, p=2) 
    idx_mini = torch.argmin(dists_mini, dim=1)
    pi_est = torch.bincount(idx_mini, minlength=args.n).float()
    pi_est /= (pi_est.sum() + 1e-12)

    # 2.5) Re-order quantisation codebook, idx_mini & pi distribution
    # Q_before  = new_Q.detach().clone()
    # pi_before = pi_est.detach().clone()
    # idx_before = idx_mini.detach().clone()
    if str(args.code_order).lower() == "none":
        args.code_order = None
    if args.code_order is not None:
        if args.code_order == "pop":
            perm, inv_perm = make_perm_popularity(pi_est)
        elif args.code_order == "spectral":
            perm, inv_perm = make_perm_spectral(new_Q, idx_mini=idx_mini, pi_est=None, blend_pop=0.0)
        elif args.code_order == "spectral_pop":
            perm, inv_perm = make_perm_spectral(new_Q, idx_mini=idx_mini, pi_est=pi_est, blend_pop=1.0)
        else:
            raise ValueError(f"Unknown code order: {args.code_order}")
        new_Q = new_Q[perm]              # reorder codewords
        pi_est = pi_est[perm]            # reorder popularity to match
        idx_mini = inv_perm[idx_mini]    # remap old assignments to new ids
        # round_idx = len(dataset_collector.K_rounds) + 1
        # b0, b1 = plot_codebook_ordering(Q_before, pi_before, idx_before, perm, args, tag=f"round{round_idx:03d}")
        # print(f"[code-order] bandedness: {b0:.3f} → {b1:.3f} (Δ={b1-b0:+.3f})")

    # 3) Message splitting, padding and reshaping
    pad = (-D) % L
    if pad > 0:
        padding = torch.zeros((K_a, pad), device=args.device, dtype=deltas_w_feedback.dtype)
        deltas_w_feedback = torch.cat([deltas_w_feedback, padding], dim=1)
    deltas_flat = deltas_w_feedback.view(-1, L) # (K_a * S, L)

    # 4) Quantize, aggregate & update error feedback
    indices_list, quant_words = quantize(deltas_flat, args, L, new_Q, batch_size=256)

    # 5) Create target sparse vectors
    idx_mat = indices_list.view(K_a, S)  # (K_a, S)
    target_vectors = F.one_hot(idx_mat, num_classes=args.n).sum(dim=0).float()
    pi_target = compute_pi_distribution(target_vectors)  # (n,)

    device_idx_round = idx_mini.to(torch.long)  # (S,)
    if device_idx_round.numel() != target_vectors.shape[0]:  # Padding guard for shape mismatch
        device_idx_round = device_idx_round[:target_vectors.shape[0]]
    dataset_collector.add(target_vectors, K_a, pi_target, pi_est, device_idx_round)
    dataset_collector.save()

    new_err_flat = deltas_flat - quant_words # (K_a * S, L)
    print("Quantization, encoding & dataset saving done")
    if not torch.isfinite(new_err_flat).all():
        raise EarlyStopNaN("new_err_flat contains NaN/Inf after quantization.")

    # 4) Remove padding & reshape quantisation error updates
    new_errors = new_err_flat.view(K_a, -1)[:, :D] # (K_a, D)
    quant_words = quant_words.view(K_a, S, L)  # per‑device splits
    rec_per_dev = quant_words.reshape(K_a, -1)[:, :D]  # drop padding, back to (K_a, D)
    recovered_update = rec_per_dev.sum(dim=0)
    return recovered_update, new_errors


def federated_training(global_model, device_loaders, test_loader, args, device, make_model):
    """
    Federated learning with wireless channel compression using ISTANet
    Args:
        global_model: the ResNet model
        device_loaders: list of DataLoaders, one per device with on-fly transformations
        test_loader: DataLoader for CIFAR-10 test set
        args: parsed arguments containing all hyperparameters
        device: torch device (cuda/cpu)
    """
    # 0) Start timer and dispaly setup info
    training_start = time.time()
    max_duration = 23 * 3600

    D = sum(p.numel() for p in global_model.parameters())
    args.model_num_params = int(D)  # so the collector can embed it
    args.num_splits = math.ceil(D / args.message_split)  # Number of message-splits per device
    print(f"Global model has {D} parameters.")
    UL_overhead = compute_ul_overhead(D, args.message_split, args.dim, P=1024, complex=False)
    print(f"UL communication overhead: {UL_overhead} time slots per round")
    bpd = compute_bits_per_dimension(args.n, args.message_split)
    print(f"Bits per dimension: {bpd:.3f} bits/dim")

    # 1) Define transforms for train/test splits
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 2) Initialize optimizers, error feedback and history
    momentum_buffer_list = [torch.zeros_like(p.data, device=device) for p in global_model.parameters()]
    exp_avgs = [torch.zeros_like(p.data, device=device) for p in global_model.parameters()]
    exp_avg_sqs = [torch.zeros_like(p.data, device=device) for p in global_model.parameters()]

    dataset_collector = TargetDatasetCollector(args=args)
    device_participation_count = [0] * args.K_t
    Q_errors = [torch.zeros((D,), device=device) for _ in range(args.K_t)]

    history = {'test_acc': []}
    best_test_acc = -float("inf")
    best_global_model_state = copy.deepcopy(global_model.state_dict())  # safe fallback
    rounds_no_improve = 0

    # 4) Run global rounds
    for rnd in range(1, args.total_rounds + 1):
        global_state = global_model.state_dict()
        par_before = [p.data.clone() for p in global_model.parameters()] # captures only the trainable parameters

        # 4a) Sample participants
        K_a = np.random.randint(args.min_p, args.max_p + 1)
        participants = np.random.choice(args.K_t, K_a, replace=False)
        participation_info = [f"{p}({device_participation_count[p]})" for p in participants]
        print(f"Round {rnd} participants: {participation_info}")

        # 4b) Local training & delta collection
        local_state_dicts = []
        local_losses = []

        for i, dev_id in tqdm(enumerate(participants), desc=f"Round {rnd} — Local training", total=len(participants), ncols=80, leave=True):
            device_participation_count[dev_id] += 1
            local_model = make_model().to(device)
            local_model.load_state_dict(global_state)
            opt = optim.SGD(local_model.parameters(), lr=args.local_lr)
            
            # 4c) Split local device's dataset into train/val/test
            full_subset = device_loaders[dev_id].dataset
            device_full_indices = np.array(full_subset.indices)
            np.random.shuffle(device_full_indices)
            n = len(device_full_indices)
            n_train = int(0.8 * n)

            train_global_idxs = device_full_indices[:n_train].tolist()
            cifar_train_transform = datasets.CIFAR10(
                root=args.data_dir,
                train=True,
                download=False,
                transform=transform_train
            )

            train_subset = Subset(cifar_train_transform, train_global_idxs)
            train_loader = DataLoader(
                train_subset,
                batch_size=args.batch_size_train,
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
            if not math.isfinite(avg_loss):
                raise EarlyStopNaN(f"Device {dev_id} produced non-finite average loss.")
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

        # Guarding against NaNs
        _ensure_finite(deltas, "deltas")
        _ensure_finite(error_matrix, "error_matrix")
        _ensure_finite(deltas_w_feedback, "deltas_w_feedback")

        try:
            # 4g) Perfect aggregation of outputted deltas into one flat update
            recovered_update, new_errors = quantize_process(deltas_w_feedback, args, K_a, dataset_collector)
        except EarlyStopNaN as e:
            print(f"[NaN-guard] {e} Stopping training early at round {rnd}.")
            break

        # 4h) Update quantisation errors per device
        for i, dev_id in enumerate(participants): 
            Q_errors[dev_id] = new_errors[i].detach()

        # 4i) Average deltas across participants
        avg_delta = []
        pointer = 0
        for p in par_before:
            n_param = p.numel()
            slice_ = recovered_update[pointer:pointer+n_param].view(p.size())
            avg_delta.append((slice_ / deltas.size(0)).detach())
            pointer += n_param
        assert pointer == recovered_update.numel(), "Mismatch in recovered update size!"

        buffer_updates = {}
        for buf_name, _ in global_model.named_buffers():
            summed = sum(local_sd[buf_name] for local_sd in local_state_dicts)
            buffer_updates[buf_name] = summed / K_a
        global_model.load_state_dict(buffer_updates, strict=False)

        # 4j) Update global model parameters via optimizer
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

        # 4k) Evaluate global model on test set
        global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = global_model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total

        # 4l) Update history & early stopping
        history['test_acc'].append(acc)
        print(f"Round {rnd}/{args.total_rounds} | Test Acc: {acc:.4%}")

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

        torch.cuda.empty_cache()
        elapsed = time.time() - training_start
        if elapsed >= max_duration:
            print(f"⏱️  Time limit reached ({elapsed/3600:.2f} h), stopping at round {rnd}.")
            break

    return history, best_global_model_state, dataset_collector.out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Collection via Wireless Federated Edge Learning with Perfect Aggregation")
    
    # === Pretraining Parameters ===
    parser.add_argument('--train-size',       type=int,   default=400,help="number of training samples")
    parser.add_argument('--valid-size',       type=int,   default=100,help="number of validation samples")
    parser.add_argument('--test-size',        type=int,   default=100,help="number of test samples")
    parser.add_argument('--pt-epochs',        type=int,   default=5,help="pretraining epochs")
    parser.add_argument('--pt-batch-size',    type=int,   default=16,help="training batch size")
    parser.add_argument('--pt-patience',      type=int,   default=5,help="early-stop patience")
    parser.add_argument('--sched-patience',   type=int,   default=5, help="scheduler patience")
    parser.add_argument('--sched-factor',     type=float, default=0.5, help="scheduler factor")
    parser.add_argument('--code-order',       type=str,   default="pop", choices=["none", "spectral", "spectral_pop", "pop"], help="codebook ordering strategy")

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
    parser.add_argument('--snr-db',           type=float, default=20, help="signal-to-noise ratio in dB")
    parser.add_argument('--temperature',      type=float, default=0.01, help="temperature for soft quantization")
    parser.add_argument('--quant-trainable',  action='store_false', help="make quantization codebook trainable")
    parser.add_argument('--post-rounding',    action='store_true', help="apply post-rounding")
    parser.add_argument('--grad-accumulation-size', type=int, default=128, help="gradient accumulation size for decoding")
    
    # === ISTANet Parameters ===
    parser.add_argument('--num-layers',       type=int, default=10, help="number of ISTANet layers")
    parser.add_argument('--num-filters',      type=int, default=32, help="number of filters in ISTANet")
    parser.add_argument('--kernel-size',      type=int, default=3, help="kernel size in ISTANet")
    
    # === Compressor Training Parameters ===
    parser.add_argument('--amp-lr',          type=float, default=0.0001, help="compressor learning rate")
    parser.add_argument('--lambda-sparse',    type=float, default=0.001, help="sparsity regularization weight")
    parser.add_argument('--lambda-w',         type=float, default=0.001, help="W matrix regularization weight")
    
    # === Training Control Parameters ===
    parser.add_argument('--early-stopping-patience', type=int, default=20, help="early stopping patience")
    parser.add_argument('--scheduler-patience', type=int, default=10, help="LR scheduler patience")
    parser.add_argument('--scheduler-factor', type=float, default=0.5, help="LR scheduler reduction factor")

    # === Model Selection ===
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'cifarcnn', 'custom'], help="which global model to use for data collection")
    parser.add_argument('--custom-model', type=str, default='', help="for --model custom: 'module.path:ClassName'")  # Untested for custom models
    parser.add_argument('--custom-kwargs', type=str, default='{}', help='for --model custom: JSON dict of ctor kwargs, e.g. {"num_classes":10}')  # Untested for custom models

    # === System Parameters ===
    parser.add_argument('--save-dir',         type=str, default="runs/results", help="directory to save results")
    parser.add_argument('--seed',             type=int, default=42, help="random seed")
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
        batch_size_test=args.batch_size_test
    )
    # global_model = ResNetSimple().to(args.device)
    make_model, model_tag, model_clsname = _model_factory_from_args(args)
    args.model_tag = model_tag
    args.model_classname = model_clsname
    global_model = make_model().to(args.device)

    torch.cuda.empty_cache()

    # Run federated training
    history, final_global_model, dataset_path = federated_training(
        global_model=global_model,
        device_loaders=device_loaders,
        test_loader=test_loader,
        args=args,
        device=args.device,
        make_model=make_model,
    )

    # Print accuracy curve & dataset analysis
    plot_dataset_collection_history(history, os.path.join(args.save_dir, "history_plots"))
    analysis_dir = os.path.join(args.save_dir, 'analysis')
    analyze_dataset(path=dataset_path, args=args, rounds_to_plot=None, save_dir=analysis_dir)

