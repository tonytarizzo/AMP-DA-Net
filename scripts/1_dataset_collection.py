import os
import copy
import math
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.datasets import load_cifar_datasets
from src.models import _model_factory_from_args
from src.federated import update_model_inplace
from src.paths import make_raw_dataset_slug, _fmt_float
from src.utils import EarlyStopNaN, _ensure_finite
from src.analysis import plot_dataset_collection_history


class TargetDatasetCollector:
    """
    Collector that stores per-device raw updates (split into L-sized blocks).
    Quotas (train/valid/test sizes) are interpreted as number of future sparse vectors/fragments.
    """
    def __init__(self, args, save_root: str = "datasets/iFed_datasets",
                 train_round_ids=None, valid_round_ids=None, test_round_ids=None):
        self.args = args
        self.L = int(args.message_split)
        self.D = int(args.model_num_params)

        self.frag_quota = {"train": int(args.train_size), "valid": int(args.valid_size), "test": int(args.test_size)}
        self.frag_remaining = self.frag_quota.copy()
        
        # buffers: flat rows and per-fragment descriptors
        self.buf = {"train": [], "valid": [], "test": []}     # list of (N_i, L) tensors
        self.frags = {"train": [], "valid": [], "test": []}   # list of dicts with metadata

        # per-round constants/meta
        self.S = math.ceil(self.D / self.L)  # blocks per device per round
        self.mode = str(getattr(args, "within_round", "prefix")).lower()
        assert self.mode in {"prefix", "contig_rand", "random"}, "Invalid within_round mode."

        # round-id splits (0-based)
        self.round_ids = {"train": list(train_round_ids or []), "valid": list(valid_round_ids or []), "test": list(test_round_ids or [])}
        self.K_rounds = []  # Records one K_a per round
        self.bs_local_blocks = {}  # Records local base station training raw values

        # slug/meta/output path
        model_tag = getattr(args, 'model_tag', 'resnet')
        slug = make_raw_dataset_slug(
            model=model_tag, D=self.D, L=self.L,
            Kt=int(args.K_t), pmin=int(args.min_p), pmax=int(args.max_p),
            frac_random=float(args.frac_random), seed=int(args.seed)
        )
        os.makedirs(save_root, exist_ok=True)
        self.out_path = os.path.join(save_root, f"{slug}.pth")
        self.meta = {
            "kind": "ifed_raw_per_device",
            "version": 2,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model_tag,
            "model_class": getattr(args, 'model_classname', None),
            "model_num_params": self.D,
            "message_split": self.L,
            "K_t": int(args.K_t),
            "min_p": int(args.min_p),
            "max_p": int(args.max_p),
            "frac_random": float(args.frac_random),
            "seed": int(args.seed),
            "round_ids": self.round_ids,
        }
        self.row_dtype = None
        print(f"[dataset] Will save → {self.out_path}  | L={self.L}  | S(per-device)={self.S}")

    def _pick_split_for_round(self, r0: int) -> str:
        if r0 in self.round_ids["test"]:
            return "test"
        if r0 in self.round_ids["valid"]:
            return "valid"
        return "train"
    
    def _rounds_left_in_split(self, split: str, r0: int) -> int:
        ids = self.round_ids[split]
        return sum(1 for x in ids if x >= r0)

    def _select_positions(self, S: int, k: int, rnd_seed: int) -> torch.Tensor:
        """Choose k fragment positions in [0..S-1] according to mode."""
        k = max(0, min(int(k), S))
        if k == 0:
            return torch.empty(0, dtype=torch.long)
        device = torch.device("cpu")
        g = torch.Generator(device=device).manual_seed(self.args.seed + 97 * rnd_seed)
        if self.mode == "prefix":
            return torch.arange(0, k, dtype=torch.long)
        if self.mode == "contig_rand":
            max_start = max(0, S - k)
            start = 0 if max_start == 0 else int(torch.randint(0, max_start + 1, (1,), generator=g).item())
            return torch.arange(start, start + k, dtype=torch.long)
        # random
        return torch.randperm(S, generator=g)[:k]

    @torch.no_grad()
    def add_bs_local(self, bs_delta_D: torch.Tensor, r1_based: int):
        """
        Record base station local training delta for this round.
        Input:
            bs_delta_D: (D,) or (1, D) float tensor on any device.
        Saves (S, L) blocks after zero-padding the tail to multiple of L.
        """
        r0 = int(r1_based) - 1
        bs_delta_D = bs_delta_D.view(-1).to(dtype=self.row_dtype or bs_delta_D.dtype)
        assert bs_delta_D.numel() == self.D, f"BS delta length {bs_delta_D.numel()} != D={self.D}"

        pad = (-self.D) % self.L
        if pad > 0:
            bs_delta_D = torch.cat([bs_delta_D, bs_delta_D.new_zeros(pad)], dim=0)

        blocks = bs_delta_D.view(self.S, self.L).contiguous().cpu()  # (S, L)
        self.bs_local_blocks[r0] = blocks

    @torch.no_grad()
    def add_round_devices(self, deltas_Ka_D: torch.Tensor, K_a: int, r1_based: int):
        """
        Select fragment positions ONCE for this round and append *per fragment* exactly K_a rows (one per device). 
        Quotas are in fragments, so we allocate a balanced number of fragments per remaining round in the split.
        """
        assert deltas_Ka_D.dim() == 2 and deltas_Ka_D.size(1) == self.D, "deltas must be (K_a, D)"
        K_a = int(K_a)
        r0 = r1_based - 1
        self.K_rounds.append(K_a)  # record K_a per round
        if self.row_dtype is None:
            self.row_dtype = deltas_Ka_D.dtype

        # reshape to (K_a, S, L)
        pad = (-self.D) % self.L
        if pad > 0:
            pad_zeros = deltas_Ka_D.new_zeros((K_a, pad))
            deltas_Ka_D = torch.cat([deltas_Ka_D, pad_zeros], dim=1)
        dev_blocks = deltas_Ka_D.view(K_a, self.S, self.L)  # (K_a, S, L)

        split = self._pick_split_for_round(r0)
        rounds_left = self._rounds_left_in_split(split, r0)
        if rounds_left <= 0 or self.frag_remaining[split] <= 0:
            return

        # balanced per-round allocation measured in fragments
        target_frags_this_round = math.ceil(self.frag_remaining[split] / rounds_left)
        k_positions = min(self.S, target_frags_this_round)
        if k_positions <= 0:
            return
    
        pos = self._select_positions(self.S, k_positions, rnd_seed=r1_based)
        if pos.numel() == 0:
            return

        rows_to_cat = []
        frag_descs = []
        running_offset = sum(t.size(0) for t in self.buf[split])  # current flat length in rows

        for p in pos.tolist():
            # (K_a, L) for this fragment index p
            frag_rows = dev_blocks[:, p, :].contiguous().cpu()
            start = running_offset
            end = start + K_a
            rows_to_cat.append(frag_rows)
            frag_descs.append({
                "round": r0,          # 0-based round id
                "pos": int(p),        # fragment index within the round
                "K_a": K_a,           # active devices this round
                "rows": [start, end]  # [start, end) in the flat X
            })
            running_offset = end

        if rows_to_cat:
            self.buf[split].append(torch.vstack(rows_to_cat))
            self.frags[split].extend(frag_descs)
            self.frag_remaining[split] -= pos.numel()

        print(f"[collect r={r1_based:03d}] split={split} took_frags={pos.numel()} "
              f"(each → K_a={K_a} rows). remaining_frags(train/valid/test)="
              f"{self.frag_remaining['train']}/{self.frag_remaining['valid']}/{self.frag_remaining['test']}")

    def finalize_and_save(self):
        """Save flat rows plus fragment descriptors and K_rounds."""
        def _cat_or_empty(lst):
            dtype = self.row_dtype or torch.float32
            return torch.vstack(lst) if lst else torch.empty((0, self.L), dtype=dtype)

        train_X = _cat_or_empty(self.buf["train"])
        valid_X = _cat_or_empty(self.buf["valid"])
        test_X = _cat_or_empty(self.buf["test"])

        for k in ("train", "valid", "test"):
            if self.frag_remaining[k] > 0:
                print(f"[warn] fragment quota for {k} underfilled by {self.frag_remaining[k]} fragments.")

        bundle = {
            "meta": self.meta,
            "D": self.D,
            "L": self.L,
            "S_per_device": self.S,
            "K_rounds": torch.tensor(self.K_rounds, dtype=torch.long),  # (R,)
            "splits": {
                "train": {"X": train_X, "frags": self.frags["train"]},
                "valid": {"X": valid_X, "frags": self.frags["valid"]},
                "test": {"X": test_X, "frags": self.frags["test"]},
            },
            "bs_local": self.bs_local_blocks
        }
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        torch.save(bundle, self.out_path)
        print(f"[dataset] Saved per-device (flat) bundle → {self.out_path}")
        return self.out_path


def split_round_ids(R, args):
    """Split round IDs into train/valid/test sets based on seed."""
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(R, generator=g).tolist()
    n_test = min(max(int(getattr(args, "num_test_rounds", 0)), 0), max(R - 1, 0))
    test_round_ids = perm[:n_test]
    remaining_rounds = [r for r in range(R) if r not in test_round_ids]
    n_valid_rounds = max(1, min(len(remaining_rounds)-1, int(0.10 * len(remaining_rounds)))) if len(remaining_rounds) > 1 else 0
    valid_round_ids = remaining_rounds[:n_valid_rounds]
    train_round_ids = remaining_rounds[n_valid_rounds:] if n_valid_rounds > 0 else remaining_rounds
    return train_round_ids, valid_round_ids, test_round_ids


def federated_training(global_model, device_loaders, test_loader, args, device, make_model):
    """
    Federated learning without quantisation.
    Collector saves per-device updates sliced to (K_a, S, L) fragments (train/valid/test)
    Also saves base station local training delta per round, reshaped to (S, L) under 'bs_local'
    """
    # 0) Start timer and display setup info
    training_start = time.time()
    max_duration = 23 * 3600

    D = sum(p.numel() for p in global_model.parameters())
    args.model_num_params = int(D)
    print(f"Global model has {D} parameters.")

    # 1) Prepare dataset collector and round splits
    R_planned = int(args.total_rounds)
    tr_ids, va_ids, te_ids = split_round_ids(R_planned, args)
    dataset_collector = TargetDatasetCollector(
        args=args, save_root=os.path.join(args.data_dir, "iFed_datasets"),
        train_round_ids=tr_ids, valid_round_ids=va_ids, test_round_ids=te_ids
    )
    device_participation_count = [0] * args.K_t

    # 2) Define transforms for train/test splits
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 3) Initialize optimizers, error feedback and history
    momentum_buffer_list = [torch.zeros_like(p.data, device=device) for p in global_model.parameters()]
    exp_avgs = [torch.zeros_like(p.data, device=device) for p in global_model.parameters()]
    exp_avg_sqs = [torch.zeros_like(p.data, device=device) for p in global_model.parameters()]

    history = {'test_acc': []}
    best_test_acc = -float("inf")
    best_global_model_state = copy.deepcopy(global_model.state_dict())  # safe fallback

    # 4) Run global rounds
    for rnd in range(1, args.total_rounds + 1):
        global_state = global_model.state_dict()
        par_before = [p.data.clone() for p in global_model.parameters()]

        # 4a) Sample participants
        K_a = np.random.randint(args.min_p, args.max_p + 1)
        participants = np.random.choice(args.K_t, K_a, replace=False)
        print(f"Round {rnd} participants: {[f'{p}({device_participation_count[p]})' for p in participants]}")

        # 4b) Local training & delta collection
        local_state_dicts = []
        for _, dev_id in tqdm(enumerate(participants), desc=f"Round {rnd} — Local training", total=len(participants), ncols=80, leave=True):
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
            cifar_train_transform = datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform_train)
            train_subset = Subset(cifar_train_transform, train_global_idxs)
            train_loader = DataLoader(train_subset, batch_size=args.batch_size_train, shuffle=True, num_workers=2, pin_memory=True)

            # 4d) Run local training
            local_model.train()
            total_samples = 0
            total_loss_sum = 0.0
            for _ in range(args.local_epochs):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = F.cross_entropy(local_model(xb), yb)
                    loss.backward()
                    opt.step()

                    batch_n = yb.size(0)
                    total_samples += batch_n
                    total_loss_sum += loss.item() * batch_n
            avg_loss = total_loss_sum / max(1, total_samples)
            if not math.isfinite(avg_loss):
                raise EarlyStopNaN(f"Device {dev_id} produced non-finite average loss.")
            print(f"Device {dev_id} sample-weighted avg loss: {avg_loss:.4f}")
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
        _ensure_finite(deltas, "deltas")

        # 4f) Save per-device updates & local BS training to dataset collector
        dataset_collector.add_bs_local(deltas[0], r1_based=rnd)
        dataset_collector.add_round_devices(deltas, K_a, r1_based=rnd)

        # 4g) Average deltas across participants
        avg_update_flat = deltas.mean(dim=0)        # (D,)
        avg_delta = []
        ptr = 0
        for p in par_before:
            n_param = p.numel()
            avg_delta.append(avg_update_flat[ptr:ptr+n_param].view(p.size()).detach())
            ptr += n_param
        assert ptr == avg_update_flat.numel(), "Mismatch in averaged update size!"

        buffer_updates = {}
        for buf_name, _ in global_model.named_buffers():
            summed = sum(local_sd[buf_name] for local_sd in local_state_dicts)
            buffer_updates[buf_name] = summed / K_a
        global_model.load_state_dict(buffer_updates, strict=False)

        # 4h) Update global model parameters via optimizer
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

        # 4i) Evaluate global model on test set
        global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = global_model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total

        # 4j) Update history
        history['test_acc'].append(acc)
        print(f"Round {rnd}/{args.total_rounds} | Test Acc: {acc:.4%}")
        if acc > best_test_acc:
            best_test_acc = acc
            best_global_model_state = copy.deepcopy(global_model.state_dict())
        elapsed = time.time() - training_start
        if elapsed >= max_duration:
            print(f"⏱️  Time limit reached, stopping at round {rnd}.")
            break

    dataset_path = dataset_collector.finalize_and_save()
    return history, best_global_model_state, dataset_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perfect Aggregation FEEL Dataset Collection")
    
    # === Training Dataset Parameters ===
    parser.add_argument('--train-size', type=int, default=400, help="number of training samples (fragments)")
    parser.add_argument('--valid-size', type=int, default=100, help="number of validation samples (fragments)")
    parser.add_argument('--test-size', type=int, default=100, help="number of test samples (fragments)")
    parser.add_argument('--num-test-rounds', type=int, default=1, help="number of rounds used in test set")
    parser.add_argument('--within-round', type=str, default="random", choices=["prefix", "contig_rand", "random"], help="Sampling strategy within each round")

    # === CIFAR Dataset Parameters ===
    parser.add_argument('--data-dir', type=str, default="runs/datasets", help="path to CIFAR-10 dataset")
    parser.add_argument('--batch-size-train', type=int, default=16, help="batch size for training")
    parser.add_argument('--batch-size-test', type=int, default=16, help="batch size for testing")
    parser.add_argument('--frac-random', type=float, default=0.2, help="fraction of random samples per device")
    
    # === Federated Learning (iFed) Parameters ===
    parser.add_argument('--K-t', type=int, default=40, help="total number of devices")
    parser.add_argument('--min-p', type=int, default=2, help="minimum participants per round")
    parser.add_argument('--max-p', type=int, default=2, help="maximum participants per round")
    parser.add_argument('--total-rounds', type=int, default=5, help="total number of global rounds")
    parser.add_argument('--local-epochs', type=int, default=3, help="number of local epochs per round")
    parser.add_argument('--local-lr', type=float, default=0.01, help="local learning rate")
    parser.add_argument('--message-split', type=int, default=10, help="message split size L (quantization dimension)")
    
    # === Global Optimizer Parameters ===
    parser.add_argument('--optimizer', type=str, default='fedavg', choices=['fedavg', 'fedavgm', 'fedadam'], help="global optimizer type")
    parser.add_argument('--global-lr', type=float, default=1.0, help="global learning rate")
    parser.add_argument('--momentum', type=float, default=0.0, help='SGD momentum')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.99, help='Adam beta2')
    parser.add_argument('--eps', type=float, default=0, help='Adam epsilon')

    # === Model Selection ===
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'cifarcnn', 'smallvgg', 'custom'], help="which global model to use for data collection")
    parser.add_argument('--custom-model', type=str, default='', help="for --model custom: 'module.path:ClassName'")
    parser.add_argument('--custom-kwargs', type=str, default='{}', help='for --model custom: JSON dict of ctor kwargs, e.g. {"num_classes":10}')

    # === System Parameters ===
    parser.add_argument('--save-dir', type=str, default="runs/results", help="directory to save results")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
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
        make_model=make_model
    )
    plot_dataset_collection_history(history, os.path.join(args.save_dir, "history_plots"))
    print(f"Dataset collection complete. Bundle saved to: {dataset_path}")
