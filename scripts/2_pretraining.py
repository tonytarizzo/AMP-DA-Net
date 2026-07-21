"""Pretraining script for URA codebook and AMP-DA-Net."""
import argparse
import os
import gc
import copy
import math
import time

import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.codebooks import (
    normalize_codebook, 
    make_gaussian_codebook, 
    make_bernoulli_codebook,
    create_quantisation_codebook,
    get_round_perm,
    compute_probability_prior,
    apply_ordering,
)
from src.decoding import (
    AMPNet1DEnhanced, 
    ista,
    amp_da,
    greedy_rounding,
    l2_refit_on_support,
    top_k_nonneg,
)
from src.paths import (
    load_raw_ifed_bundle,
    save_pretrained_artifacts,
    _sha1_tensor,
)
from src.utils import (
    compute_nmse,
    compute_nmse_batch,
    compute_accuracy,
    compute_accuracy_batch,
    compute_pupe,
    compute_pupe_batch,
    add_awgn_noise,
    bernoulli_encode_with_noise,
    make_snr_sampler,
    _parse_snr_list,
)
from src.analysis import (
    plot_pretraining,
    plot_parameter_traces,
    analyse_codebook,
    print_codebook_analysis,
    save_json,
    append_jsonl,
    run_manifest,
    _triplet_to_dict,
)


@dataclass
class BlockMeta:
    """Metadata for a fragment block in the split."""
    r_global: int
    local_start: int
    local_end: int
    global_start: int
    global_end: int
    round_len: int
    sel_global: list[int] | None = None

    @property
    def block_len(self) -> int:
        return self.local_end - self.local_start


def split_to_tensors(split_pack, K_rounds_vec: torch.Tensor, *, device) -> tuple[torch.Tensor, torch.Tensor, list[BlockMeta]]:
    """
    Convert a split pack from the raw bundle into tensors and metadata.
    
    split_pack: {"X": FloatTensor [N_rows, L], "frags": List[dict(round,pos,K_a,rows=[s,e))]}
    K_rounds_vec: LongTensor [R] (per-round K_a)
    
    Returns:
      X_split [N_rows, L] on device
      K_vec   [N_rows]   (per-row K_a copied from its fragment's round)
      blocks  list[BlockMeta] (one block per fragment, contiguous in local coords)
    """
    X = split_pack["X"].to(device=device, dtype=torch.float32)  # [N_rows, L]
    frags = split_pack["frags"]

    # Build K per-row and blocks
    K_rows = torch.empty((X.size(0),), device=device, dtype=torch.float32)
    blocks = []
    for f in frags:
        r0 = int(f["round"])
        s_row, e_row = int(f["rows"][0]), int(f["rows"][1])   # in X's local indexing
        K_rows[s_row:e_row] = float(K_rounds_vec[r0].item())  # Fill K for rows in this fragment
        blocks.append(BlockMeta(
            r_global=r0,
            local_start=s_row,
            local_end=e_row,
            global_start=s_row,
            global_end=e_row,
            round_len=-1,
            sel_global=list(range(s_row, e_row))
        ))
    return X, K_rows, blocks


def load_raw_splits(args, device):
    """Load the new raw iFed bundle and materialize per-split tensors."""
    bundle_path, bundle = load_raw_ifed_bundle(args, device=device)
    args.bundle_path = bundle_path
    collection_complete = bundle.get("meta", {}).get("collection_complete")
    if collection_complete is False and not args.allow_incomplete_dataset:
        remaining = bundle.get("meta", {}).get("fragment_remaining", {})
        raise ValueError(
            f"Dataset collection is incomplete (remaining fragments: {remaining}). "
            "Resume/re-run collection, or pass --allow-incomplete-dataset for a non-paper smoke run."
        )

    K_rounds = bundle["K_rounds"].to(device=device)  # [R]
    splits = bundle["splits"]                        # {"train": {...}, "valid": {...}, "test": {...}}

    # Load tensors and fragments
    train_X, train_K, train_frags = split_to_tensors(splits["train"], K_rounds, device=device)
    valid_X, valid_K, valid_frags = split_to_tensors(splits["valid"], K_rounds, device=device)
    test_X, test_K, test_frags = split_to_tensors(splits["test"], K_rounds, device=device)

    print(f"[dataset] split sizes  train={train_X.size(0)}  valid={valid_X.size(0)}  test={test_X.size(0)}")
    print(f"[dataset] fragment counts  train={len(train_frags)}  valid={len(valid_frags)}  test={len(test_frags)}")
    return (train_X, train_K, train_frags,
            valid_X, valid_K, valid_frags,
            test_X, test_K, test_frags,
            bundle)


def quantize(x_raw, Q):
    """Quantize x_raw to nearest codeword in Q."""
    if x_raw.device != Q.device or x_raw.dtype != Q.dtype:
        x = x_raw.to(Q.device, dtype=Q.dtype)
    else:
        x = x_raw
    dists = torch.cdist(x, Q, p=2)           # [B,n]
    idx = torch.argmin(dists, dim=1)         # [B]
    x_q = Q.index_select(0, idx)             # [B,L]
    return idx, (x_q - x)


def aggregate_fragment_to_sparse_vector(device_rows, Q_r):
    """Aggregate exactly K_a device rows into one sparse vector."""
    K_a = device_rows.size(0)
    
    # Quantize all device rows
    idx_all, q_err_all = quantize(device_rows, Q_r)  # [K_a], [K_a, L]
    
    # Convert to one-hots and sum
    one_hots = F.one_hot(idx_all, num_classes=Q_r.size(0)).float()  # [K_a, n]
    sparse_vec = one_hots.sum(dim=0)  # [n]
    q_err_avg = q_err_all.mean(dim=0)  # [L]
    
    return sparse_vec, q_err_avg


@torch.no_grad()
def prepare_round(model, Q_r, bs_blocks_r, pi_round):
    """Initialize AMPNet for a new round using base station blocks to form a clean signal."""
    device = model.C_syn.device
    if bs_blocks_r.device != device or bs_blocks_r.dtype != torch.float32:
        Xbs = bs_blocks_r.to(device=device, dtype=torch.float32)
    else:
        Xbs = bs_blocks_r
    if Q_r.device != device:
        Q_gpu = Q_r.to(device)
    else:
        Q_gpu = Q_r
    idx_bs = torch.cdist(Xbs, Q_gpu, p=2).argmin(dim=1)                 # [S]
    x_onehot_bs = F.one_hot(idx_bs, num_classes=Q_r.size(0)).to(dtype=torch.float32)  # [S,n]
    z_round = x_onehot_bs @ model.C_syn                              # [S,d] - clean encoded signal
    model.ampnet.start_new_round(z_round=z_round, pi_round=pi_round, external_URA_codebook=model.C_syn)


@torch.no_grad()
def get_round_Q_pi(r: int, cache: dict, *, bundle, args):
    """Get or create per-round quantization codebook and pi prior."""
    if r not in cache:
        bs = bundle["bs_local"][r]                                       # (S,L)
        Q_raw = create_quantisation_codebook(bs, args)                   # (n,L)
        pi_raw = compute_probability_prior(Q_raw, bs, device=args.device) # (n,)
        Q_ord, pi_ord, perm, inv = apply_ordering(Q_raw, pi_raw, args.code_order, bs)
        cache[r] = {"Q": Q_ord, "pi": pi_ord, "perm": perm, "inv": inv}
    return cache[r]["Q"], cache[r]["pi"]


class PretrainDecoder(nn.Module):
    def __init__(self, args, C_init, W_init):
        super().__init__()
        assert C_init.shape == (args.n, args.dim), "C_init must be (n x dim)"
        assert W_init.shape == (args.dim, args.dim), "W_init must be (dim x dim)"

        self.C_mat = nn.Parameter(C_init.clone())
        self.W = nn.Parameter(W_init.clone())
        self.num_layers = args.num_layers

        # Respect fixed/learned URA codebook training
        self.codebook_trainable = bool(getattr(args, "codebook_trainable", True))
        if not self.codebook_trainable:
            self.C_mat.requires_grad_(False)
            self.W.requires_grad_(False)

        # Compose initial synthesis codebook (unit-norm rows)
        with torch.no_grad():
            raw_C = self.C_mat @ self.W
            init_C = normalize_codebook(raw_C)

        self.ampnet = AMPNet1DEnhanced(
            init_C, self.num_layers,
            num_filters=args.num_filters,
            kernel_size=args.kernel_size,
            use_poisson=args.use_poisson,
            update_alpha=args.update_alpha,
            learn_sigma2=args.learn_sigma2,
            finetune_K=args.finetune_K,
            finetune_pi=args.finetune_pi,
            finetune_sigma2=args.finetune_sigma2,
            K_max=(None if args.K_max < 0 else args.K_max),
            blend_init=args.blend_init,
            per_sample_sigma=args.per_sample_sigma,
            per_sample_alpha=args.per_sample_alpha
        ).to(args.device)

    @property
    def C_syn(self):
        return normalize_codebook(self.C_mat @ self.W)
    
    def encode_with_noise(self, x, snr_db):
        """Encode with the current synthesis codebook and add AWGN at snr_db (dB)."""
        z = x @ self.C_syn
        if snr_db is not None:
            z = add_awgn_noise(z, snr_db)
        return z
    
    def forward(self, x, snr_db=None, pi=None):
        z = self.encode_with_noise(x, snr_db)
        x_est, k_final, K_trace, sigma_trace = self.ampnet(z, external_URA_codebook=self.C_syn, pi_round=pi)
        return z, x_est, k_final, K_trace, sigma_trace


class SimAMPNetLossWithSparsityQ(nn.Module):
    """Loss function with sparsity, W orthogonality, K active device, and quantization penalties."""
    def __init__(self, lambda_sparse=0.001, lambda_w=0.001, lambda_k=0.01, lambda_q=0.0):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_w = lambda_w
        self.lambda_k = lambda_k
        self.lambda_q = lambda_q

    def forward(self, model, x_gt, x_pred, K_a, K_final, q_err):
        # MSE reconstruction loss
        rec_loss = F.mse_loss(x_pred, x_gt, reduction='mean')
        
        # L1 sparsity regularization
        target_scale = torch.mean(torch.abs(x_gt)) + 1e-10
        l1_norm = torch.mean(torch.abs(x_pred))
        sparsity = l1_norm / target_scale

        # W orthogonality regularization
        I = torch.eye(model.W.shape[0], device=model.W.device, dtype=model.W.dtype)
        w_reg = torch.norm(model.W.conj().T @ model.W - I, p='fro') ** 2

        # K active device loss
        K_pred = torch.as_tensor(K_final, device=K_a.device, dtype=K_a.dtype)
        K_pred = K_pred.reshape(1).expand_as(K_a)
        k_loss = F.mse_loss(K_pred, K_a.detach(), reduction='mean')

        # Normalised Q error loss
        q_loss = ((q_err.pow(2)).sum(dim=1) / (x_gt.pow(2).sum(dim=1).clamp_min(1e-8))).mean()
        
        return rec_loss + self.lambda_sparse * sparsity + self.lambda_w * w_reg + self.lambda_k * k_loss + self.lambda_q * q_loss


def pre_train_decoder(train_set, train_K, train_blocks, valid_set, valid_K, valid_blocks, bundle, args):
    """Pretrain AMP-Net & (optionally) URA codebook on the generated dataset."""
    
    # Codebook initialization options (all unit-norm rows)
    init_mode = str(getattr(args, "codebook_init", "gaussian")).lower()
    if init_mode == "q_init":
        raise NotImplementedError(
            "q_init is not implemented for raw per-device bundles. Refusing to "
            "label a Gaussian fallback as data-driven initialisation. Use "
            "--codebook-init gaussian/bernoulli, or materialise the training "
            "count vectors before computing q_init."
        )
    elif init_mode == "gaussian":
        C_init = make_gaussian_codebook(args.n, args.dim, args.device)
    elif init_mode == "bernoulli":
        C_init = make_bernoulli_codebook(args.n, args.dim, args.device)
    else:
        raise ValueError(f"--codebook-init must be one of ['q_init','gaussian','bernoulli'], got {init_mode!r}")

    W_init = torch.eye(args.dim, device=args.device)

    # Model & loss init
    model = PretrainDecoder(args, C_init.to(args.device), W_init.to(args.device)).to(args.device)
    criterion = SimAMPNetLossWithSparsityQ(
        lambda_sparse=args.lambda_sparse, 
        lambda_w=args.lambda_w, 
        lambda_k=args.lambda_k,
        lambda_q=args.lambda_q
    ).to(args.device)

    # Optimiser & scheduling init
    trainable_param_groups = []
    cb_params = [p for p in (model.C_mat, model.W) if p.requires_grad]
    if cb_params:
        trainable_param_groups.append({"params": cb_params})
    amp_params = [p for p in model.ampnet.parameters() if p.requires_grad]
    if amp_params:
        trainable_param_groups.append({"params": amp_params})
    if not trainable_param_groups:
        raise RuntimeError("No trainable parameters found (both URA codebook and AMPNet are frozen).")
    opt_amp = optim.Adam(trainable_param_groups, lr=args.amp_lr)
    sched_amp = optim.lr_scheduler.ReduceLROnPlateau(opt_amp, mode='min', patience=args.sched_patience, factor=args.sched_factor)

    best_val_loss = float('inf')
    best_train_loss = None
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0
    tol = float(getattr(args, "pt_delta", 1e-6))

    train_losses, val_losses = [], []
    snr_next_train = make_snr_sampler(args, stream="train")
    snr_next_val = make_snr_sampler(args, stream="valid")
    val_round_ids = sorted({b.r_global for b in valid_blocks})
    val_snr_by_round = {r: snr_next_val() for r in val_round_ids}

    start_time = time.time()
    for epoch in range(args.pt_epochs):
        model.train()
        losses = []
        q_cache = {}
        
        # Batch accumulators
        sparse_batch = []
        q_err_batch = []
        K_batch = []
        current_round_id = None
        current_Q_r = None
        current_pi_round = None

        for frag in tqdm(train_blocks, desc=f"Epoch {epoch+1}"):
            r = frag.r_global
            start, end = frag.local_start, frag.local_end
            K_a = int(train_K[start].item())  # All rows in fragment have same K_a

            # Per-round quantisation, popularity prior, model cache update
            if r not in q_cache:
                Q_r, pi_r = get_round_Q_pi(r, q_cache, bundle=bundle, args=args)
            
            Q_r = q_cache[r]["Q"]
            pi_round = q_cache[r]["pi"]

            # If round changed, process any accumulated batch first
            if current_round_id is not None and current_round_id != r and len(sparse_batch) > 0:
                batch_x = torch.stack(sparse_batch)  # [B, n]
                batch_q_err = torch.stack(q_err_batch)  # [B, L]
                batch_K = torch.tensor(K_batch, device=args.device, dtype=torch.float32)  # [B]
                
                opt_amp.zero_grad(set_to_none=True)
                snr_db = snr_next_train()
                _, x_hat, k_final, _, _ = model(batch_x, snr_db=snr_db, pi=current_pi_round)
                loss = criterion(model, batch_x, x_hat, batch_K, k_final, batch_q_err)
                loss.backward()
                opt_amp.step()
                losses.append(float(loss.detach().item()))
                
                # Reset batch
                sparse_batch, q_err_batch, K_batch = [], [], []

            # Update round tracking and prepare model
            if current_round_id != r:
                current_round_id = r
                current_Q_r = Q_r
                current_pi_round = pi_round
                prepare_round(model, Q_r, bundle["bs_local"][r], pi_round)
            
            # Aggregate this fragment (K_a rows) → 1 sparse vector
            device_rows = train_set[start:end]  # [K_a, L]
            sparse_vec, q_err = aggregate_fragment_to_sparse_vector(device_rows, Q_r)  # [n], [L]
            
            # Collect sparse vectors into batches
            sparse_batch.append(sparse_vec)
            q_err_batch.append(q_err)
            K_batch.append(float(K_a))
            
            # When batch is full, train
            if len(sparse_batch) >= args.pt_batch_size:
                batch_x = torch.stack(sparse_batch)  # [B, n]
                batch_q_err = torch.stack(q_err_batch)  # [B, L]
                batch_K = torch.tensor(K_batch, device=args.device, dtype=torch.float32)  # [B]
                
                opt_amp.zero_grad(set_to_none=True)
                snr_db = snr_next_train()
                _, x_hat, k_final, _, _ = model(batch_x, snr_db=snr_db, pi=current_pi_round)
                loss = criterion(model, batch_x, x_hat, batch_K, k_final, batch_q_err)
                loss.backward()
                opt_amp.step()
                losses.append(float(loss.detach().item()))
                
                # Reset batch
                sparse_batch, q_err_batch, K_batch = [], [], []
        
        # Handle final remainder batch at end of epoch
        if len(sparse_batch) > 0:
            batch_x = torch.stack(sparse_batch)  # [B, n]
            batch_q_err = torch.stack(q_err_batch)  # [B, L]
            batch_K = torch.tensor(K_batch, device=args.device, dtype=torch.float32)  # [B]
            
            opt_amp.zero_grad(set_to_none=True)
            snr_db = snr_next_train()
            _, x_hat, k_final, _, _ = model(batch_x, snr_db=snr_db, pi=current_pi_round)
            loss = criterion(model, batch_x, x_hat, batch_K, k_final, batch_q_err)
            loss.backward()
            opt_amp.step()
            losses.append(float(loss.detach().item()))

        avg_loss = float(np.mean(losses)) if losses else 0.0
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_losses_round = []
        with torch.no_grad():
            val_sparse_batch = []
            val_q_err_batch = []
            val_K_batch = []
            val_current_round_id = None
            val_current_pi_round = None
            val_current_snr_db = None
            
            for frag in valid_blocks:
                r = frag.r_global
                start, end = frag.local_start, frag.local_end
                K_a = int(valid_K[start].item())

                if r not in q_cache:
                    Q_r, pi_r = get_round_Q_pi(r, q_cache, bundle=bundle, args=args)
                
                Q_r = q_cache[r]["Q"]
                pi_round = q_cache[r]["pi"]
                snr_db_r = val_snr_by_round[r]

                # If round changed, process any accumulated batch first
                if val_current_round_id is not None and val_current_round_id != r and len(val_sparse_batch) > 0:
                    batch_x = torch.stack(val_sparse_batch)
                    batch_q_err = torch.stack(val_q_err_batch)
                    batch_K = torch.tensor(val_K_batch, device=args.device, dtype=torch.float32)
                    
                    _, x_hat, k_final, _, _ = model(batch_x, snr_db=val_current_snr_db, pi=val_current_pi_round)
                    val_loss = criterion(model, batch_x, x_hat, batch_K, k_final, batch_q_err)
                    val_losses_round.append(float(val_loss.detach().item()))
                    
                    val_sparse_batch, val_q_err_batch, val_K_batch = [], [], []

                # Update round tracking and prepare model
                if val_current_round_id != r:
                    val_current_round_id = r
                    val_current_pi_round = pi_round
                    val_current_snr_db = snr_db_r
                    prepare_round(model, Q_r, bundle["bs_local"][r], pi_round)

                # Aggregate this fragment
                device_rows = valid_set[start:end]
                sparse_vec, q_err = aggregate_fragment_to_sparse_vector(device_rows, Q_r)
                
                val_sparse_batch.append(sparse_vec)
                val_q_err_batch.append(q_err)
                val_K_batch.append(float(K_a))
                
                # When batch is full, evaluate
                if len(val_sparse_batch) >= args.pt_batch_size:
                    batch_x = torch.stack(val_sparse_batch)
                    batch_q_err = torch.stack(val_q_err_batch)
                    batch_K = torch.tensor(val_K_batch, device=args.device, dtype=torch.float32)
                    
                    _, x_hat, k_final, _, _ = model(batch_x, snr_db=val_current_snr_db, pi=val_current_pi_round)
                    val_loss = criterion(model, batch_x, x_hat, batch_K, k_final, batch_q_err)
                    val_losses_round.append(float(val_loss.detach().item()))
                    
                    val_sparse_batch, val_q_err_batch, val_K_batch = [], [], []
            
            # Handle final remainder batch
            if len(val_sparse_batch) > 0:
                batch_x = torch.stack(val_sparse_batch)
                batch_q_err = torch.stack(val_q_err_batch)
                batch_K = torch.tensor(val_K_batch, device=args.device, dtype=torch.float32)
                
                _, x_hat, k_final, _, _ = model(batch_x, snr_db=val_current_snr_db, pi=val_current_pi_round)
                val_loss = criterion(model, batch_x, x_hat, batch_K, k_final, batch_q_err)
                val_losses_round.append(float(val_loss.detach().item()))

        val_loss = float(np.mean(val_losses_round)) if val_losses_round else 0.0
        val_losses.append(val_loss)
        sched_amp.step(val_loss)
        lr_amp = sched_amp.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.pt_epochs} AMP-train={train_losses[-1]:.6f} AMP-val={val_loss:.6f} LR_AMP={lr_amp:.6f}")

        # Runtime check: stop before the enclosing scheduler allocation expires.
        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours >= args.max_hours:
            print(f"[early stop] Time limit reached after {elapsed_hours:.2f} hours, proceeding to evaluation")
            break

        # Early stopping with tolerance
        if val_loss < (best_val_loss - tol):
            best_val_loss = val_loss
            best_train_loss = avg_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.pt_patience:
                print(f"Early stopping triggered. Best Epoch: {best_epoch}, Train Loss: {best_train_loss:.6f}, Validation Loss: {best_val_loss:.6f}")
                break

    if best_model_state:
        print(f"Loading best model from epoch {best_epoch}/{epoch+1}")
        model.load_state_dict(best_model_state)
        print(f"Best Train Loss: {best_train_loss:.6f}, Best Validation Loss: {best_val_loss:.6f}")

    plot_pretraining(train_losses, val_losses, epoch+1, args.amp_lr, args.pt_batch_size, len(train_blocks), args.num_layers, args.save_dir)
    dec_pth, cb_pt = save_pretrained_artifacts(model, args)

    with torch.no_grad():
        amp = model.ampnet
        print("\n=== Learned AMP-DA-Net parameters ===")
        print(f"use_poisson={amp.use_poisson}  update_alpha={amp.update_alpha}  "
              f"learn_sigma2={amp.learn_sigma2}  "
              f"finetune_K={amp.finetune_K}  finetune_pi={amp.finetune_pi}  finetune_sigma2={amp.finetune_sigma2}  "
              f"per_sample_sigma={amp.per_sample_sigma}  per_sample_alpha={amp.per_sample_alpha}")

        print(f"n={amp.n}  d={amp.d}  T={amp.T}  "
              f"K_max={'adaptive (2*Ka)' if amp.fixed_Kmax is None else amp.fixed_Kmax}")

        I = torch.eye(model.W.shape[0], device=model.W.device)
        W_dev = torch.norm(model.W.mH @ model.W - I, p='fro').item()
        ortho_err_rel = W_dev / torch.norm(I, p='fro').item()
        print(f"W relative orthogonality error: {ortho_err_rel:.4e}")

    pt_stats = {
        "best_epoch": best_epoch,
        "best_train_loss": float(best_train_loss) if best_train_loss is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "decoder_pth": dec_pth,
        "codebook_pt": cb_pt,
    }
    return model, pt_stats


def evaluate_model(model, test_set, test_K, test_blocks, bundle, args):
    """Evaluate model with per-round parameter estimation using the same noisy signals for decoding."""
    model.eval()
    nmse_values, accuracy_values, pupe_values = [], [], []
    estimation_logs = []

    snr_next_eval = make_snr_sampler(args, stream="eval")
    snr_by_round = {}
    q_cache = {}

    with torch.no_grad():
        # Track fragments by round for per-round reporting
        round_fragments = {}
        for frag in test_blocks:
            r = frag.r_global
            if r not in round_fragments:
                round_fragments[r] = []
            round_fragments[r].append(frag)
        
        # Process each round
        for round_idx, r in enumerate(sorted(round_fragments.keys())):
            frags_in_round = round_fragments[r]
            
            # 1) Per-round SNR + (Q_r, pi_r) + AMP init
            if r not in snr_by_round:
                snr_by_round[r] = snr_next_eval()

            snr_db_r = snr_by_round[r]
            Q_r, pi_r = get_round_Q_pi(r, q_cache, bundle=bundle, args=args)
            prepare_round(model, Q_r, bundle["bs_local"][r], pi_r)

            # 2) Aggregate all fragments in this round to sparse vectors
            sparse_vecs_round = []
            K_sparse_round = []
            
            for frag in frags_in_round:
                start, end = frag.local_start, frag.local_end
                K_a = int(test_K[start].item())
                
                device_rows = test_set[start:end]  # [K_a, L]
                sparse_vec, _ = aggregate_fragment_to_sparse_vector(device_rows, Q_r)  # [n]
                
                sparse_vecs_round.append(sparse_vec)
                K_sparse_round.append(float(K_a))
            
            if len(sparse_vecs_round) == 0:
                continue
            
            sparse_vecs = torch.stack(sparse_vecs_round)  # [N_sparse, n]
            K_sparse = torch.tensor(K_sparse_round, device=args.device, dtype=torch.float32)  # [N_sparse]
            
            # 3) Decode batches of sparse vectors
            K_last, Sig_last, W = [], [], []
            K_tr_all, Sig_tr_all = [], []

            for i in range(0, sparse_vecs.size(0), args.pt_batch_size):
                x_true_sparse_b = sparse_vecs[i:i+args.pt_batch_size]         # (B,n)
                K_true_b = K_sparse[i:i+x_true_sparse_b.size(0)]              # (B,)

                z_b = model.encode_with_noise(x_true_sparse_b, snr_db=snr_db_r)  # (B,d)
                x_hat_b, K_final, K_trace, Sigma_trace = model.ampnet(z_b, external_URA_codebook=model.C_syn, pi_round=pi_r, snr_db=None)

                # Metrics vs sparse vector targets (n-dim)
                nmse_values.extend(compute_nmse_batch(x_true_sparse_b.cpu().numpy(), x_hat_b.cpu().numpy()))
                accuracy_values.extend(compute_accuracy_batch(x_true_sparse_b.cpu().numpy(), x_hat_b.cpu().numpy()))
                pupe_values.extend(compute_pupe_batch(x_true_sparse_b.cpu().numpy(), x_hat_b.cpu().numpy()))

                # Traces for logging
                B_here = z_b.size(0)
                K_last.append(float(K_final))  
                W.append(B_here)
                Sig_last.append(float(Sigma_trace[-1]))
                K_tr_all.append(torch.tensor(K_trace, dtype=torch.float32))
                Sig_tr_all.append(torch.tensor(Sigma_trace, dtype=torch.float32))

            # Round summaries
            K_final_round = float(np.average(np.array(K_last), weights=np.array(W))) if W else float('nan')
            Sig_final_round = float(np.average(np.array(Sig_last), weights=np.array(W))) if W else float('nan')
            K_trace_mean = torch.stack(K_tr_all, dim=0).mean(dim=0).tolist() if K_tr_all else []
            Sig_trace_mean = torch.stack(Sig_tr_all, dim=0).mean(dim=0).tolist() if Sig_tr_all else []

            estimation_logs.append({
                "round_idx": round_idx,
                "round_size": sparse_vecs.size(0),
                "K_true": float(K_sparse[0].item()) if K_sparse.numel() > 0 else float('nan'),
                "K_final": K_final_round,
                "sigma2_final": Sig_final_round,
                "K_trace": K_trace_mean,
                "sigma_trace": Sig_trace_mean,
            })

    avg_nmse = float(np.mean(nmse_values)) if nmse_values else float("nan")
    avg_acc = float(np.mean(accuracy_values)) if accuracy_values else float("nan")
    avg_pupe = float(np.mean(pupe_values)) if pupe_values else float("nan")

    print("\n=== Parameter Summary (finals only) ===")
    for log in estimation_logs:
        print(f"Round {log['round_idx']:2d} (N={log['round_size']:4d}): "
              f"K_true={log['K_true']:5.1f}  K_final={log['K_final']:.3f}  "
              f"sigma2_final={log['sigma2_final']:.3e}")
    print(f"\nJoint AMPNet - NMSE: {avg_nmse:.6f}, Accuracy: {avg_acc:.6f}")

    plot_parameter_traces(estimation_logs, args.save_dir)
    return avg_nmse, avg_acc, avg_pupe


def compare_decoders(model, test_set, test_K, test_blocks, bundle, args, test_snr_fixed: float | None = None):
    """Compare methods using the same noisy signals per sample for the entire round."""
    model.eval()
    results = {
        'ampnet':    {'nmse': [], 'acc': [], 'pupe': []},
        'ista':      {'nmse': [], 'acc': [], 'pupe': []},
        'post_basic':{'nmse': [], 'acc': [], 'pupe': []},
        'ampdm':     {'nmse': [], 'acc': [], 'pupe': []},
        'ampnet_v2': {'nmse': [], 'acc': [], 'pupe': []},
    }

    C_syn = model.C_syn.detach()                 # (n,d)
    A = C_syn.t().contiguous()                   # (d,n) for ISTA
    A_np = A.cpu().numpy()

    snr_next = make_snr_sampler(args, stream="test")
    snr_by_round = {}
    q_cache = {}

    with torch.no_grad():
        round_fragments = {}
        for frag in test_blocks:
            r = frag.r_global
            if r not in round_fragments:
                round_fragments[r] = []
            round_fragments[r].append(frag)
        
        for ridx, r in enumerate(sorted(round_fragments.keys())):
            frags_in_round = round_fragments[r]

            snr_db_r = float(test_snr_fixed) if test_snr_fixed is not None else snr_by_round.setdefault(r, snr_next())
            Q_r, pi_r = get_round_Q_pi(r, q_cache, bundle=bundle, args=args)
            prepare_round(model, Q_r, bundle["bs_local"][r], pi_r)

            sparse_vecs_round = []
            K_sparse_round = []
            
            for frag in frags_in_round:
                start, end = frag.local_start, frag.local_end
                K_a = int(test_K[start].item())
                
                device_rows = test_set[start:end]
                sparse_vec, _ = aggregate_fragment_to_sparse_vector(device_rows, Q_r)
                
                sparse_vecs_round.append(sparse_vec)
                K_sparse_round.append(float(K_a))
            
            if len(sparse_vecs_round) == 0:
                continue
            
            sparse_vecs = torch.stack(sparse_vecs_round)
            K_sparse = torch.tensor(K_sparse_round, device=args.device, dtype=torch.float32)
            
            # Batches
            K_last, W, Sig_last = [], [], []
            cache_ampnet, cache_z_cs, cache_x_true = [], [], []

            for i in range(0, sparse_vecs.size(0), args.pt_batch_size):
                x_true_sparse_b = sparse_vecs[i:i+args.pt_batch_size]

                z_b = model.encode_with_noise(x_true_sparse_b, snr_db=snr_db_r)

                x_ampnet_b, k_final, _, sigma_trace = model.ampnet(
                    z_b, external_URA_codebook=model.C_syn, pi_round=pi_r, snr_db=None
                )
                cache_ampnet.append(x_ampnet_b.cpu().numpy())
                cache_z_cs.append(z_b.cpu().numpy())
                cache_x_true.append(x_true_sparse_b.cpu().numpy())

                B_here = z_b.size(0)
                K_last.append(float(k_final)); W.append(B_here)
                Sig_last.append(float(sigma_trace[-1]))

            if W:
                K_final_round = float(np.average(np.array(K_last), weights=np.array(W)))
                Sig_final_round = float(np.average(np.array(Sig_last), weights=np.array(W)))
                K_int_final = int(round(max(0.0, min(float(model.ampnet.n), K_final_round))))
            else:
                K_final_round = Sig_final_round = float('nan'); K_int_final = 0
            print(f"[Round {ridx}] K_final={K_final_round:.3f} (int {K_int_final}), sigma2_final={Sig_final_round:.3e}")

            for b in range(len(cache_ampnet)):
                x_ampnet_np = cache_ampnet[b]
                z_np = cache_z_cs[b]
                x_true_np = cache_x_true[b]

                B = x_true_np.shape[0]
                for j in range(B):
                    # 1) AMPNet (joint)
                    results['ampnet']['nmse'].append(compute_nmse(x_true_np[j], x_ampnet_np[j]))
                    results['ampnet']['acc'].append(compute_accuracy(x_true_np[j], x_ampnet_np[j]))
                    results['ampnet']['pupe'].append(compute_pupe(x_true_np[j], x_ampnet_np[j]))

                    # 2) ISTA on the same z, same A
                    try:
                        x_est_ista = ista(z_np[j], A_np)
                        results['ista']['nmse'].append(compute_nmse(x_true_np[j], x_est_ista))
                        results['ista']['acc'].append(compute_accuracy(x_true_np[j], x_est_ista))
                        results['ista']['pupe'].append(compute_pupe(x_true_np[j], x_est_ista))
                    except Exception as e:
                        print(f"Warning: ISTA failed for sample {j}: {e}")

                    # 3) AMPNet + basic post (clip, sum-K rounding)
                    try:
                        x_post = np.clip(x_ampnet_np[j], a_min=0, a_max=None)
                        x_post = greedy_rounding(x_post, K_int_final)
                        results['post_basic']['nmse'].append(compute_nmse(x_true_np[j], x_post))
                        results['post_basic']['acc'].append(compute_accuracy(x_true_np[j], x_post))
                        results['post_basic']['pupe'].append(compute_pupe(x_true_np[j], x_post))
                    except Exception as e:
                        print(f"Warning: Basic post failed for sample {j}: {e}")

                    # 4) AMPNet + advanced post
                    try:
                        x_adv = top_k_nonneg(x_ampnet_np[j], K_int_final)
                        x_adv = l2_refit_on_support(z_np[j], A_np, x_adv, ridge=1e-6, nonneg=True)
                        x_adv = greedy_rounding(x_adv, K_int_final)
                        results['ampnet_v2']['nmse'].append(compute_nmse(x_true_np[j], x_adv))
                        results['ampnet_v2']['acc'].append(compute_accuracy(x_true_np[j], x_adv))
                        results['ampnet_v2']['pupe'].append(compute_pupe(x_true_np[j], x_adv))
                    except Exception as e:
                        print(f"Warning: Advanced post failed for sample {j}: {e}")

            # 5) AMP-DA baseline once per round. This used to be repeated for
            # every AMP-DA-Net mini-batch, duplicating identical work/metrics.
            try:
                Z_ampdm, A_ampdm = bernoulli_encode_with_noise(
                    sparse_vecs.to(args.device), n=model.ampnet.n, d=model.ampnet.d, snr_db=snr_db_r,
                    device=args.device, dtype=sparse_vecs.dtype
                )
                x_ampdm_b, stats = amp_da(Z_ampdm, A_ampdm, device=args.device)
                xa = x_ampdm_b.cpu().numpy(); xt = sparse_vecs.cpu().numpy()
                for j in range(xa.shape[0]):
                    results['ampdm']['nmse'].append(compute_nmse(xt[j], xa[j]))
                    results['ampdm']['acc'].append(compute_accuracy(xt[j], xa[j]))
                    results['ampdm']['pupe'].append(compute_pupe(xt[j], xa[j]))
            except Exception as e:
                print(f"Warning: AMP-DA baseline failed on round {ridx}: {e}")

    print("\n" + "="*75)
    print(f"{'Method':<25}{'NMSE':<20}{'Accuracy':<20}{'PUPE':<20}")
    print("-"*75)
    results_avg = {}
    names = {
        'ampnet': 'Joint AMPNet',
        'ista': 'Basic ISTA',
        'post_basic': 'AMPNet + Basic Post',
        'ampdm': 'AMP-DA (Bernoulli A)',
        'ampnet_v2': 'AMPNet + Adv Post',
    }
    for key, label in names.items():
        data = results[key]
        if data['nmse']:
            nmse = np.mean(data['nmse']); acc = np.mean(data['acc']); pupe = np.mean(data['pupe'])
            print(f"{label:<25}{nmse:.6f}{'':<13}{acc:.6f}{'':<13}{pupe:.6f}")
            results_avg[key] = (nmse, acc, pupe)
        else:
            print(f"{label:<25}{'N/A':<20}{'N/A':<20}{'N/A':<20}")
            results_avg[key] = (None, None, None)
    print("="*75)
    
    best_key, best_acc = None, -1
    for k, (_, acc, _) in results_avg.items():
        if acc is not None and acc > best_acc:
            best_key, best_acc = k, acc
    if best_key:
        print(f"\nBest method: {names[best_key]} with accuracy: {best_acc:.6f}")
    return results_avg


def save_codebook_analysis_artifacts(metrics, args, *, codebook_pt_path: str | None = None, plot_path: str | None = None, base_name: str = "codebook_analysis"):
    """Save analysis so plots can be fully recreated."""
    summ_dir = os.path.join(args.save_dir, "summaries")
    os.makedirs(summ_dir, exist_ok=True)

    arrays = metrics.get("_arrays", {})
    arrays_to_save = {}
    for key in ["G", "S", "rip_deltas", "G_hist_counts", "G_hist_edges"]:
        if key in arrays and arrays[key] is not None:
            arrays_to_save[key] = np.asarray(arrays[key])

    npz_path = os.path.join(summ_dir, f"{base_name}_arrays.npz")
    if arrays_to_save:
        np.savez_compressed(npz_path, **arrays_to_save)
    else:
        npz_path = None

    codebook_sha1 = None
    try:
        if "decoder_model" in globals():
            codebook_sha1 = _sha1_tensor(decoder_model.C_syn.data.clone())
    except Exception:
        pass

    scalars = metrics.get("_scalars", {})
    cc_percentiles = metrics.get("_cc_percentiles", {})
    cfg = metrics.get("_config", {})
    basic = metrics.get("_basic", {})

    json_payload = {
        "metric_source": "codebook_analysis",
        "version": 1,
        "dimensions": metrics.get("dimensions"),
        "basic": basic,
        "config": cfg,
        "scalars": scalars,
        "cc_percentiles": cc_percentiles,
        "rip_quality": metrics.get("rip_quality", None),
        "files": {
            "arrays_npz": (None if npz_path is None else os.path.relpath(npz_path, args.save_dir)),
            "plot_path": (None if plot_path is None else os.path.relpath(plot_path, args.save_dir)),
            "codebook_pt": (None if not codebook_pt_path else os.path.relpath(codebook_pt_path, args.save_dir)),
        },
        "codebook_sha1": codebook_sha1,
        "context": run_manifest(args),
    }
    save_json(json_payload, os.path.join(summ_dir, f"{base_name}.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wireless Federated Learning with AMPNet Compression")
    
    # === Pretraining Parameters ===
    parser.add_argument('--pt-epochs', type=int, default=2, help="pretraining epochs")
    parser.add_argument('--pt-batch-size', type=int, default=16, help="training batch size")
    parser.add_argument('--pt-patience', type=int, default=5, help="early-stop patience")
    parser.add_argument('--pt-delta', type=float, default=1e-6, help="min val-loss improvement to reset patience")
    parser.add_argument('--sched-patience', type=int, default=5, help="scheduler patience")
    parser.add_argument('--sched-factor', type=float, default=0.5, help="scheduler factor")

    # === Dataset Selection (slug/path) ===
    parser.add_argument('--dataset-dir', type=str, default="runs/datasets/iFed_datasets", help="Directory containing iFed .pth bundles")
    parser.add_argument('--dataset-path', type=str, default="", help="Exact path to a .pth dataset (overrides slug)")
    parser.add_argument('--dataset-slug', type=str, default="", help="Optional slug override (fuzzy match allowed)")
    parser.add_argument('--allow-incomplete-dataset', action=argparse.BooleanOptionalAction, default=False, help="allow a time-limited, underfilled bundle (not recommended for paper runs)")
    parser.add_argument('--code-order', type=str, default="pop", choices=["none", "spectral", "spectral_pop", "pop"], help="Codebook ordering strategy")
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'cifarcnn', 'smallvgg', 'custom'], help="Model tag used during collection (affects slug)")
    parser.add_argument('--custom-model', type=str, default='', help="For --model custom: 'module.path:ClassName'")
    parser.add_argument('--custom-kwargs', type=str, default='{}', help='JSON dict of ctor kwargs for --model custom')
    parser.add_argument('--within-round', type=str, default="random", choices=["prefix", "contig_rand", "random"], help="Sampling strategy within each round")

    # === Federated Learning Parameters ===
    parser.add_argument('--K-t', type=int, default=40, help="total number of devices")
    parser.add_argument('--min-p', type=int, default=2, help="minimum participants per round")
    parser.add_argument('--max-p', type=int, default=2, help="maximum participants per round")
    
    # === Wireless Channel & Compressor Parameters ===
    parser.add_argument('--message-split', type=int, default=10, help="message split size & quantization dimension")
    parser.add_argument('--n', type=int, default=256, help="number of codewords in codebooks")
    parser.add_argument('--dim', type=int, default=64, help="encoding dimension")
    parser.add_argument('--snr-db', type=float, default=20, help="fixed SNR used when --snr-mode=fixed")
    parser.add_argument('--snr-mode', type=str, default='range', choices=['fixed', 'range'], help="Use a single fixed SNR for the whole run, or sample new per-round SNRs within each epoch.")
    parser.add_argument('--snr-min-db', type=float, default=0.0, help="Lower bound (dB) when --snr-mode=range.")
    parser.add_argument('--snr-max-db', type=float, default=20.0, help="Upper bound (dB) when --snr-mode=range.")
    parser.add_argument('--test-snrs', type=str, default='20,15,10,5,3,0', help="Comma-separated SNR(dB) list for test-time comparison")

    # === AMPNet Parameters ===
    parser.add_argument('--num-layers', type=int, default=10, help="number of AMPNet layers")
    parser.add_argument('--num-filters', type=int, default=32, help="number of filters in AMPNet")
    parser.add_argument('--kernel-size', type=int, default=3, help="kernel size in AMPNet")
    parser.add_argument('--use-poisson', action=argparse.BooleanOptionalAction, default=True, help="Use spike+Poisson counts prior")
    parser.add_argument('--update-alpha', action=argparse.BooleanOptionalAction, default=True, help="Enable per-layer alpha prior-posterior mixing")
    parser.add_argument('--learn-sigma2', action=argparse.BooleanOptionalAction, default=True, help="Enable σ² EM head for noise-variance updates.")
    parser.add_argument('--finetune-K', action=argparse.BooleanOptionalAction, default=True, help="Enable learned refinement of K per layer.")
    parser.add_argument('--finetune-pi', action=argparse.BooleanOptionalAction, default=True, help="Enable learned refinement of π per layer.")
    parser.add_argument('--finetune-sigma2', action=argparse.BooleanOptionalAction, default=True, help="Blend σ² via learned step size.")
    parser.add_argument('--per-sample-sigma', action=argparse.BooleanOptionalAction, default=True, help="Track per-sample σ² (B,1) instead of a scalar.")
    parser.add_argument('--per-sample-alpha', action=argparse.BooleanOptionalAction, default=False, help="Use per-sample alpha instead of batch-averaged alpha.")
    parser.add_argument('--K-max', type=int, default=-1, help="Fixed discrete prior cap K_max (≤ n) -1 represents adaptive (2*Ka)")
    parser.add_argument('--blend-init', type=float, default=0.85, help="Initial CNN blend weight in (0,1) for m1-CNN")

    # === URA & Quant Codebook Options ===
    parser.add_argument('--codebook-trainable', action=argparse.BooleanOptionalAction, default=True, help="Train the URA codebook (C,W)")
    parser.add_argument('--codebook-init', type=str, default='gaussian', choices=['q_init', 'gaussian', 'bernoulli'], help="Initializer for URA codebook")
    parser.add_argument('--quant-type', type=str, default='kmeans_plusplus', choices=['kmeans_plusplus'], help='Quantizer type for per-round fragment quantization')
    parser.add_argument('--kmeans-max-iter', type=int, default=0, help="Max Lloyd iterations for K-means (0 = just k-means++ seeds)")
    parser.add_argument('--curv-mode', type=str, default='none', choices=['none', 'diagonal_hessian_proxy', 'data_whitening'], help="Curvature-aware codebook creation mode")
    parser.add_argument('--curv-max-samples', type=int, default=10000, help="Max samples for curvature estimation")
    parser.add_argument('--curv-eps', type=float, default=1e-6, help="Regularization epsilon for curvature methods")

    # === Compressor Training Parameters ===
    parser.add_argument('--amp-lr', type=float, default=0.0001, help="compressor learning rate")
    parser.add_argument('--lambda-sparse', type=float, default=0.001, help="sparsity regularization weight")
    parser.add_argument('--lambda-w', type=float, default=0.001, help="W matrix regularization weight")
    parser.add_argument('--lambda-k', type=float, default=0.01, help="K active device loss weight")
    parser.add_argument('--lambda-q', type=float, default=0.0, help='Quantization penalty weight')

    # === Training Control & System ===
    parser.add_argument('--ckpt-dir', type=str, default="runs/checkpoints/pretrain", help="directory to save pretrained decoder & codebook")
    parser.add_argument('--save-dir', type=str, default="runs/results/pretrain", help="directory to save results")
    parser.add_argument('--max-hours', type=float, default=17.0, help="stop cleanly after this wall-clock duration")
    parser.add_argument('--seed', type=int, default=42, help="random seed")

    args = parser.parse_args()

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        args.device = torch.device('cpu')
        print("GPU not available, using CPU")

    # Load dataset
    (train_X, train_K, train_blocks, 
     valid_X, valid_K, valid_blocks,
     test_X, test_K, test_blocks, bundle) = load_raw_splits(args, args.device)

    # Pretraining
    decoder_model, pt_stats = pre_train_decoder(
        train_X, train_K, train_blocks,
        valid_X, valid_K, valid_blocks,
        bundle, args
    )

    print("\nEvaluating pre-trained model on test set...")
    avg_nmse, avg_accuracy, avg_pupe = evaluate_model(
        decoder_model, test_X, test_K, test_blocks, bundle, args
    )

    print("\nComparing decoders...")
    snr_list = _parse_snr_list(args.test_snrs)
    summ_dir = os.path.join(args.save_dir, "summaries")
    os.makedirs(summ_dir, exist_ok=True)

    if snr_list:
        all_results_by_snr = {}
        for snr in snr_list:
            print(f"\n[compare_decoders @ {snr:.1f} dB]")
            res = compare_decoders(decoder_model, test_X, test_K, test_blocks, bundle, args, test_snr_fixed=snr)
            all_results_by_snr[f"{snr:.1f}dB"] = res
        save_json(all_results_by_snr, os.path.join(summ_dir, "compare_decoders_by_snr.json"))
        results_avg = all_results_by_snr[f"{snr_list[0]:.1f}dB"]
    else:
        results_avg = compare_decoders(decoder_model, test_X, test_K, test_blocks, bundle, args)

    # Persist JSON artifacts
    ampnet_raw = _triplet_to_dict(results_avg, 'ampnet')
    ampnet_basic = _triplet_to_dict(results_avg, 'post_basic')
    ampnet_adv = _triplet_to_dict(results_avg, 'ampnet_v2')
    ista_basic = _triplet_to_dict(results_avg, 'ista')
    ampdm_baseline = _triplet_to_dict(results_avg, 'ampdm')

    finals = dict(
        metric_source="pretrain_eval",
        final_val_loss=pt_stats["best_val_loss"],
        final_nmse=avg_nmse,
        final_acc=avg_accuracy,
        final_pupe=avg_pupe,
        ampnet_raw=ampnet_raw,
        ampnet_post_basic=ampnet_basic,
        ampnet_post_adv=ampnet_adv,
        ista=ista_basic,
        ampdm=ampdm_baseline,
        snr_mode=str(getattr(args, "snr_mode", "fixed")).lower(),
        snr_db=getattr(args, "snr_db", None),
        snr_range=[getattr(args, "snr_min_db", None), getattr(args, "snr_max_db", None)],
        code_order=str(getattr(args, "code_order", "")),
        within_round=str(getattr(args, "within_round", "prefix")).lower(),
        decoder_pth=pt_stats.get("decoder_pth"),
        codebook_pt=pt_stats.get("codebook_pt"),
    )
    save_json(finals, os.path.join(summ_dir, "final_metrics.json"))
    save_json(dict(train=pt_stats["train_losses"], valid=pt_stats["val_losses"]), os.path.join(summ_dir, "loss_curves.json"))
    append_jsonl({**run_manifest(args), **finals}, os.path.join(summ_dir, "runs.jsonl"))

    # Cleanup
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(decoder_model, os.path.join(args.save_dir, "final_compressor.pth"))
    print(f"Training completed. Results saved to {args.save_dir}")

    # Analyse final models & codebooks
    analysis_dir = os.path.join(args.save_dir, "codebook_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    analysis_fig_path = os.path.join(analysis_dir, "codebook_analysis.png")
    
    URA_codebook_metrics = analyse_codebook(
        decoder_model.C_syn.data.clone().T,
        plot=True,
        k_rip=args.max_p,
        save_path=analysis_fig_path,
        show_plots=False
    )
    print_codebook_analysis(URA_codebook_metrics, print_full=True)
    save_codebook_analysis_artifacts(
        URA_codebook_metrics,
        args,
        codebook_pt_path=pt_stats.get("codebook_pt", None),
        plot_path=analysis_fig_path,
        base_name="codebook_analysis"
    )
