
import argparse
import os
import gc, copy
import math

import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from src.codebooks import (
    normalize_codebook, 
    make_gaussian_codebook, 
    make_bernoulli_codebook, 
    compute_q_init,
)
from src.decoding import (
    AMPNet1DEnhanced, 
    SimAMPNetLossWithSparsity,
    ista,
    amp_da,
    greedy_rounding,
    l2_refit_on_support,
    top_k_nonneg,
)
from src.paths import (
    load_ifed_from_args, 
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
    pi_from_indices,
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
    

def split_round_ids(R, args):
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(R, generator=g).tolist()
    n_test = min(max(int(args.num_test_rounds), 0), max(R - 1, 0))
    test_round_ids = perm[:n_test]
    remaining_rounds = [r for r in range(R) if r not in test_round_ids]
    n_valid_rounds = max(1, min(len(remaining_rounds)-1, int(0.10 * len(remaining_rounds)))) if len(remaining_rounds) > 1 else 0
    valid_round_ids = remaining_rounds[:n_valid_rounds]
    train_round_ids = remaining_rounds[n_valid_rounds:] if n_valid_rounds > 0 else remaining_rounds
    return train_round_ids, valid_round_ids, test_round_ids


def build_split(
    X_all, K_rounds_all,
    wanted_samples, R, round_len, batch_size,
    shuffle_rounds=False,
    max_batches_per_round=None,
    round_pool=None,
    within_round: str = "random",   # Random is most principled, but prefix retains within round order. Options: ['prefix' | 'contig_rand' | 'random']
):
    """
    Build a split where:
      - each selected round contributes at most `max_batches_per_round` batches,
      - batches are contiguous per round (round-pure),
      - total length is batch-aligned,
      - we return BlockMeta for each contiguous per-round block.

    within_round options:
      - 'prefix'      → take the first N samples of the round.
      - 'contig_rand' → take one contiguous window at a random start position.
      - 'random'      → take N samples uniformly w/o replacement across the round (non-contiguous selection).

    Returns:
      X_split [N, n], K_vec [N], blocks_meta [List[BlockMeta]]
    """
    device = X_all.device
    dtype  = torch.float32
    g = torch.Generator(device=X_all.device).manual_seed(args.seed)  # for now use that arg is global

    cap_batches = round_len // batch_size
    if cap_batches == 0:
        raise ValueError("pt_batch_size is larger than samples-per-round; decrease pt_batch_size or increase round_len.")

    target_len     = (wanted_samples // batch_size) * batch_size
    target_batches = target_len // batch_size
    if target_batches == 0:
        raise ValueError("wanted_samples < batch_size after alignment; increase wanted_samples or reduce batch_size.")

    # --- limit the candidate rounds ---
    if round_pool is None:
        candidate_rounds = list(range(R))
    else:
        candidate_rounds = list(round_pool)
    if not candidate_rounds:
        raise ValueError("round_pool is empty; no rounds available for this split.")

    if shuffle_rounds:
        perm = torch.randperm(len(candidate_rounds)).tolist()
        candidate_rounds = [candidate_rounds[p] for p in perm]

    R_eff = len(candidate_rounds)  # effective number of rounds we draw from

    # per-round cap (in batches)
    if max_batches_per_round in (None, "auto"):
        max_b = (target_batches + R_eff - 1) // R_eff   # ceil(target_batches / R_eff)
    else:
        max_b = int(max(1, max_batches_per_round))
    max_b = min(max_b, cap_batches)

    if max_b * R_eff < target_batches:
        raise ValueError(
            f"Insufficient capacity: max_batches_per_round={max_b}, rounds={R_eff} "
            f"→ capacity {max_b*R_eff} < target_batches {target_batches}."
        )

    # accumulate
    blocks_X, blocks_K = [], []
    blocks_meta: list[BlockMeta] = []
    total_batches = 0
    local_cursor = 0

    mode = str(within_round).lower()
    if mode not in {"prefix","contig_rand","random"}:
        raise ValueError(f"within_round must be one of ['prefix','contig_rand','random'], got {within_round!r}")

    for r in candidate_rounds:
        if total_batches >= target_batches:
            break

        take_batches = min(max_b, target_batches - total_batches)
        take_samples = take_batches * batch_size
        s_round_glb  = r * round_len

        # --- choose indices inside the round ---
        if mode == "prefix":
            sel = torch.arange(
                s_round_glb, s_round_glb + take_samples, device=device, dtype=torch.long
            )
        elif mode == "contig_rand":
            max_start_b = max(0, cap_batches - take_batches)
            if max_start_b == 0:
                start_b = 0
            else:
                start_b = int(torch.randint(0, max_start_b + 1, (1,), generator=g).item())
            s_glb = s_round_glb + start_b * batch_size
            sel = torch.arange(s_glb, s_glb + take_samples, device=device, dtype=torch.long)
        else:  # mode == "random"
            offs = torch.randperm(round_len, generator=g, device=device)[:take_samples]
            sel = s_round_glb + offs.to(torch.long)

        X_blk = X_all.index_select(0, sel)
        K_blk = torch.full((take_samples,), float(K_rounds_all[r].item()), device=device, dtype=dtype)
        blocks_X.append(X_blk)
        blocks_K.append(K_blk)

        # record metadata
        local_start = local_cursor
        local_end = local_cursor + take_samples
        blocks_meta.append(BlockMeta(
            r_global=int(r),
            local_start=int(local_start),
            local_end=int(local_end),
            global_start=int(sel[0].item()),
            global_end=int(sel[-1].item()),
            round_len=int(round_len),
            sel_global=[int(x.item()) for x in sel]
        ))
        local_cursor = local_end
        total_batches += take_batches

    if total_batches == 0:
        raise ValueError("No samples taken; check wanted_samples, batch_size, round_len, and round_pool.")

    X_split = torch.cat(blocks_X, dim=0)
    K_vec = torch.cat(blocks_K, dim=0)

    # Checks stats that should always be true
    assert X_split.size(0) == total_batches * batch_size
    for b in blocks_meta:
        assert (b.local_end - b.local_start) == len(b.sel_global), "meta length mismatch vs selected indices"
        assert b.local_start % batch_size == 0 and b.local_end % batch_size == 0

    return X_split, K_vec, blocks_meta


def select_block_from_Z_full(blk: BlockMeta, Z_full: torch.Tensor, round_len: int) -> torch.Tensor:
    """Return the Z slice for a per-round block `blk` from the round-wise Z_full."""
    r = blk.r_global
    s_full = r * round_len
    if getattr(blk, "sel_global", None):
        offs = torch.tensor(blk.sel_global, device=Z_full.device, dtype=torch.long) - s_full
        return Z_full.index_select(0, offs)
    off0 = blk.global_start - s_full
    off1 = off0 + (blk.global_end - blk.global_start)
    return Z_full[off0:off1]


def save_codebook_analysis_artifacts(metrics, args, *, codebook_pt_path: str | None = None, plot_path: str | None = None, base_name: str = "codebook_analysis"):
    """
    Save analysis so plots can be fully recreated:
      - <save_dir>/summaries/codebook_analysis.json        (scalars + meta)
      - <save_dir>/summaries/codebook_analysis_arrays.npz  (arrays)
    """
    summ_dir = os.path.join(args.save_dir, "summaries")
    os.makedirs(summ_dir, exist_ok=True)

    # Arrays → NPZ (compressed)
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

    # JSON (scalars + meta + file references)
    scalars = metrics.get("_scalars", {})
    cc_percentiles = metrics.get("_cc_percentiles", {})
    cfg = metrics.get("_config", {})
    basic = metrics.get("_basic", {})

    json_payload = {
        "metric_source": "codebook_analysis",
        "version": 1,
        "dimensions": metrics.get("dimensions"),
        "basic": basic,  # {m, n}
        "config": cfg,   # {k_rip, num_trials, heatmap_vmax, hist_bins}
        "scalars": scalars,
        "cc_percentiles": cc_percentiles,
        "rip_quality": metrics.get("rip_quality", None),

        # references
        "files": {
            "arrays_npz": (None if npz_path is None else os.path.relpath(npz_path, args.save_dir)),
            "plot_path": (None if plot_path is None else os.path.relpath(plot_path, args.save_dir)),
            "codebook_pt": (None if not codebook_pt_path else os.path.relpath(codebook_pt_path, args.save_dir)),
        },
        "codebook_sha1": codebook_sha1,
        "context": run_manifest(args),
    }
    save_json(json_payload, os.path.join(summ_dir, f"{base_name}.json"))


@torch.no_grad()
def prepare_round(model: "PretrainDecoder",
                  X_all: torch.Tensor,
                  idx_all: torch.Tensor,
                  round_id: int,
                  round_len: int,
                  snr_db_signal: float,
                  *,
                  use_counts_pi: bool = True,
                  pi_target: torch.Tensor | None = None,
                  ):
    s = round_id * round_len
    e = s + round_len
    X_full   = X_all[s:e]
    idx_full = idx_all[s:e]

    # Use stream SNR for generating noisy measurements to be decoded
    Z_full = model.encode_with_noise(X_full, snr_db_signal)
    if use_counts_pi:
        pi_round = pi_from_indices(idx_full, model.ampnet.n, X_full.device)
    else:
        assert pi_target is not None, "pi_target required when use_counts_pi=False"
        pi_round = pi_target.to(X_full.device).to(torch.float32)
    model.ampnet.start_new_round(
        z_round=Z_full,
        pi_round=pi_round,
        external_URA_codebook=model.C_syn,
    )
    return {"pi_round": pi_round, "Z_full": Z_full, "idx_full": idx_full}


class PretrainDecoder(nn.Module):
    def __init__(self, args, C_init, W_init):
        super().__init__()
        assert C_init.shape == (args.n, args.dim), "C_init must be (n x dim)"
        assert W_init.shape == (args.dim, args.dim), "W_init must be (dim x dim)"

        self.C_mat = nn.Parameter(C_init.clone())
        self.W = nn.Parameter(W_init.clone())
        self.num_layers = args.num_layers

        # Respect fixed/learned URA codebook training (URA, not the quant codebook)
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
    
    def forward(self, x, snr_db=None, pi=None, use_cached_sigma=False):
        z = self.encode_with_noise(x, snr_db)
        snr_for_amp = None if use_cached_sigma else snr_db
        x_est, k_final, K_trace, sigma_trace = self.ampnet(z, external_URA_codebook=self.C_syn, pi_round=pi, snr_db=snr_for_amp)
        return z, x_est, k_final, K_trace, sigma_trace


def pre_train_decoder(train_set, train_K, train_blocks, valid_set, valid_K, valid_blocks,
                      X_all, idx_rounds_all, pi_est_rounds_all, round_len, args):
    """Pretrain AMP-DA-Net & (optionally) URA codebook on the generated dataset."""

    # Codebook initialization options (all unit-norm rows)
    init_mode = str(getattr(args, "codebook_init", "q_init")).lower()
    if init_mode == "q_init":
        snr_for_init = args.snr_db if args.snr_mode == 'fixed' else 0.5*(args.snr_min_db + args.snr_max_db)
        C_init = compute_q_init(train_set, snr_for_init, args.dim, args.device)
    elif init_mode == "gaussian":
        C_init = make_gaussian_codebook(args.n, args.dim, args.device)
    elif init_mode == "bernoulli":
        C_init = make_bernoulli_codebook(args.n, args.dim, args.device)
    else:
        raise ValueError(f"--codebook-init must be one of ['q_init','gaussian','bernoulli'], got {init_mode!r}")

    W_init = torch.eye(args.dim, device=args.device)

    # Model & loss init
    model = PretrainDecoder(args, C_init.to(args.device), W_init.to(args.device)).to(args.device)
    criterion = SimAMPNetLossWithSparsity(lambda_sparse=args.lambda_sparse, lambda_w=args.lambda_w, lambda_k=args.lambda_k).to(args.device)

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
    tol = float(getattr(args, "pt_delta", 1e-6))  # small tolerance for early stopping

    train_losses, val_losses = [], []
    snr_next_train = make_snr_sampler(args, stream="train")
    snr_next_val = make_snr_sampler(args, stream="valid")
    val_round_ids = sorted({b.r_global for b in valid_blocks})
    val_snr_by_round = {r: snr_next_val() for r in val_round_ids}

    for epoch in range(args.pt_epochs):
        model.train()
        losses = []

        for blk in tqdm(train_blocks, desc=f"Epoch {epoch+1}"):
            s_split, e_split = blk.local_start, blk.local_end
            r_global = blk.r_global

            snr_db_r = snr_next_train() 
            prep = prepare_round(
                model,
                X_all=X_all,
                idx_all=idx_rounds_all,
                round_id=r_global,
                round_len=round_len,
                snr_db_signal=snr_db_r,
                use_counts_pi=False,
                pi_target=pi_est_rounds_all[r_global]
            )
            opt_amp.zero_grad(set_to_none=True)
            for i in range(s_split, e_split, args.pt_batch_size):
                batch_x_true = train_set[i:i+args.pt_batch_size]
                batch_k_true = train_K[i:i+batch_x_true.size(0)].to(dtype=torch.float32)

                _, x_hat, k_final, _, _ = model(batch_x_true, snr_db=snr_db_r, pi=prep["pi_round"], use_cached_sigma=True)
                loss = criterion(model, batch_x_true, x_hat, batch_k_true, k_final)
                loss.backward()
                losses.append(float(loss.item()))
            opt_amp.step()  # one step per contiguous round block (keeps cache coherent)

        avg_loss = float(np.mean(losses)) if losses else 0.0
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_losses_round = []
        with torch.no_grad():
            for blk in valid_blocks:
                s_split, e_split = blk.local_start, blk.local_end
                r_global = blk.r_global

                snr_db_r = val_snr_by_round[r_global] 
                prep = prepare_round(model, X_all, idx_rounds_all, r_global, round_len,
                                     snr_db_signal=snr_db_r,
                                     use_counts_pi=False, pi_target=pi_est_rounds_all[r_global])
                for j in range(s_split, e_split, args.pt_batch_size):
                    batch_x_true = valid_set[j:j+args.pt_batch_size]
                    batch_k_true = valid_K[j:j+batch_x_true.size(0)].to(dtype=torch.float32)

                    _, x_hat, k_final, _, _ = model(batch_x_true, snr_db=snr_db_r, pi=prep["pi_round"], use_cached_sigma=True)
                    val_losses_round.append(criterion(model, batch_x_true, x_hat, batch_k_true, k_final).item())

        val_loss = float(np.mean(val_losses_round)) if val_losses_round else 0.0
        val_losses.append(val_loss)
        sched_amp.step(val_loss)
        lr_amp = sched_amp.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.pt_epochs} AMP-train={train_losses[-1]:.6f} AMP-val={val_loss:.6f} LR_AMP={lr_amp:.6f}")

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

    plot_pretraining(train_losses, val_losses, epoch+1, args.amp_lr, args.pt_batch_size, len(train_set), args.num_layers, args.save_dir)
    dec_pth, cb_pt = save_pretrained_artifacts(model, args)

    with torch.no_grad():
        amp = model.ampnet
        print("\n=== Learned AMP-DM-Net parameters ===")
        print(f"use_poisson={amp.use_poisson}  update_alpha={amp.update_alpha}  "
              f"learn_sigma2={amp.learn_sigma2}  "
              f"finetune_K={amp.finetune_K}  finetune_pi={amp.finetune_pi}  finetune_sigma2={amp.finetune_sigma2}  "
              f"per_sample_sigma={amp.per_sample_sigma}  per_sample_alpha={amp.per_sample_alpha}")

        print(f"n={amp.n}  d={amp.d}  T={amp.T}  "
              f"K_max={'adaptive (2*Ka)' if amp.fixed_Kmax is None else amp.fixed_Kmax}")

        alpha_mix = torch.sigmoid(amp.alpha_mix_raw.detach()).cpu().numpy()
        print(f"alpha_mix (prior↔posterior) per-layer summary → "
              f"mean={alpha_mix.mean():.3f}, min={alpha_mix.min():.3f}, max={alpha_mix.max():.3f}")

        scale_out = amp._scale_centered(amp.res_scale_raw, 0.3, 2.0).detach().cpu().numpy()
        inv_scale = amp._scale_centered(amp.inv_scale_raw, 0.5, 2.0).detach().cpu().numpy()
        damp      = torch.sigmoid(amp.damp_raw).detach().cpu().numpy()
        mix       = torch.sigmoid(amp.mix_raw).detach().cpu().numpy()
        tau       = torch.exp(amp.log_tau).detach().cpu().numpy()
        sK        = (2.0 * torch.sigmoid(amp.k_step_raw)).detach().cpu().numpy()
        sPi       = torch.sigmoid(amp.pi_step_raw).detach().cpu().numpy()
        sSig      = torch.sigmoid(amp.sigma_step_raw).detach().cpu().numpy()

        for t in range(amp.T):
            print(f"Layer {t+1:02d}: "
                  f"scale_out={scale_out[t]:.3f}, inv_scale={inv_scale[t]:.3f}, "
                  f"damp={damp[t]:.3f}, mix(CNN)={mix[t]:.3f}, tau={tau[t]:.3f}, "
                  f"sK={sK[t]:.3f}, sPi={sPi[t]:.3f}, sSig={sSig[t]:.3f}")

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


def evaluate_model(model, test_set, test_K, test_blocks, idx_rounds_all, pi_est_rounds_all, pi_targets_rounds_all, X_all, round_len, args):
    """Evaluate model using the same noisy signals for decoding."""
    model.eval()
    nmse_values, accuracy_values, pupe_values = [], [], []
    estimation_logs = []
    snr_next_eval = make_snr_sampler(args, stream="eval")
    eval_snr_by_round = {}

    with torch.no_grad():
        for round_idx, blk in enumerate(test_blocks):
            s_loc, e_loc = blk.local_start, blk.local_end
            s_glb, e_glb = blk.global_start, blk.global_end
            r = blk.r_global

            if r not in eval_snr_by_round:
                eval_snr_by_round[r] = snr_next_eval()

            # Round-level metadata / targets
            pi_est  = pi_est_rounds_all[r]

            # Estimation using all samples from the round
            snr_db_r = eval_snr_by_round[r]
            prep = prepare_round(
                model,
                X_all=X_all,
                idx_all=idx_rounds_all,
                round_id=r,
                round_len=round_len,
                snr_db_signal=snr_db_r,
                use_counts_pi=False,
                pi_target=pi_est
            )
            Z_full = prep["Z_full"]
            pi_for_round = prep["pi_round"]
            Z_round = select_block_from_Z_full(blk, Z_full, round_len)

            # Subset to decode, using same noisy signals as estimated parameters with
            X_round = test_set[s_loc:e_loc]             # (B_r, n)

            # Use cached (K, sigma^2), 
            k_last_vals, sig_last_vals, weights = [], [], []
            K_trace_batches, sigma_trace_batches = [], []
            round_block_len = X_round.size(0)

            for offset in range(0, round_block_len, args.pt_batch_size):
                batch_end = min(offset + args.pt_batch_size, round_block_len)
                z_b = Z_round[offset:batch_end]                  # (B, d)
                x_b = X_round[offset:batch_end]                  # (B, n)

                # Use cached per-round params (snr_db=None, K_round=None)
                x_est_b, k_final, K_trace, sigma_trace = model.ampnet(z_b, model.C_syn, pi_round=pi_for_round, snr_db=None)
                B_cur = z_b.size(0)
                k_last_vals.append(float(k_final))
                sig_last_vals.append(float(sigma_trace[-1]))
                weights.append(B_cur)

                K_trace_batches.append(torch.tensor(K_trace, dtype=torch.float32))
                sigma_trace_batches.append(torch.tensor(sigma_trace, dtype=torch.float32))
                nmse_values.extend(compute_nmse_batch(x_b.detach().cpu().numpy(), x_est_b.detach().cpu().numpy()))
                accuracy_values.extend(compute_accuracy_batch(x_b.detach().cpu().numpy(), x_est_b.detach().cpu().numpy()))
                pupe_values.extend(compute_pupe_batch(x_b.detach().cpu().numpy(), x_est_b.detach().cpu().numpy()))

            K_final_round = float(np.average(np.array(k_last_vals), weights=np.array(weights)))
            sigma2_final_round = float(np.average(np.array(sig_last_vals), weights=np.array(weights)))
            K_trace_full = torch.stack(K_trace_batches, dim=0).mean(dim=0).tolist()  # For ribbon logging only
            sigma_trace_full = torch.stack(sigma_trace_batches, dim=0).mean(dim=0).tolist()  # For ribbon logging only

            estimation_logs.append({
                "round_idx": round_idx,
                "round_size": blk.block_len,
                "K_true": float(test_K[s_loc].item()),
                "K_final": K_final_round,
                "sigma2_final": sigma2_final_round,
                "K_trace": K_trace_full,
                "sigma_trace": sigma_trace_full,
            })

    avg_nmse = float(np.mean(nmse_values)) if nmse_values else float("nan")
    avg_accuracy = float(np.mean(accuracy_values)) if accuracy_values else float("nan")
    avg_pupe = float(np.mean(pupe_values)) if pupe_values else float("nan")

    # Logging
    print("\n=== Parameter Estimation Summary ===")
    print("\n=== Parameter Summary (finals only) ===")
    for log in estimation_logs:
        print(f"Round {log['round_idx']:2d} (N={log['round_size']:4d}): "
            f"K_true={log['K_true']:5.1f}  K_final={log['K_final']:.3f}  "
            f"sigma2_final={log['sigma2_final']:.3e}")
    print(f"\nJoint AMPNet - NMSE: {avg_nmse:.6f}, Accuracy: {avg_accuracy:.6f}")
    plot_parameter_traces(estimation_logs, args.save_dir)
    return avg_nmse, avg_accuracy, avg_pupe


def compare_decoders(model, test_set, test_K, test_blocks, idx_rounds_all, pi_est_rounds_all, X_all, round_len, args, test_snr_fixed: float | None = None):
    """
    Compare methods using the same noisy signals per sample for the entire round.
    We estimate (K, sigma2) from Z_round and then decode with those cached params.
    All baselines use the same Z_round slices for fairness.
    """
    model.eval()
    results_data = {
        'amp-da-net':    {'nmse': [], 'acc': [], 'pupe': []},
        'ista':      {'nmse': [], 'acc': [], 'pupe': []},
        'amp-da-net_v1':{'nmse': [], 'acc': [], 'pupe': []},
        'amp-da':     {'nmse': [], 'acc': [], 'pupe': []},
        'amp-da-net_v2': {'nmse': [], 'acc': [], 'pupe': []},
    }

    # Codebook & sensing matrix
    C_syn = model.C_syn.detach()                  # (n, d)
    A     = C_syn.t().contiguous()                # (d, n)
    A_np  = A.detach().cpu().numpy()
    snr_next_eval = make_snr_sampler(args, stream="test")
    test_snr_by_round = {}

    with torch.no_grad():
        for round_idx, blk in enumerate(test_blocks):
            s_loc, e_loc = blk.local_start, blk.local_end
            s_glb, e_glb = blk.global_start, blk.global_end
            r = blk.r_global

            if test_snr_fixed is not None:
                snr_db_r = float(test_snr_fixed)
            else:
                if r not in test_snr_by_round:
                    test_snr_by_round[r] = snr_next_eval()
                snr_db_r = test_snr_by_round[r]

            X_block  = test_set[s_loc:e_loc]
            pi_est = pi_est_rounds_all[r]

            prep = prepare_round(
                model,
                X_all=X_all,
                idx_all=idx_rounds_all,
                round_id=r,
                round_len=round_len,
                snr_db_signal=snr_db_r,
                use_counts_pi=False,
                pi_target=pi_est
            )
            Z_full = prep["Z_full"]
            pi_est = prep["pi_round"]
            Z_round = select_block_from_Z_full(blk, Z_full, round_len)

            # Same noisy signals used, slicing converted to offset for batching
            Z_ampda_block, A_ampda = bernoulli_encode_with_noise(X_block, n=model.ampnet.n, d=model.ampnet.d, snr_db=snr_db_r, device=X_block.device, dtype=X_block.dtype)

            # Runs AMP-DA-Net over all batches, decodes, gathers final K & sigma2
            K_last_vals, K_last_wts = [], []
            Sig_last_vals, Sig_last_wts = [], []
            cache_ampnet_out, cache_z_np, cache_z_ampda, cache_x_true_np = [], [], [], []

            round_block_len = X_block.size(0)
            for offset in range(0, round_block_len, args.pt_batch_size):
                batch_end = min(offset + args.pt_batch_size, round_block_len)
                z_batch       = Z_round[offset:batch_end]
                z_batch_ampda = Z_ampda_block[offset:batch_end]
                x_true_b      = X_block[offset:batch_end]

                if z_batch.numel() == 0: # TEMP Gaurd
                    print(f"[warn] empty batch at offset={offset}, block_len={round_block_len} — skipping")
                    continue

                x_ampnet_b, k_final, _, sigma_trace = model.ampnet(z_batch, model.C_syn, pi_round=pi_est, snr_db=None)
                cache_ampnet_out.append(x_ampnet_b.detach().cpu().numpy())
                cache_z_np.append(z_batch.detach().cpu().numpy())
                cache_z_ampda.append(z_batch_ampda.detach().cpu().numpy())
                cache_x_true_np.append(x_true_b.detach().cpu().numpy())

                # final parameters results
                B_here = z_batch.size(0)
                K_last_vals.append(float(k_final))
                K_last_wts.append(B_here)
                Sig_last_vals.append(float(sigma_trace[-1]))
                Sig_last_wts.append(B_here)

            # Round-level final scalars
            K_final_round       = float(np.average(np.array(K_last_vals),  weights=np.array(K_last_wts)))
            sigma2_final_round  = float(np.average(np.array(Sig_last_vals), weights=np.array(Sig_last_wts)))
            K_int_final         = int(round(max(0.0, min(float(model.ampnet.n), K_final_round))))
            print(f"[Round {round_idx}] K_final={K_final_round:.3f} (int {K_int_final}), "
                  f"sigma2_final={sigma2_final_round:.3e}")

            # Metrics for comparison decoders, using same noisy signals
            for b in range(len(cache_ampnet_out)):
                x_ampnet_np = cache_ampnet_out[b]
                z_np        = cache_z_np[b]
                z_ampda_np  = cache_z_ampda[b]
                x_true_np   = cache_x_true_np[b]

                # AMP-DA
                x_ampda_np = None
                try:
                    y_t = torch.from_numpy(z_ampda_np).to(device=args.device, dtype=A_ampda.dtype)
                    C_t = A_ampda.to(device=args.device, dtype=y_t.dtype)
                    x_ampda_b, stats = amp_da(y_t, C_t, device=args.device)
                    x_ampda_np = x_ampda_b.detach().cpu().numpy()
                    k_est = float(stats["K_count_per_split"].mean().item())
                    k_int = int(round(k_est))
                    snr_est_db = 10.0 * math.log10((x_ampda_b @ A_ampda.T).pow(2).mean().item() / float(stats["sigma2"]))
                    print(f"[Round {round_idx}] AMP-DA: K_est={k_est:.2f}  K_int={k_int}  SNR_est={snr_est_db:.2f} dB")
                except Exception as e:
                    print(f"Warning: AMP-DA failed: {e}")

                B = x_true_np.shape[0]
                for j in range(B):
                    # 3) AMP-DA-Net Enhanced
                    results_data['amp-da-net']['nmse'].append(compute_nmse(x_true_np[j], x_ampnet_np[j]))
                    results_data['amp-da-net']['acc'].append(compute_accuracy(x_true_np[j], x_ampnet_np[j]))
                    results_data['amp-da-net']['pupe'].append(compute_pupe(x_true_np[j], x_ampnet_np[j]))

                    # 4) ISTA (baseline) with final K
                    try:
                        x_est_ista = ista(z_np[j], A_np)
                        results_data['ista']['nmse'].append(compute_nmse(x_true_np[j], x_est_ista))
                        results_data['ista']['acc'].append(compute_accuracy(x_true_np[j], x_est_ista))
                        results_data['ista']['pupe'].append(compute_pupe(x_true_np[j], x_est_ista))
                    except Exception as e:
                        print(f"Warning: ISTA failed for sample {j}: {e}")

                    # 5) AMP-DA-Net + Basic post with final K
                    try:
                        x_post_basic = np.clip(x_ampnet_np[j], a_min=0, a_max=None)
                        x_post_basic = greedy_rounding(x_post_basic, K_int_final)
                        results_data['amp-da-net_v1']['nmse'].append(compute_nmse(x_true_np[j], x_post_basic))
                        results_data['amp-da-net_v1']['acc'].append(compute_accuracy(x_true_np[j], x_post_basic))
                        results_data['amp-da-net_v1']['pupe'].append(compute_pupe(x_true_np[j], x_post_basic))
                    except Exception as e:
                        print(f"Warning: Basic post-processing failed for sample {j}: {e}")

                    # 6) AMP-DA
                    if x_ampda_np is not None:
                        results_data['amp-da']['nmse'].append(compute_nmse(x_true_np[j], x_ampda_np[j]))
                        results_data['amp-da']['acc'].append(compute_accuracy(x_true_np[j], x_ampda_np[j]))
                        results_data['amp-da']['pupe'].append(compute_pupe(x_true_np[j], x_ampda_np[j]))

                    # 7) AMP-DA-Net + advanced post with final K
                    try:
                        x_adv = top_k_nonneg(x_ampnet_np[j], K_int_final)
                        x_adv = l2_refit_on_support(z_np[j], A_np, x_adv, ridge=1e-6, nonneg=True)
                        x_adv = greedy_rounding(x_adv, K_int_final)
                        results_data['amp-da-net_v2']['nmse'].append(compute_nmse(x_true_np[j], x_adv))
                        results_data['amp-da-net_v2']['acc'].append(compute_accuracy(x_true_np[j], x_adv))
                        results_data['amp-da-net_v2']['pupe'].append(compute_pupe(x_true_np[j], x_adv))
                    except Exception as e:
                        print(f"Warning: Advanced post-processing failed for sample {j}: {e}")

    # Display results
    print("\n" + "="*75)
    print(f"{'Method':<25}{'NMSE':<20}{'Accuracy':<20}{'PUPE':<20}")
    print("-"*75)

    methods = {
        'amp-da-net': 'Joint AMP-DA-Net',
        'ista': 'ISTA',
        'amp-da-net_v1': 'AMP-DA-Net + Basic Post',  # Non-negative + greedy rounding
        'amp-da': 'AMP-DA',
        'amp-da-net_v2': 'AMP-DA-Net + Adv Post',  # Non-negative + TopK + l2 refit + greedy rounding (found to be more consistent overall)
    }
    
    # Calculate and display averages
    results_avg = {}
    for method, name in methods.items():
        data = results_data[method]
        if data['nmse']:
            nmse_avg = np.mean(data['nmse'])
            acc_avg = np.mean(data['acc'])
            pupe_avg = np.mean(data['pupe'])
            print(f"{name:<25}{nmse_avg:.6f}{'':<13}{acc_avg:.6f}{'':<13}{pupe_avg:.6f}")
            results_avg[method] = (nmse_avg, acc_avg, pupe_avg)
        else:
            results_avg[method] = (None, None, None)
            print(f"{name:<25}{'N/A':<20}{'N/A':<20}{'N/A':<20}")
    print("="*75)
    
    best_method = None
    best_acc = -1
    for method, (_, acc, _) in results_avg.items():
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_method = method
    if best_method:
        print(f"\nBest method: {methods[best_method]} with accuracy: {best_acc:.6f}")
    return results_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wireless Federated Learning with AMP-DA-Net Compression")
    
    # === Pretraining Parameters ===
    parser.add_argument('--train-size',         type=int,   default=4000, help="number of training samples")
    parser.add_argument('--valid-size',         type=int,   default=1000, help="number of validation samples")
    parser.add_argument('--test-size',          type=int,   default=1000, help="number of test samples")
    parser.add_argument('--num-test-rounds',    type=int,   default=1, help="number of rounds used in test set")
    parser.add_argument('--pt-epochs',          type=int,   default=1, help="pretraining epochs")
    parser.add_argument('--pt-batch-size',      type=int,   default=16, help="training batch size")
    parser.add_argument('--pt-patience',        type=int,   default=5, help="early-stop patience")
    parser.add_argument('--pt-delta',           type=float, default=1e-6, help="min val-loss improvement to reset patience")
    parser.add_argument('--sched-patience',     type=int,   default=5, help="scheduler patience")
    parser.add_argument('--sched-factor',       type=float, default=0.5, help="scheduler factor")

    # === Dataset Selection (slug/path) ===
    parser.add_argument('--dataset-dir',        type=str, default="runs/datasets/iFed_datasets", help="Directory containing iFed .pth bundles")
    parser.add_argument('--dataset-path',       type=str, default="", help="Exact path to a .pth dataset (overrides slug)")
    parser.add_argument('--dataset-slug',       type=str, default="", help="Optional slug override (fuzzy match allowed)")
    parser.add_argument('--code-order',         type=str, default="pop", choices=["none", "spectral", "spectral_pop", "pop"], help="Ordering used during collection (embedded in dataset)")
    parser.add_argument('--model',              type=str, default='resnet', choices=['resnet', 'cifarcnn', 'custom'], help="Model tag used during collection (affects slug)")
    parser.add_argument('--custom-model',       type=str, default='', help="For --model custom: 'module.path:ClassName'")
    parser.add_argument('--custom-kwargs',      type=str, default='{}', help='JSON dict of ctor kwargs for --model custom')
    parser.add_argument('--within-round',       type=str, default="random", choices=["prefix", "contig_rand", "random"], help="Sampling strategy within each round")

    # === Dataset Parameters ===
    parser.add_argument('--frac-random',        type=float, default=0.2, help="fraction of random samples per device")
    
    # === Federated Learning Parameters ===
    parser.add_argument('--K-t',                type=int, default=40, help="total number of devices") # Make sure is a factor of 50,000
    parser.add_argument('--min-p',              type=int, default=2, help="minimum participants per round")
    parser.add_argument('--max-p',              type=int, default=2, help="maximum participants per round")
    
    # === Wireless Channel & Compressor Parameters ===
    parser.add_argument('--message-split',      type=int, default=10, help="message split size & quantization dimension")
    parser.add_argument('--n',                  type=int, default=256, help="number of codewords in codebooks")
    parser.add_argument('--dim',                type=int, default=64, help="encoding dimension")
    parser.add_argument('--snr-db',             type=float, default=20, help="fixed SNR used when --snr-mode=fixed")
    parser.add_argument('--snr-mode',           type=str, default='range', choices=['fixed','range'], help="Use a single fixed SNR for the whole run, or sample new per-round SNRs within each epoch.")
    parser.add_argument('--snr-min-db',         type=float, default=0.0, help="Lower bound (dB) when --snr-mode=range.")
    parser.add_argument('--snr-max-db',         type=float, default=20.0, help="Upper bound (dB) when --snr-mode=range.")
    parser.add_argument('--test-snrs',          type=str, default='20,15,10,5,3', help="Comma-separated SNR(dB) list for test-time ISTA comparison, e.g. '20,15,10,5,3'. Empty = use snr-mode.")

    # === AMP-DA-Net Parameters ===
    parser.add_argument('--num-layers',         type=int, default=10, help="number of AMP-DA-Net layers")
    parser.add_argument('--num-filters',        type=int, default=32, help="number of filters in AMP-DA-Net")
    parser.add_argument('--kernel-size',        type=int, default=3, help="kernel size in AMP-DA-Net")
    parser.add_argument('--use-poisson',        action=argparse.BooleanOptionalAction, default=True, help="Use spike+Poisson counts prior (use --no-use-poisson for uniform counts)")
    parser.add_argument('--update-alpha',       action=argparse.BooleanOptionalAction, default=True, help="Enable per-layer alpha prior-posterior mixing")
    parser.add_argument('--learn-sigma2',       action=argparse.BooleanOptionalAction, default=True, help="Enable σ² EM head for noise-variance updates.")
    parser.add_argument('--finetune-K',         action=argparse.BooleanOptionalAction, default=True, help="Enable learned refinement of K per layer.")
    parser.add_argument('--finetune-pi',        action=argparse.BooleanOptionalAction, default=True, help="Enable learned refinement of π per layer.")
    parser.add_argument('--finetune-sigma2',    action=argparse.BooleanOptionalAction, default=True, help="Blend σ² via learned step size (requires --learn-sigma2).")
    parser.add_argument('--per-sample-sigma',   action=argparse.BooleanOptionalAction, default=True, help="Track per-sample σ² (B,1) instead of a scalar.")
    parser.add_argument('--per-sample-alpha',   action=argparse.BooleanOptionalAction, default=False, help="Use per-sample alpha instead of batch-averaged alpha.")
    parser.add_argument('--K-max',              type=int, default=-1, help="Fixed discrete prior cap K_max (≤ n) -1 represents adaptive (2*Ka)")
    parser.add_argument('--blend-init',         type=float, default=0.85, help="Initial CNN blend weight in (0,1) for m1-CNN")

    # === URA Codebook Options (fixed vs learned) ===
    parser.add_argument('--codebook-trainable', action=argparse.BooleanOptionalAction, default=True, help="Train the URA codebook (C,W). Use --no-codebook-trainable to freeze")
    parser.add_argument('--codebook-init',      type=str, default='gaussian', choices=['q_init', 'gaussian', 'bernoulli'], help="Initializer for URA codebook when frozen or at t=0 (unit-norm rows)")

    # === Compressor Training Parameters ===
    parser.add_argument('--amp-lr',             type=float, default=0.0001, help="compressor learning rate")
    parser.add_argument('--lambda-sparse',      type=float, default=0.001, help="sparsity regularization weight")
    parser.add_argument('--lambda-w',           type=float, default=0.001, help="W matrix regularization weight")
    parser.add_argument('--lambda-k',           type=float, default=0.01, help="K active device loss weight")

    # === Training Control & System ===
    parser.add_argument('--ckpt-dir',           type=str, default="runs/checkpoints/pretrain", help="directory to save pretrained decoder & codebook (separate from --save-dir)")
    parser.add_argument('--save-dir',           type=str, default="runs/results/pretrain", help="directory to save results")
    parser.add_argument('--seed',               type=int, default=42, help="random seed")

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
    bundle_path, X_all, K_rounds_all, pi_targets_rounds_all, pi_est_rounds_all, idx_rounds_all, R, round_len = load_ifed_from_args(args, args.device)
    train_round_ids, valid_round_ids, test_round_ids = split_round_ids(R, args)
    args.bundle_path = bundle_path

    train_X, train_K, train_blocks = build_split(
        X_all, K_rounds_all, 
        wanted_samples=args.train_size,
        R=R, 
        round_len=round_len,
        batch_size=args.pt_batch_size,
        shuffle_rounds=True,
        max_batches_per_round=None,
        round_pool=train_round_ids,
        within_round=args.within_round,
    )
    valid_X, valid_K, valid_blocks = build_split(
        X_all, K_rounds_all,
        wanted_samples=args.valid_size,
        R=R, 
        round_len=round_len,
        batch_size=args.pt_batch_size,
        shuffle_rounds=True,
        max_batches_per_round=None,
        round_pool=valid_round_ids,
        within_round=args.within_round,
    )
    test_X, test_K, test_blocks = build_split(
        X_all, K_rounds_all,
        wanted_samples=args.test_size,
        R=R,
        round_len=round_len,
        batch_size=args.pt_batch_size,
        shuffle_rounds=False,
        max_batches_per_round=None,
        round_pool=test_round_ids,
        within_round=args.within_round, # Maybe not necessary to do it in test set
    )
    print("Chosen test rounds:", test_round_ids)
    print(f"Dataset splits  train={train_X.size(0)}  valid={valid_X.size(0)}  test={test_X.size(0)}")

    # Pretraining
    decoder_model, pt_stats = pre_train_decoder(
        train_X, train_K, train_blocks, valid_X, valid_K, valid_blocks,
        X_all, idx_rounds_all, pi_est_rounds_all, round_len, args
    )

    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    print("\nEvaluating pre-trained model on test set...")
    avg_nmse, avg_accuracy, avg_pupe = evaluate_model(
        decoder_model, test_X, test_K, test_blocks,
        idx_rounds_all, pi_est_rounds_all, pi_targets_rounds_all,
        X_all, round_len, args
    )

    print("\nComparing decoders...")
    snr_list = _parse_snr_list(args.test_snrs)
    summ_dir = os.path.join(args.save_dir, "summaries")
    os.makedirs(summ_dir, exist_ok=True)

    if snr_list:
        all_results_by_snr = {}
        for snr in snr_list:
            print(f"\n[compare_decoders @ {snr:.1f} dB]")
            res = compare_decoders(
                decoder_model, test_X, test_K, test_blocks,
                idx_rounds_all, pi_est_rounds_all,
                X_all, round_len, args,
                test_snr_fixed=snr
            )
            all_results_by_snr[f"{snr:.1f}dB"] = res
        save_json(all_results_by_snr, os.path.join(summ_dir, "compare_decoders_by_snr.json"))
        pick_key = f"{snr_list[0]:.1f}dB"
        results_avg = all_results_by_snr[pick_key]
    else:
        results_avg = compare_decoders(
            decoder_model, test_X, test_K, test_blocks,
            idx_rounds_all, pi_est_rounds_all,
            X_all, round_len, args
        )

    # Persist JSON artifacts (finals + losses)
    summ_dir = os.path.join(args.save_dir, "summaries")
    os.makedirs(summ_dir, exist_ok=True)

    # 1) Minimal “finals” for logging 
    ampnet_raw       = _triplet_to_dict(results_avg, 'amp-da-net')
    ampnet_basic     = _triplet_to_dict(results_avg, 'amp-da-net_v1')
    ampnet_adv       = _triplet_to_dict(results_avg, 'amp-da-net_v2')
    ista_basic       = _triplet_to_dict(results_avg, 'ista')
    amp_da_baseline  = _triplet_to_dict(results_avg, 'amp-da')

    finals = dict(
        metric_source="pretrain",
        final_val_loss=pt_stats["best_val_loss"],

        # raw AMPNet from evaluate_model
        final_nmse=avg_nmse,
        final_acc=avg_accuracy,
        final_pupe=avg_pupe,

        # structured per-method blocks
        ampnet_raw=ampnet_raw,
        ampnet_post_basic=ampnet_basic,
        ampnet_post_adv=ampnet_adv,
        ista=ista_basic,
        ampda=amp_da_baseline,
    )
    save_json(finals, os.path.join(summ_dir, "final_metrics.json"))

    # 2) Loss curves for later re-plots & JSONL info for later aggregation across experiments
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
    analysis_fig_path = os.path.join(args.save_dir, "codebook_analysis", "codebook_analysis.png")
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