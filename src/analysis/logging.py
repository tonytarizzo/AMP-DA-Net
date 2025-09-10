
import os
import json
import csv
import socket
import platform
import datetime
import numpy as np
import torch

def _to_py(x):
    if isinstance(x, (float, int, str, bool)) or x is None: return x
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if torch.is_tensor(x): 
        return x.detach().cpu().tolist() if x.dim() else x.detach().cpu().item()
    if isinstance(x, (list, tuple)): return [_to_py(v) for v in x]
    if isinstance(x, dict): return {k: _to_py(v) for k, v in x.items()}
    return str(x)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_py(obj), f, indent=2)
    print(f"[save] {path}")

def append_jsonl(record, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_to_py(record)) + "\n")
    print(f"[append] {path}")

def save_round_history_csv(history_dict, path):
    """history_dict keys are series of equal length; writes one row per round."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(history_dict.keys())
    rows = len(history_dict[keys[0]]) if keys else 0
    for k in keys:
        assert len(history_dict[k]) == rows, f"history[{k}] length mismatch"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round"] + keys)
        for r in range(rows):
            w.writerow([r+1] + [_to_py(history_dict[k][r]) for k in keys])
    print(f"[save] {path}")

def run_manifest(args, extra=None):
    """Small metadata bundle to join results across runs."""
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    sysinfo = dict(host=socket.gethostname(), os=platform.platform())
    base = dict(
        timestamp=stamp,
        seed=int(getattr(args, "seed", -1)),
        n=int(getattr(args,"n",-1)), dim=int(getattr(args,"dim",-1)),
        message_split=int(getattr(args,"message_split",-1)),
        snr_db=float(getattr(args,"snr_db", float("nan"))),
        snr_mode=str(getattr(args,"snr_mode","fixed")).lower(),
        snr_min_db=float(getattr(args,"snr_min_db", float('nan'))),
        snr_max_db=float(getattr(args,"snr_max_db", float('nan'))),
        code_order=str(getattr(args,"code_order","")),
        model=str(getattr(args,"model","")),
        codebook_init=str(getattr(args,"codebook_init","")),
        codebook_trainable=bool(getattr(args,"codebook_trainable", True)),
        within_round=str(getattr(args,"within_round","prefix")).lower(),  # NEW
        K_t=int(getattr(args,"K_t",-1)),
        min_p=int(getattr(args,"min_p",-1)),
        max_p=int(getattr(args,"max_p",-1)),
        ckpt_dir=str(getattr(args,"ckpt_dir","")),
        save_dir=str(getattr(args,"save_dir","")),
        job=os.environ.get("PBS_JOBNAME") or os.environ.get("SLURM_JOB_NAME"),
        sys=sysinfo,
    )
    if extra: base.update(extra)
    return base

def _triplet_to_dict(results_avg, key):
        nmse, acc, pupe = results_avg.get(key, (None, None, None))
        return {"nmse": (None if nmse is None else float(nmse)),
                "acc":  (None if acc  is None else float(acc)),
                "pupe": (None if pupe is None else float(pupe))}
