
import os
import json
import hashlib
import warnings
import torch
from .dataset_handling import _fmt_float, _norm_order_tag
from ..models.factory import _model_tag_from_args

def make_core_cb_tag(args):
    return f"cb({int(args.n)}_{int(args.dim)}_{int(args.message_split)})"

def make_decoder_slug(args):
    order_tag = _norm_order_tag(args.code_order)
    model_tag = _model_tag_from_args(args)
    core = f"ifed_{model_tag}_{order_tag}_{make_core_cb_tag(args)}"
    arch = f"T{int(args.num_layers)}_nf{int(args.num_filters)}_ks{int(args.kernel_size)}"
    opts = (
        f"poiss{int(bool(args.use_poisson))}_ua{int(bool(args.update_alpha))}"
        f"_ls2{int(bool(args.learn_sigma2))}_pssig{int(bool(args.per_sample_sigma))}"
        f"_psa{int(bool(args.per_sample_alpha))}"
    )
    if str(getattr(args,'snr_mode','fixed')).lower() == 'range':
        snr = f"snrU[{_fmt_float(args.snr_min_db,0)}-{_fmt_float(args.snr_max_db,0)}]dB"
    else:
        snr = f"snr{_fmt_float(float(args.snr_db),1)}dB"
    split = f"split-{str(getattr(args, 'within_round', 'prefix')).lower()}"  # NEW
    cbinit  = f"cbinit-{str(args.codebook_init).lower()}"
    cbtrain = f"cbtrain-{int(bool(args.codebook_trainable))}"
    return f"{core}_{arch}_{opts}_{snr}_{split}_{cbinit}_{cbtrain}_seed{int(args.seed)}"  # NEW: split in slug

def _sha1_tensor(t: torch.Tensor) -> str:
    b = t.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha1(b).hexdigest()

def save_pretrained_artifacts(model, args):
    """
    Save decoder and its synthesis codebook as a matched pair using the decoder slug as run ID:
      - <slug>.pth        (decoder: state_dict + meta)
      - <slug>.json       (decoder meta for quick inspection)
      - <slug>_cb.pt      (codebook tensor)
      - <slug>_cb.json    (codebook sidecar)
    """
    ckpt_dir = os.path.join(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    dec_slug = make_decoder_slug(args)          # keep your existing verbose decoder slug
    base     = os.path.join(ckpt_dir, dec_slug)

    # Codebook tensor and fingerprint
    codebook = model.C_syn.detach().to("cpu")   # (n, d), row-unit-norm
    cb_sha1  = _sha1_tensor(codebook)

    # Decoder payload (meta references codebook fingerprint)
    amp_state = {
        "state_dict": model.ampnet.state_dict(),
        "meta": {
            # Hard-compat keys
            "n": int(model.ampnet.n),
            "dim": int(model.ampnet.d),
            "message_split": int(args.message_split),
            "T": int(model.ampnet.T),
            # Your existing architectural/context tags
            "num_filters": int(args.num_filters),
            "kernel_size": int(args.kernel_size),
            "snr_db_train": float(args.snr_db) if args.snr_mode == 'fixed' else float(0.5*(args.snr_min_db + args.snr_max_db)),
            "snr_mode": str(getattr(args,"snr_mode","fixed")).lower(),
            "snr_min_db": float(getattr(args,"snr_min_db", 0.0)),
            "snr_max_db": float(getattr(args,"snr_max_db", 20.0)),
            "dataset_basename": os.path.basename(getattr(args, "bundle_path", "")),
            "saved_with_seed": int(args.seed),
            "within_round": str(getattr(args, "within_round", "prefix")).lower(),  # NEW
            # AMPNet toggles (kept out of filename, but preserved here)
            "use_poisson": bool(model.ampnet.use_poisson),
            "update_alpha": bool(model.ampnet.update_alpha),
            "learn_sigma2": bool(model.ampnet.learn_sigma2),
            "finetune_K": bool(model.ampnet.finetune_K),
            "finetune_pi": bool(model.ampnet.finetune_pi),
            "finetune_sigma2": bool(model.ampnet.finetune_sigma2),
            "per_sample_sigma": bool(model.ampnet.per_sample_sigma),
            "per_sample_alpha": bool(model.ampnet.per_sample_alpha),
            # K_max encoding (your CLI convention: -1 → adaptive)
            "K_max_mode": (int(args.K_max) if int(getattr(args, "K_max", -1)) >= 0 else "2*Ka"),
            # Codebook info
            "codebook_init": str(args.codebook_init).lower(),
            "codebook_trainable": bool(args.codebook_trainable),
            "codebook_sha1": cb_sha1,  # codebook fingerprint
        }
    }

    # Paths
    decoder_pth  = base + ".pth"
    decoder_json = base + ".json"
    codebook_pt  = base + "_cb.pt"
    codebook_json= base + "_cb.json"

    # Save files
    torch.save(amp_state, decoder_pth)
    with open(decoder_json, "w") as f:
        json.dump(amp_state["meta"], f, indent=2)
    torch.save(codebook, codebook_pt)
    with open(codebook_json, "w") as f:
        json.dump({
            "shape": list(codebook.shape),
            "row_unit_norm": True,
            "dataset_basename": os.path.basename(getattr(args, "bundle_path", "")),
            "saved_with_seed": int(args.seed),
            "codebook_sha1": cb_sha1,
            "paired_decoder": os.path.basename(decoder_pth),
            "codebook_init": str(args.codebook_init).lower(),
            "codebook_trainable": bool(args.codebook_trainable),
        }, f, indent=2)

    print(f"[save] Decoder  → {decoder_pth}")
    print(f"[save] Codebook → {codebook_pt}")
    return decoder_pth, codebook_pt

def load_paired_decoder(decoder_path: str, device="cpu"):
    """
    Load decoder and its paired codebook sitting next to it:
      - expects <base>.pth and <base>_cb.pt
      - validates (n,d) shape using decoder meta
      - validates codebook SHA1 if present
    """
    base, ext = os.path.splitext(decoder_path)
    if ext.lower() != ".pth":
        raise ValueError("decoder_path must be a .pth file")

    amp_state = torch.load(decoder_path, map_location=device)
    meta = amp_state.get("meta", {})
    n = int(meta.get("n", -1)); d = int(meta.get("dim", -1))

    cb_path = base + "_cb.pt"
    if not os.path.isfile(cb_path):
        raise FileNotFoundError(f"Paired codebook not found. Expected: {cb_path}")

    codebook = torch.load(cb_path, map_location=device)
    if not torch.is_tensor(codebook) or codebook.dim() != 2:
        raise ValueError(f"Codebook at {cb_path} is not a 2D tensor")

    if (n > 0 and d > 0) and codebook.shape != (n, d):
        raise ValueError(f"Decoder/codebook shape mismatch: codebook {tuple(codebook.shape)} vs decoder meta (n={n}, d={d})")

    # SHA1 check if available
    expected = meta.get("codebook_sha1", None)
    if expected is not None:
        actual = _sha1_tensor(codebook)
        if expected != actual:
            raise ValueError(f"Codebook SHA1 mismatch: expected {expected}, got {actual}")

    return amp_state, codebook, cb_path

def find_decoder_artifacts(args):
    """
    Returns (decoder_pth, codebook_pt) using precedence:
      explicit paths → slug → error.
    """
    # explicit
    if args.decoder_path and args.codebook_path:
        return args.decoder_path, args.codebook_path

    # slug
    slug = args.decoder_slug or make_decoder_slug(args)
    base = os.path.join(args.ckpt_dir, slug)
    dec = base + ".pth"
    cb  = base + "_cb.pt"
    if os.path.isfile(dec) and os.path.isfile(cb):
        return dec, cb

    raise FileNotFoundError(
        f"Could not find decoder pair. Looked for:\n  {dec}\n  {cb}\n"
        "Provide --decoder-path and --codebook-path, or correct --ckpt-dir/slug & args."
    )

def load_pretrained_pair(args, device):
    """
    Loads the decoder state and matching URA codebook. Verifies basic compatibility.
    Returns: (amp_state_dict, amp_meta, C_nt)
    """
    dec_pth, cb_pt = find_decoder_artifacts(args)
    amp_payload = torch.load(dec_pth, map_location=device)
    C_nt = torch.load(cb_pt, map_location=device)
    meta = amp_payload.get("meta", {})

    # Sanity checks
    need = {"n": int(args.n), "dim": int(args.dim), "message_split": int(args.message_split)}
    got = {k: int(meta.get(k, -1)) for k in need}
    mm = {k: (got[k], need[k]) for k in need if got[k] != need[k]}
    if mm:
        msg = "; ".join([f"{k}: ckpt={a} vs args={b}" for k,(a,b) in mm.items()])
        raise ValueError(f"Decoder/codebook meta mismatch: {msg}")

    # Optional checks — informative but not fatal
    def _warn_if(name, val):
        if name in meta and meta[name] != val:
            warnings.warn(f"Checkpoint meta {name}={meta[name]} differs from args {name}={val}")

    _warn_if("within_round", str(getattr(args,"within_round","")).lower())
    _warn_if("snr_mode", str(getattr(args,"snr_mode","")).lower())
    if str(getattr(args,"snr_mode","")).lower() == "range":
        _warn_if("snr_min_db", float(getattr(args,"snr_min_db", float('nan'))))
        _warn_if("snr_max_db", float(getattr(args,"snr_max_db", float('nan'))))

    # Check codebook info is consistent
    meta_init = str(meta.get("codebook_init", "")).lower()
    if meta_init and meta_init != str(args.codebook_init).lower():
        raise ValueError(f"Codebook init mismatch: ckpt='{meta_init}' vs args='{args.codebook_init}'")

    meta_train = meta.get("codebook_trainable", None)
    if (meta_train is not None) and (bool(meta_train) != bool(args.codebook_trainable)):
        raise ValueError(f"Codebook trainable mismatch: ckpt='{bool(meta_train)}' vs args='{bool(args.codebook_trainable)}'")

    # Optional fingerprint check
    want_sha = meta.get("codebook_sha1", None)
    got_sha  = hashlib.sha1(C_nt.detach().to('cpu', dtype=torch.float32).contiguous().numpy().tobytes()).hexdigest()
    if want_sha and want_sha != got_sha:
        warnings.warn(f"Codebook fingerprint mismatch: meta={want_sha[:8]}.. vs file={got_sha[:8]}..")

    return amp_payload["state_dict"], meta, C_nt.to(device).to(torch.float32), dec_pth, cb_pt