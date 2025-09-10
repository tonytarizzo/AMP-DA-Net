
import os
import torch
from ..models.factory import _model_tag_from_args


def _fmt_float(f: float, ndigits=2):
    return f"{f:.{ndigits}f}".rstrip('0').rstrip('.') if isinstance(f, float) else str(f)

def _norm_order_tag(order: str | None):
    if order is None or str(order).lower() in {"none", "null", "na"}:
        return "none"
    if order == "spectral_pop":
        return "spectral+pop"
    return str(order)

def make_dataset_slug(*, model: str, order: str, n: int, dim: int, msg: int, Kt: int,
                      pmin: int, pmax: int, frac_random: float, seed: int):
    # e.g. ifed_resnet_pop_cb(256_64_10)_Kt40_p(2-8)_fr0.20_seed42
    return (
        f"ifed_{model}_{order}_cb({n}_{dim}_{msg})"
        f"_Kt{Kt}_p({pmin}-{pmax})_fr{_fmt_float(frac_random,2)}_seed{seed}"
    )

def load_ifed_from_args(args, device):
    """
    Resolve the iFed dataset path using the slug logic and load tensors to `device`.
    Requires the slug helpers you already added:
      - _fmt_float, _norm_order_tag, _model_tag_from_args, make_dataset_slug,
        auto_find_dataset, load_ifed_bundle

    Returns:
      (bundle_path, X_all, K_rounds_all, pi_targets_rounds_all, pi_est_rounds_all, idx_rounds_all, R, round_len)
    """
    bundle_path, bundle = load_ifed_bundle(args, device=device)

    X_all = bundle['X'].to(device).to(torch.float32)                                # (R*S, n)
    K_rounds_all = bundle['K_rounds'].to(device).to(torch.float32).view(-1)         # (R,)
    pi_targets_rounds_all = bundle['pi_targets'].to(device).to(torch.float32)       # (R, n)
    pi_est_rounds_all = bundle['pi_estimates'].to(device).to(torch.float32)         # (R, n)
    idx_rounds_all = bundle['device_idx'].to(device=device, dtype=torch.long)       # (R*S,)
    # Prefer round_len from bundle if present; otherwise infer
    round_len = int(bundle.get('round_len', 0)) or (X_all.size(0) // K_rounds_all.numel())
    R = int(K_rounds_all.numel())

    assert X_all.size(0) == R * round_len, "Dataset shape mismatch: (R*S) != X_all.size(0)"
    return (bundle_path, X_all, K_rounds_all, pi_targets_rounds_all, pi_est_rounds_all, idx_rounds_all, R, round_len)


def load_ifed_bundle(args, device):
    path = auto_find_dataset(args)
    bundle = torch.load(path, map_location=device)
    meta = bundle.get('meta', {})

    # Hard compatibility checks
    req = {
        'n': int(args.n),
        'dim': int(args.dim),
        'message_split': int(args.message_split),
    }
    got = {k: int(meta.get(k, -1)) for k in req}
    mismatches = {k: (got[k], req[k]) for k in req if got[k] != req[k]}
    if mismatches:
        msg = "\n".join([f"  {k}: dataset={v[0]} vs args={v[1]}" for k, v in mismatches.items()])
        raise ValueError(
            "Dataset meta does not match your arguments.\n" + msg +
            "\nEither pick a different dataset (see --dataset-dir) or run the collector again."
        )

    print(f"[dataset] Loaded {path}")
    return path, bundle

def auto_find_dataset(args):
    """Return a dataset path based on --dataset-path or composed slug in --dataset-dir."""
    # 1) Direct path wins
    if args.dataset_path:
        path = args.dataset_path
        if os.path.isfile(path):
            return path
        else:
            raise FileNotFoundError(f"--dataset-path not found: {path}")

    # 2) Explicit slug override
    if args.dataset_slug:
        cand = os.path.join(args.dataset_dir, args.dataset_slug + ".pth")
        if os.path.isfile(cand):
            return cand
        # allow loose match
        matches = [f for f in os.listdir(args.dataset_dir) if args.dataset_slug in f and f.endswith('.pth')]
        if len(matches) == 1:
            return os.path.join(args.dataset_dir, matches[0])
        if len(matches) > 1:
            raise RuntimeError(f"Multiple datasets match slug '{args.dataset_slug}': {matches}")
        raise FileNotFoundError(f"No dataset matches slug '{args.dataset_slug}' in {args.dataset_dir}")

    # 3) Compose slug from args
    order_tag = _norm_order_tag(args.code_order)
    model_tag = _model_tag_from_args(args)
    slug = make_dataset_slug(model=model_tag, order=order_tag,
                             n=int(args.n), dim=int(args.dim), msg=int(args.message_split),
                             Kt=int(args.K_t), pmin=int(args.min_p), pmax=int(args.max_p),
                             frac_random=float(args.frac_random), seed=int(args.seed))
    exact = os.path.join(args.dataset_dir, slug + ".pth")
    if os.path.isfile(exact):
        return exact

    # fallback: try to find a single fuzzy match sharing the cb(...) core
    core = f"ifed_{model_tag}_{order_tag}_cb({args.n}_{args.dim}_{args.message_split})"
    matches = [f for f in os.listdir(args.dataset_dir)
               if f.startswith(core) and f.endswith('.pth')]
    if len(matches) == 1:
        return os.path.join(args.dataset_dir, matches[0])
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple datasets match settings; specify --dataset-slug or --dataset-path.\n" +
            "Candidates: " + ", ".join(matches)
        )
    raise FileNotFoundError(
        f"Could not find dataset. Looked for '{slug}.pth' or unique match under {args.dataset_dir}."
    )
