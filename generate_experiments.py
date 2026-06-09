"""
generate_experiments.py

Generates the basic experiment config file for VGG11-BN / CIFAR-10.
The configuration is saved to `configs/config.json`.

This script uses log-spaced parameters (geomspace) for even
distribution in accuracy × log(params) plots.
Analytical parameter-count formulas are used to find the exact ranks.
"""

import json
import os
import numpy as np

# ---------------------------------------------------------------------------
# VGG11-BN CIFAR-10 — only the 11 target layers
# ---------------------------------------------------------------------------
CONV_LAYERS = [
    # (out_channels, in_channels, kh, kw)
    ( 64,   3, 3, 3),   # features.0
    (128,  64, 3, 3),   # features.4
    (256, 128, 3, 3),   # features.8
    (256, 256, 3, 3),   # features.11
    (512, 256, 3, 3),   # features.15
    (512, 512, 3, 3),   # features.18
    (512, 512, 3, 3),   # features.22
    (512, 512, 3, 3),   # features.25
]
LINEAR_LAYERS = [
    # (in_features, out_features)
    (512, 512),          # classifier.0
    (512, 512),          # classifier.3
    (512,  10),          # classifier.6
]
# BN layers + non-target modules — does not change with rank
FIXED_OVERHEAD = 8_256

TARGET_LAYERS = [
    "features.0",  "features.4",  "features.8",  "features.11",
    "features.15", "features.18", "features.22", "features.25",
    "classifier.0", "classifier.3", "classifier.6",
]

# Max ranks for FT configs: SVD/Tucker/TT at 400; CP capped at 1500
# to keep per-layer compression time ≤ ~200 s.
METHODS_BOUNDS = [
    # (method, min_rank, max_rank)
    ("SVD",    2,  400),
    ("Tucker", 2,  400),
    ("TT",     2,  400),
    ("CP",     2, 1500),
]


# ---------------------------------------------------------------------------
# Analytical parameter-count functions
# ---------------------------------------------------------------------------

def _svd_conv(out, inp, kh, kw, r):
    """SVD: two convs, no bias (BN after original layer)."""
    max_valid = min(out, inp * kh * kw)
    eff = min(r, max_valid)
    max_comp = max(1, int((out * inp * kh * kw) / max(1, out + inp * kh * kw)))
    eff = min(eff, max_comp)
    return eff * (out + inp * kh * kw)


def _tucker_conv(out, inp, kh, kw, r):
    """Tucker-2: pointwise → core(kxk) → pointwise, no bias."""
    R_out = min(r, out)
    R_in  = min(r, inp)
    return R_in * inp + R_out * R_in * kh * kw + out * R_out


def _cp_conv(out, inp, kh, kw, r):
    """CP: four factor convs (depthwise for spatial dims), no bias."""
    denom = max(1, inp + out + kh + kw)
    max_r = max(1, int((out * inp * kh * kw) / denom))
    eff   = min(r, max_r)
    return eff * (inp + kh + kw + out)


def _tt_conv(out, inp, kh, kw, r):
    """TT: four cores [1,r1,r2,r3,1], ranks capped by tensorly rules."""
    r1 = min(r, min(out, inp * kh * kw))
    r2 = min(r, min(out * inp, kh * kw))
    r3 = min(r, min(out * inp * kh, kw))
    return out * r1 + r1 * inp * r2 + r2 * kh * r3 + r3 * kw


def _linear(inp, out, r):
    """SVD for Linear (all methods fall back to SVD here), with bias."""
    eff = min(r, min(inp, out))
    return eff * (inp + out) + out   # +out for bias


_CONV_FN = {
    "SVD":    _svd_conv,
    "Tucker": _tucker_conv,
    "CP":     _cp_conv,
    "TT":     _tt_conv,
}


def total_params(method: str, rank: int) -> int:
    """Analytical total parameter count for model compressed at `rank`."""
    fn   = _CONV_FN[method]
    conv = sum(fn(o, i, kh, kw, rank) for o, i, kh, kw in CONV_LAYERS)
    lin  = sum(_linear(i, o, rank) for i, o in LINEAR_LAYERS)
    return conv + lin + FIXED_OVERHEAD


# ---------------------------------------------------------------------------
# Rank sampling strategies
# ---------------------------------------------------------------------------

def _closest_rank(method: str, target: float, lo: int, hi: int) -> int:
    """Binary search: rank in [lo, hi] minimising |total_params(rank) - target|."""
    best_r, best_d = lo, abs(total_params(method, lo) - target)
    while lo <= hi:
        mid = (lo + hi) // 2
        p   = total_params(method, mid)
        d   = abs(p - target)
        if d < best_d:
            best_d, best_r = d, mid
        if p < target:
            lo = mid + 1
        elif p > target:
            hi = mid - 1
        else:
            return mid
    return best_r


def get_param_spaced_ranks(method: str, min_rank: int, max_rank: int, num_samples: int):
    """
    Return exactly *num_samples* unique integer ranks in [min_rank, max_rank] whose
    total_params values are as evenly distributed on a log scale as possible.
    """
    p_min   = total_params(method, min_rank)
    p_max   = total_params(method, max_rank)
    targets = np.geomspace(p_min, p_max, 100)
    seen, pool = set(), []
    for t in targets:
        r = _closest_rank(method, float(t), min_rank, max_rank)
        if r not in seen:
            seen.add(r)
            pool.append(r)
    pool.sort()
    
    if len(pool) <= num_samples:
        return pool
    idxs = np.round(np.linspace(0, len(pool) - 1, num_samples)).astype(int)
    return [pool[i] for i in idxs]


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _fine_tuning_block(epochs: int, max_train_batches: int, max_val_batches: int) -> dict:
    return {
        "epochs":                      epochs,
        "learning_rate":               1e-4,   # unused (dynamic LR takes over)
        "early_stopping":              True,
        "patience":                    3,
        "min_improvement":             0.01,
        "monitor":                     "train_loss",
        "max_train_batches_per_epoch": max_train_batches,
        "max_val_batches_per_epoch":   max_val_batches,
        "kfold":                       1,
        "kfold_seed":                  42,
        "checkpoint_strategy":         "best_train",
        "val_overfit_ceiling":         100.0,
    }


def build_config(
    num_samples: int,
    ft_epochs: int,
    max_train_batches: int,
    max_val_batches: int,
) -> dict:
    global_settings = {
        "dataset":     "cifar10",
        "model":       "vgg11_bn",
        "batch_size":  128,
        "use_gpu":     True,
        "num_classes": 10,
        "pretrained":  True,
        "fine_tuning": _fine_tuning_block(ft_epochs, max_train_batches, max_val_batches),
    }

    experiments = []
    for method, min_r, max_r in METHODS_BOUNDS:
        for r in get_param_spaced_ranks(method, min_r, max_r, num_samples):
            experiments.append({
                "name":          f"{method} rank {r:04d} | ft",
                "method":        method,
                "target_layers": TARGET_LAYERS,
                "rank":          r,
                "fine_tuning":   True,
            })

    return {
        "global_settings": global_settings,
        "resource_limits": {"max_rank": 4000, "max_batch_size": 256},
        "experiments":     experiments,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(cfg: dict, label: str, bounds=None) -> None:
    if bounds is None:
        bounds = METHODS_BOUNDS
    exps     = cfg["experiments"]
    n_json   = len(exps)
    has_ft   = any(e.get("fine_tuning") for e in exps)
    n_csv    = n_json * 2 if has_ft else n_json
    baseline = 9_756_426            # VGG11-BN CIFAR-10 baseline params

    print(f"\n{'─'*64}")
    print(f"  {label}")
    print(f"  {n_json} JSON entries  →  {n_csv} CSV rows")
    print(f"{'─'*64}")

    for method, _min_r, max_r in bounds:
        ranks  = [e["rank"]  for e in exps if e["method"] == method]
        params = [total_params(method, r) for r in ranks]
        cr_min = baseline / max(params)
        cr_max = baseline / min(params)
        print(
            f"  {method:8s}: {len(ranks):2d} ranks  "
            f"r=[{ranks[0]:4d}…{ranks[-1]:5d}]  "
            f"params=[{params[0]:>8,} … {params[-1]:>9,}]  "
            f"CR=[{cr_min:.2f}x … {cr_max:.0f}x]"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure configs directory exists
    os.makedirs("configs", exist_ok=True)
    
    full_cfg = build_config(
        num_samples        = 30,
        ft_epochs          = 10,
        max_train_batches  = 50,    # 50 batches * 128 = 6400 samples/epoch
        max_val_batches    = 1,     # only 1 batch to print a rough val accuracy very fast
    )

    out_path = "configs/config.json"
    with open(out_path, "w") as f:
        json.dump(full_cfg, f, indent=4)

    _print_summary(full_cfg,   f"Basic VGG11-BN Experiment  ({out_path})")

    print(f"Configuration successfully generated and saved to {out_path}.")
    print()


if __name__ == "__main__":
    main()
