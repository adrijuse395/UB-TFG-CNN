"""
generate_experiments.py

Generates two experiment config files for VGG11-BN / CIFAR-10:
  - config_smoke.json  : 10 ranks/method, limited batches — quick pipeline check
  - config_full.json   : 68 ranks/method, unlimited batches — final experiment

Key improvement over the previous version:
  Ranks are chosen so that the resulting total_parameters values are
  **evenly distributed on a log scale**.  This guarantees that the
  accuracy × total_parameters scatter plot has proportionally spaced
  data points across the entire compression range for every method,
  instead of a cluster at high rank and a gap in the middle.

  Technique: target param values are sampled with geomspace, then for
  each target a binary search finds the integer rank whose analytical
  parameter count is closest.  Analytical formulas are verified against
  run_20260515_100131 results (errors < 0.01%).
"""

import json
import numpy as np

# ---------------------------------------------------------------------------
# VGG11-BN CIFAR-10 — only the 11 target layers (verified from experiment CSV)
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

# Max ranks: SVD/Tucker/TT stay at 400 (near-baseline); CP capped at 1500
# to keep per-layer compression time ≤ ~200 s (rank 2000 took 317 s).
METHODS_BOUNDS = [
    # (method, min_rank, max_rank)
    ("SVD",    2,  400),
    ("Tucker", 2,  400),
    ("TT",     2,  400),
    ("CP",     2, 1500),
]


# ---------------------------------------------------------------------------
# Analytical parameter-count functions — match the actual decomposition code
# ---------------------------------------------------------------------------

def _svd_conv(out, inp, kh, kw, r):
    """SVD: two convs, no bias (BN after original layer)."""
    max_valid = min(out, inp * kh * kw)
    eff = min(r, max_valid)
    # Correct cap: decomposed params < original params
    # rank < (out * inp*kh*kw) / (out + inp*kh*kw)
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
# Parameter-space rank sampling
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

    Two-phase approach:
      1. Oversample: generate 100 log-spaced param targets → find closest rank for
         each → deduplicate.  This builds a pool of ~68–86 well-distributed ranks.
      2. Subsample: if the pool exceeds num_samples, take num_samples evenly-spaced
         by index (preserving log-param distribution since the pool is already sorted
         by monotone total_params).

    This is the best achievable distribution for integer ranks: at very low ranks the
    param function grows so steeply that multiple log-spaced targets collapse to the
    same integer rank, making perfect uniformity impossible there.  The important
    mid-range (where accuracy transitions) is always well-covered.
    """
    p_min   = total_params(method, min_rank)
    p_max   = total_params(method, max_rank)
    # Phase 1: build a log-param-spaced pool of unique ranks
    targets = np.geomspace(p_min, p_max, 100)
    seen, pool = set(), []
    for t in targets:
        r = _closest_rank(method, float(t), min_rank, max_rank)
        if r not in seen:
            seen.add(r)
            pool.append(r)
    pool.sort()
    # Phase 2: cap to num_samples by uniform index subsampling
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
        "min_improvement":             0.1,
        "monitor":                     "val_accuracy",
        "max_train_batches_per_epoch": max_train_batches,
        "max_val_batches_per_epoch":   max_val_batches,
        "kfold":                       1,
        "kfold_seed":                  42,
        # Overfitting guards for the high-accuracy "final" checkpoint regime
        "val_overfit_margin":  5.0,    # revert if val jumps > pre_ft + 5 pp
        "val_overfit_ceiling": 96.0,   # revert if val exceeds 96 % absolute
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

def _print_summary(cfg: dict, label: str) -> None:
    exps     = cfg["experiments"]
    n_json   = len(exps)
    n_csv    = n_json * 2           # each fine_tuning=True entry → 2 CSV rows
    baseline = 9_756_426            # VGG11-BN CIFAR-10 baseline params

    print(f"\n{'─'*64}")
    print(f"  {label}")
    print(f"  {n_json} JSON entries  →  {n_csv} CSV rows")
    print(f"{'─'*64}")

    for method, min_r, max_r in METHODS_BOUNDS:
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
    smoke_cfg = build_config(
        num_samples        = 10,
        ft_epochs          = 3,
        max_train_batches  = 5,    # ≈640 samples/epoch — pipeline sanity only
        max_val_batches    = 3,
    )
    full_cfg = build_config(
        num_samples        = 68,
        ft_epochs          = 5,
        max_train_batches  = 0,    # 0 = unlimited
        max_val_batches    = 0,
    )

    with open("config_smoke.json", "w") as f:
        json.dump(smoke_cfg, f, indent=4)
    with open("config_full.json", "w") as f:
        json.dump(full_cfg, f, indent=4)

    _print_summary(smoke_cfg, "Smoke test   (config_smoke.json)")
    _print_summary(full_cfg,  "Full experiment  (config_full.json)")

    # Rough timing estimate for the full experiment
    full_exps = full_cfg["experiments"]
    n_cp  = sum(1 for e in full_exps if e["method"] == "CP")
    n_rest = len(full_exps) - n_cp
    est_comp  = n_rest * 3 + n_cp * 55     # s — SVD/Tucker/TT ≈3s, CP ≈55s avg
    est_ft    = len(full_exps) * 75         # s — ~75 s FT per experiment
    est_total = (est_comp + est_ft) / 3600
    print(f"  Rough full-experiment estimate: {est_total:.1f} h on Kaggle GPU")
    print(f"  (CP compression dominates; {n_cp} CP experiments × ~55 s avg)")
    print()


if __name__ == "__main__":
    main()
