"""
generate_experiments_resnet56.py

Generates config_resnet56.json and config_resnet56_smoke.json for CIFAR-10.

ResNet-56 architecture (chenyaofo/pytorch-cifar-models):
  - conv1:               3  → 16, 3×3
  - layer1.{0..8}.conv1: 16 → 16, 3×3   (9 blocks × 2 convs = 18 layers)
  - layer1.{0..8}.conv2: 16 → 16, 3×3
  - layer2.{0..8}.conv1: 16 → 32, 3×3   (first block transitions)
  - layer2.{0..8}.conv2: 32 → 32, 3×3
  - layer2.0.downsample.0: 16→32, 1×1   ← EXCLUDED (skip connection 1×1)
  - layer3.{0..8}.conv1: 32 → 64, 3×3
  - layer3.{0..8}.conv2: 64 → 64, 3×3
  - layer3.0.downsample.0: 32→64, 1×1   ← EXCLUDED (skip connection 1×1)
  - fc:                  64 → 10  (Linear)

Total: 54 conv 3×3 + 1 fc = 55 target layers.
Baseline accuracy: ~94.37% on CIFAR-10.
Total parameters: ~853K.

Break-even ranks (SVD) per group:
  conv1        (3→16,  3×3): r* = (3×16×9)/(16+27)   ≈ 10
  layer1.*     (16→16, 3×3): r* = (16×16×9)/(16+144) ≈ 14
  layer2.0.c1  (16→32, 3×3): r* = (16×32×9)/(32+144) ≈ 26
  layer2.*.c2  (32→32, 3×3): r* = (32×32×9)/(32+288) ≈ 29
  layer3.0.c1  (32→64, 3×3): r* = (32×64×9)/(64+288) ≈ 52
  layer3.*.c2  (64→64, 3×3): r* = (64×64×9)/(64+576) ≈ 57
  fc           (64→10):      r* = (64×10)/(64+10)    ≈ 8

These caps are enforced automatically by the decomposition code (no hard-coding needed here).
"""

import json
import math
import numpy as np

# ---------------------------------------------------------------------------
# ResNet-56 CIFAR-10 — explicit layer shapes (excluding 1×1 downsamples)
# ---------------------------------------------------------------------------

# All 3×3 conv layers: (out, in, kh, kw)
CONV_LAYERS = [
    # conv1
    (16,  3, 3, 3),

    # layer1: 9 blocks × 2 convs each, all 16→16
    (16, 16, 3, 3), (16, 16, 3, 3),   # layer1.0
    (16, 16, 3, 3), (16, 16, 3, 3),   # layer1.1
    (16, 16, 3, 3), (16, 16, 3, 3),   # layer1.2
    (16, 16, 3, 3), (16, 16, 3, 3),   # layer1.3
    (16, 16, 3, 3), (16, 16, 3, 3),   # layer1.4
    (16, 16, 3, 3), (16, 16, 3, 3),   # layer1.5
    (16, 16, 3, 3), (16, 16, 3, 3),   # layer1.6
    (16, 16, 3, 3), (16, 16, 3, 3),   # layer1.7
    (16, 16, 3, 3), (16, 16, 3, 3),   # layer1.8

    # layer2.0: transition 16→32 (conv1) then 32→32 (conv2)
    (32, 16, 3, 3), (32, 32, 3, 3),   # layer2.0
    # layer2.1–8: all 32→32
    (32, 32, 3, 3), (32, 32, 3, 3),   # layer2.1
    (32, 32, 3, 3), (32, 32, 3, 3),   # layer2.2
    (32, 32, 3, 3), (32, 32, 3, 3),   # layer2.3
    (32, 32, 3, 3), (32, 32, 3, 3),   # layer2.4
    (32, 32, 3, 3), (32, 32, 3, 3),   # layer2.5
    (32, 32, 3, 3), (32, 32, 3, 3),   # layer2.6
    (32, 32, 3, 3), (32, 32, 3, 3),   # layer2.7
    (32, 32, 3, 3), (32, 32, 3, 3),   # layer2.8

    # layer3.0: transition 32→64 (conv1) then 64→64 (conv2)
    (64, 32, 3, 3), (64, 64, 3, 3),   # layer3.0
    # layer3.1–8: all 64→64
    (64, 64, 3, 3), (64, 64, 3, 3),   # layer3.1
    (64, 64, 3, 3), (64, 64, 3, 3),   # layer3.2
    (64, 64, 3, 3), (64, 64, 3, 3),   # layer3.3
    (64, 64, 3, 3), (64, 64, 3, 3),   # layer3.4
    (64, 64, 3, 3), (64, 64, 3, 3),   # layer3.5
    (64, 64, 3, 3), (64, 64, 3, 3),   # layer3.6
    (64, 64, 3, 3), (64, 64, 3, 3),   # layer3.7
    (64, 64, 3, 3), (64, 64, 3, 3),   # layer3.8
]

# Linear layers: (in, out)
LINEAR_LAYERS = [
    (64, 10),   # fc
]

# Fixed overhead: BN params + 1×1 downsample convs (excluded from compression).
# layer2.0.downsample.0: 32*16*1*1 = 512
# layer3.0.downsample.0: 64*32*1*1 = 2048
# BN layers (each BN has 2*channels params):
#   conv1 BN: 2*16=32; layer1 BNs: 9*2*(2*16)=576; layer2 BNs: 9*2*(2*32)=1152
#   layer3 BNs: 9*2*(2*64)=2304; downsample BNs: 2*(2*32)+2*(2*64)=384
# Total fixed ≈ 512 + 2048 + 32 + 576 + 1152 + 2304 + 384 + fc_bias(10) = 7028
FIXED_OVERHEAD = 7_028

# Explicit target layer names (all 3×3 convs + fc, no 1×1 downsamples)
TARGET_LAYERS = [
    "conv1",
    # layer1 — 9 blocks
    "layer1.0.conv1", "layer1.0.conv2",
    "layer1.1.conv1", "layer1.1.conv2",
    "layer1.2.conv1", "layer1.2.conv2",
    "layer1.3.conv1", "layer1.3.conv2",
    "layer1.4.conv1", "layer1.4.conv2",
    "layer1.5.conv1", "layer1.5.conv2",
    "layer1.6.conv1", "layer1.6.conv2",
    "layer1.7.conv1", "layer1.7.conv2",
    "layer1.8.conv1", "layer1.8.conv2",
    # layer2 — 9 blocks (skip layer2.0.downsample.0)
    "layer2.0.conv1", "layer2.0.conv2",
    "layer2.1.conv1", "layer2.1.conv2",
    "layer2.2.conv1", "layer2.2.conv2",
    "layer2.3.conv1", "layer2.3.conv2",
    "layer2.4.conv1", "layer2.4.conv2",
    "layer2.5.conv1", "layer2.5.conv2",
    "layer2.6.conv1", "layer2.6.conv2",
    "layer2.7.conv1", "layer2.7.conv2",
    "layer2.8.conv1", "layer2.8.conv2",
    # layer3 — 9 blocks (skip layer3.0.downsample.0)
    "layer3.0.conv1", "layer3.0.conv2",
    "layer3.1.conv1", "layer3.1.conv2",
    "layer3.2.conv1", "layer3.2.conv2",
    "layer3.3.conv1", "layer3.3.conv2",
    "layer3.4.conv1", "layer3.4.conv2",
    "layer3.5.conv1", "layer3.5.conv2",
    "layer3.6.conv1", "layer3.6.conv2",
    "layer3.7.conv1", "layer3.7.conv2",
    "layer3.8.conv1", "layer3.8.conv2",
    # fc
    "fc",
]

# ---------------------------------------------------------------------------
# Max ranks per method — calibrated for ResNet-56's small layers.
# The decomposition code automatically enforces break-even caps per layer,
# so using a conservatively high max here is safe (extra ranks get capped).
# SVD/Tucker: layer1 break-even ≈ 14, layer3 break-even ≈ 57 → cap at 60.
# CP:         layer3 break-even is higher (≈ 92) → cap at 90.
# TT:         same SVD-derived cap as SVD.
# ---------------------------------------------------------------------------
METHODS_BOUNDS = [
    # (method, min_rank, max_rank)
    # Only CP is enabled to avoid recalculating the other 3 algorithms.
    # CP break-even: layer3.c2 (64→64, 3×3) = 36864/134 ≈ 275 → cap at 275
    ("CP",     2, 275),
]

METHODS_BOUNDS_SMOKE = [
    ("CP",     2, 275),
]


# ---------------------------------------------------------------------------
# Analytical parameter-count functions (mirror decomposition code exactly)
# ---------------------------------------------------------------------------

def _svd_conv(out, inp, kh, kw, r):
    max_valid = min(out, inp * kh * kw)
    eff = min(r, max_valid)
    max_comp = max(1, int((out * inp * kh * kw) / max(1, out + inp * kh * kw)))
    eff = min(eff, max_comp)
    return eff * (out + inp * kh * kw)


def _tucker_conv(out, inp, kh, kw, r):
    R_out = min(r, out)
    R_in  = min(r, inp)
    return R_in * inp + R_out * R_in * kh * kw + out * R_out


def _cp_conv(out, inp, kh, kw, r):
    denom = max(1, inp + out + kh + kw)
    max_r = max(1, int((out * inp * kh * kw) / denom))
    eff   = min(r, max_r)
    return eff * (inp + kh + kw + out)


def _tt_conv(out, inp, kh, kw, r):
    r1 = min(r, min(out, inp * kh * kw))
    r2 = min(r, min(out * inp, kh * kw))
    r3 = min(r, min(out * inp * kh, kw))
    return out * r1 + r1 * inp * r2 + r2 * kh * r3 + r3 * kw


def _linear(inp, out, r):
    eff = min(r, min(inp, out))
    return eff * (inp + out) + out   # +out for bias


_CONV_FN = {
    "SVD":    _svd_conv,
    "Tucker": _tucker_conv,
    "CP":     _cp_conv,
    "TT":     _tt_conv,
}


def total_params(method: str, rank: int) -> int:
    fn   = _CONV_FN[method]
    conv = sum(fn(o, i, kh, kw, rank) for o, i, kh, kw in CONV_LAYERS)
    lin  = sum(_linear(i, o, rank) for i, o in LINEAR_LAYERS)
    return conv + lin + FIXED_OVERHEAD


# ---------------------------------------------------------------------------
# Rank sampling (log-param spaced, same strategy as VGG generator)
# ---------------------------------------------------------------------------

def _closest_rank(method: str, target: float, lo: int, hi: int) -> int:
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
        "learning_rate":               1e-4,
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


def build_config(num_samples: int, ft_epochs: int,
                 max_train_batches: int, max_val_batches: int,
                 bounds=None) -> dict:
    if bounds is None:
        bounds = METHODS_BOUNDS

    global_settings = {
        "dataset":     "cifar10",
        "model":       "resnet56",
        "batch_size":  128,
        "use_gpu":     True,
        "num_classes": 10,
        "pretrained":  True,
        "fine_tuning": _fine_tuning_block(ft_epochs, max_train_batches, max_val_batches),
    }

    experiments = []
    for method, min_r, max_r in bounds:
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
        "resource_limits": {"max_rank": 200, "max_batch_size": 256},
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
    baseline = 853_018   # ResNet-56 CIFAR-10 baseline params

    print(f"\n{'─'*72}")
    print(f"  {label}")
    print(f"  {n_json} experiments  →  {n_csv} CSV rows")
    print(f"{'─'*72}")

    for method, _min_r, max_r in bounds:
        ranks  = [e["rank"]  for e in exps if e["method"] == method]
        params = [total_params(method, r) for r in ranks]
        cr_min = baseline / max(params)
        cr_max = baseline / min(params)
        print(
            f"  {method:8s}: {len(ranks):2d} ranks  "
            f"r=[{ranks[0]:3d}…{ranks[-1]:3d}]  "
            f"params=[{params[0]:>8,} … {params[-1]:>8,}]  "
            f"CR=[{cr_min:.2f}x … {cr_max:.0f}x]"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    smoke_cfg = build_config(
        num_samples       = 5,
        ft_epochs         = 3,
        max_train_batches = 5,
        max_val_batches   = 3,
    )
    full_cfg = build_config(
        num_samples       = 30,
        ft_epochs         = 10,
        max_train_batches = 50,
        max_val_batches   = 1,
    )

    with open("config_resnet56_smoke.json", "w") as f:
        json.dump(smoke_cfg, f, indent=4)
    with open("config_resnet56.json", "w") as f:
        json.dump(full_cfg, f, indent=4)

    _print_summary(smoke_cfg, "Smoke test         (config_resnet56_smoke.json)")
    _print_summary(full_cfg,  "Full FT experiment (config_resnet56.json)")

    n_exps  = len(full_cfg["experiments"])
    est_s   = n_exps * 120   # ~2 min avg per experiment (FT included)
    print(f"  Rough full estimate: {est_s/3600:.1f} h on Kaggle GPU")
    print()


if __name__ == "__main__":
    main()
