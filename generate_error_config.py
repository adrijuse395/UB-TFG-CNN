"""
generate_error_config.py

Generates an experiment config file for VGG11-BN / CIFAR-10 specifically
designed to measure mathematical reconstruction error (Frobenius norm).
  - No fine-tuning is performed.
  - Ranks are linearly spaced to provide an even sweep of parameters.
  - The configuration is saved to `configs/config_error.json`.
"""

import json
import os
import numpy as np

# ---------------------------------------------------------------------------
# VGG11-BN CIFAR-10 — only the 11 target layers
# ---------------------------------------------------------------------------
CONV_LAYERS = [
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
    (512, 512),          # classifier.0
    (512, 512),          # classifier.3
    (512,  10),          # classifier.6
]
FIXED_OVERHEAD = 8_256

TARGET_LAYERS = [
    "features.0",  "features.4",  "features.8",  "features.11",
    "features.15", "features.18", "features.22", "features.25",
    "classifier.0", "classifier.3", "classifier.6",
]

# Max ranks for no-FT configs: CP raised to 2000 to match SVD/Tucker/TT param range.
METHODS_BOUNDS_NO_FT = [
    ("SVD",    2,  400),
    ("Tucker", 2,  400),
    ("TT",     2,  400),
    ("CP",     2, 2000),
]


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
    return eff * (inp + out) + out

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

def build_config_error(num_samples: int) -> dict:
    global_settings = {
        "dataset":     "cifar10",
        "model":       "vgg11_bn",
        "batch_size":  128,
        "use_gpu":     True,
        "num_classes": 10,
        "pretrained":  True,
    }

    experiments = []
    for method, min_r, max_r in METHODS_BOUNDS_NO_FT:
        for r in get_param_spaced_ranks(method, min_r, max_r, num_samples):
            experiments.append({
                "name":          f"{method} rank {r:04d}",
                "method":        method,
                "target_layers": TARGET_LAYERS,
                "rank":          r,
                "fine_tuning":   False,
            })

    return {
        "global_settings": global_settings,
        "resource_limits": {"max_rank": 4000, "max_batch_size": 256},
        "experiments":     experiments,
    }

def main() -> None:
    os.makedirs("configs", exist_ok=True)
    
    # 20 samples per method for Kaggle evaluation without FT
    error_cfg = build_config_error(num_samples=20)

    out_path = "configs/config_error.json"
    with open(out_path, "w") as f:
        json.dump(error_cfg, f, indent=4)

    # Print summary
    exps     = error_cfg["experiments"]
    n_json   = len(exps)
    baseline = 9_756_426
    
    print(f"\n{'─'*64}")
    print(f"  Error Reconstruction Experiment ({out_path})")
    print(f"  {n_json} JSON entries")
    print(f"{'─'*64}")

    for method, _min_r, max_r in METHODS_BOUNDS_NO_FT:
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
        
    print(f"\nConfiguration successfully generated and saved to {out_path}.")
    print("Ready to run on Kaggle without Fine-tuning!")

if __name__ == "__main__":
    main()
