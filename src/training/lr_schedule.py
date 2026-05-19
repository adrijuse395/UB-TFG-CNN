"""
Learning-rate schedule for post-compression fine-tuning.

Below ``HIGH_ACCURACY_THRESHOLD`` the legacy linear mapping is unchanged.
At and above the threshold the LR continues from the same value at the threshold
but drops non-linearly (exponential) toward a small floor — no extra linear segment.

All constants are module-level defaults used when no override is supplied from the
experiment config (global_settings.fine_tuning.*).  Override them via config.json:

  "fine_tuning": {
    "ft_lr_max": 1e-3,
    "ft_lr_min": 1e-4,
    "ft_high_acc_threshold": 85.0,
    "ft_lr_floor": 1e-6,
    "ft_lr_decay_rate": 6.0
  }
"""

from __future__ import annotations

import math

FT_LR_MAX = 1e-3
FT_LR_MIN = 1e-4
HIGH_ACCURACY_THRESHOLD = 85.0
HIGH_ACCURACY_LR_FLOOR = 1e-6
# Steepness of exponential decay between threshold and 100% compressed test accuracy.
HIGH_ACCURACY_DECAY_RATE = 6.0


def linear_fine_tune_lr(
    compressed_test_acc: float,
    *,
    lr_max: float = FT_LR_MAX,
    lr_min: float = FT_LR_MIN,
) -> float:
    """Original mapping: higher compressed test accuracy → lower LR."""
    acc = max(0.0, min(100.0, float(compressed_test_acc)))
    return lr_min + (lr_max - lr_min) * (1.0 - acc / 100.0)


def resolve_dynamic_fine_tune_lr(
    compressed_test_acc: float,
    *,
    lr_max: float = FT_LR_MAX,
    lr_min: float = FT_LR_MIN,
    threshold: float = HIGH_ACCURACY_THRESHOLD,
    lr_floor: float = HIGH_ACCURACY_LR_FLOOR,
    decay_rate: float = HIGH_ACCURACY_DECAY_RATE,
) -> float:
    """
    Piecewise LR from compressed **test** accuracy (post-compression eval).

    - acc < threshold: unchanged linear schedule.
    - acc >= threshold: LR(threshold) from the linear branch, then exponential decay
      to ``lr_floor`` as acc approaches 100%.
    """
    acc = max(0.0, min(100.0, float(compressed_test_acc)))
    if acc < threshold:
        return round(linear_fine_tune_lr(acc, lr_max=lr_max, lr_min=lr_min), 6)

    lr_at_threshold = linear_fine_tune_lr(threshold, lr_max=lr_max, lr_min=lr_min)
    span = 100.0 - threshold
    t = (acc - threshold) / span
    decay = math.exp(-decay_rate * t)
    lr = lr_floor + (lr_at_threshold - lr_floor) * decay
    return round(lr, 6)


def use_high_accuracy_ft_regime(
    compressed_test_acc: float,
    *,
    threshold: float = HIGH_ACCURACY_THRESHOLD,
) -> bool:
    """Models already strong on test: avoid val-based checkpoint selection."""
    return float(compressed_test_acc) >= threshold
