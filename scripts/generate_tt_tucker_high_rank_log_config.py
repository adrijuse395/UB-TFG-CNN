import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


TARGET_LAYERS = [
    "features.0",
    "features.4",
    "features.8",
    "features.11",
    "features.15",
    "features.18",
    "features.22",
    "features.25",
]


def make_log_ranks(min_rank: int = 72, max_rank: int = 400, n_points: int = 14) -> List[int]:
    """
    Build strictly increasing, approximately logarithmic ranks.
    Keeps all ranks > 64 and <= 400.
    """
    if min_rank <= 64:
        raise ValueError("min_rank must be > 64.")
    if max_rank < min_rank:
        raise ValueError("max_rank must be >= min_rank.")
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")

    ratio = (max_rank / min_rank) ** (1.0 / (n_points - 1))
    values = []
    x = float(min_rank)
    for _ in range(n_points):
        values.append(int(round(x)))
        x *= ratio
    values[0] = min_rank
    values[-1] = max_rank

    # Ensure strict monotonicity and bounds.
    ranks: List[int] = []
    for v in values:
        v = max(min_rank, min(max_rank, int(v)))
        if not ranks or v > ranks[-1]:
            ranks.append(v)
    if ranks[-1] != max_rank:
        ranks.append(max_rank)
    return ranks


def build_experiment(name_prefix: str, method: str, rank: int, fine_tuning: bool) -> Dict[str, Any]:
    return {
        "name": f"{name_prefix} rank {rank:03d} | {'ft' if fine_tuning else 'no_ft'}",
        "method": method,
        "target_layers": TARGET_LAYERS,
        "rank": rank,
        "fine_tuning": fine_tuning,
    }


def build_config(ranks: List[int]) -> Dict[str, Any]:
    experiments: List[Dict[str, Any]] = []

    # IMPORTANT ORDER: all Tucker first, then all TT.
    for rank in ranks:
        experiments.append(build_experiment("Tucker", "Tucker", rank, fine_tuning=False))
        experiments.append(build_experiment("Tucker", "Tucker", rank, fine_tuning=True))

    for rank in ranks:
        experiments.append(build_experiment("TT", "TT", rank, fine_tuning=False))
        experiments.append(build_experiment("TT", "TT", rank, fine_tuning=True))

    return {
        "global_settings": {
            "dataset": "cifar10",
            "model": "vgg11_bn",
            "batch_size": 128,
            "use_gpu": True,
            "num_classes": 10,
            "pretrained": True,
            "fine_tuning": {
                "epochs": 4,
                "learning_rate": 0.0001,
                "early_stopping": True,
                "patience": 1,
                "min_improvement": 0.5,
                "monitor": "val_accuracy",
                "max_train_batches_per_epoch": 60,
                "max_val_batches_per_epoch": 20,
                "kfold": 1,
                "kfold_seed": 42,
            },
        },
        "resource_limits": {
            "max_rank": 400,
            "max_batch_size": 256,
        },
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate high-rank logarithmic sweep config (Tucker then TT)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config_tt_tucker_rank_sweep_065_400_log.json",
        help="Path to output config file",
    )
    parser.add_argument(
        "--min-rank",
        type=int,
        default=72,
        help="Minimum rank (>64)",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=400,
        help="Maximum rank",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=14,
        help="Number of logarithmic points",
    )
    args = parser.parse_args()

    ranks = make_log_ranks(min_rank=args.min_rank, max_rank=args.max_rank, n_points=args.points)
    config = build_config(ranks)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(config, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} with {len(config['experiments'])} experiments.")
    print(f"Log ranks ({len(ranks)}): {ranks}")
    print("Order: Tucker sweep first, then TT sweep.")


if __name__ == "__main__":
    main()
