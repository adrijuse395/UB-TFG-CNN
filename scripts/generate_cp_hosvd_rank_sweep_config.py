import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


RANKS_20 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 58, 64]
RANKS_HIGH_LOG = [72, 82, 94, 107, 122, 139, 159, 181, 207, 236, 269, 307, 351, 400]
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


def build_config() -> Dict[str, Any]:
    experiments: List[Dict[str, Any]] = []
    all_ranks = RANKS_20 + RANKS_HIGH_LOG

    for rank in all_ranks:
        experiments.append(
            {
                "name": f"CP_HOSVD rank {rank:02d} | no_ft",
                "method": "CP_HOSVD",
                "target_layers": TARGET_LAYERS,
                "rank": rank,
                "fine_tuning": False,
            }
        )
        experiments.append(
            {
                "name": f"CP_HOSVD rank {rank:02d} | ft",
                "method": "CP_HOSVD",
                "target_layers": TARGET_LAYERS,
                "rank": rank,
                "fine_tuning": True,
                "epochs": 4,
                "learning_rate": 1e-4,
                "early_stopping": True,
                "patience": 1,
                "min_improvement": 0.5,
                "monitor": "val_accuracy",
                "max_train_batches_per_epoch": 60,
                "max_val_batches_per_epoch": 20,
            }
        )

    return {
        "global_settings": {
            "dataset": "cifar10",
            "model": "vgg11_bn",
            "batch_size": 128,
            "use_gpu": True,
            "num_classes": 10,
            "pretrained": True,
        },
        "resource_limits": {
            "max_rank": 64,
            "cp_parafac_n_iter_max": 25,
            "cp_parafac_tol": 0.0001,
            "cp_parafac_on_cpu": True,
            "cp_memory_efficient_mttkrp": True,
            "cp_normalize_factors": True,
            "cp_layer_timeout_s": 12.0,
            "cp_abort_if_mem_available_mb_below": 1200,
            "cp_init": "svd",
            "cpu_num_threads": 2,
            "cpu_num_interop_threads": 1,
        },
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CP_HOSVD rank sweep config")
    parser.add_argument(
        "--output",
        type=str,
        default="config_cp_hosvd_rank_sweep.json",
        help="Path to output config file",
    )
    args = parser.parse_args()
    config = build_config()
    out = Path(args.output)
    out.write_text(json.dumps(config, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} with {len(config['experiments'])} experiments.")
    print(f"Ranks ({len(RANKS_20 + RANKS_HIGH_LOG)}): {RANKS_20 + RANKS_HIGH_LOG}")


if __name__ == "__main__":
    main()
