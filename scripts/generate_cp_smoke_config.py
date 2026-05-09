import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


RANKS_SMOKE = [2, 4, 8, 16]
CP_TARGET_LAYERS = [
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
    for rank in RANKS_SMOKE:
        experiments.append(
            {
                "name": f"CP smoke rank {rank:02d} | no_ft",
                "method": "CP",
                "target_layers": CP_TARGET_LAYERS,
                "rank": rank,
                "fine_tuning": False,
            }
        )
        experiments.append(
            {
                "name": f"CP smoke rank {rank:02d} | ft",
                "method": "CP",
                "target_layers": CP_TARGET_LAYERS,
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
            "cp_parafac_n_iter_max": 4,
            "cp_parafac_tol": 0.0001,
            "cp_parafac_on_cpu": True,
            # Keep disabled for smoke stability; can enable later if stable.
            "cp_memory_efficient_mttkrp": False,
            "cp_layer_timeout_s": 4.0,
            "cp_abort_if_mem_available_mb_below": 2500,
            "cp_init": "svd",
            "cp_normalize_factors": True,
            "cpu_num_threads": 2,
            "cpu_num_interop_threads": 1,
        },
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CP smoke-test config")
    parser.add_argument(
        "--output",
        type=str,
        default="config_cp_smoke.json",
        help="Path to output config file",
    )
    args = parser.parse_args()

    config = build_config()
    output_path = Path(args.output)
    output_path.write_text(json.dumps(config, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} with {len(config['experiments'])} experiments.")


if __name__ == "__main__":
    main()
