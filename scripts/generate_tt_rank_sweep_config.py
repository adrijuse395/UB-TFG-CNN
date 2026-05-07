import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


RANKS_20 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 58, 64]
TT_TARGET_LAYERS = [
    "features.0",
    "features.4",
    "features.8",
    "features.11",
    "features.15",
    "features.18",
    "features.22",
    "features.25",
]


def build_config(method: str = "TT") -> Dict[str, Any]:
    method = method.strip()
    if method not in {"TT", "Tucker"}:
        raise ValueError("method must be 'TT' or 'Tucker'")

    experiments: List[Dict[str, Any]] = []

    for rank in RANKS_20:
        experiments.append(
            {
                "name": f"{method} rank {rank:02d} | no_ft",
                "method": method,
                "target_layers": TT_TARGET_LAYERS,
                "rank": rank,
                "fine_tuning": False,
            }
        )
        experiments.append(
            {
                "name": f"{method} rank {rank:02d} | ft",
                "method": method,
                "target_layers": TT_TARGET_LAYERS,
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
            "cp_layer_timeout_s": 12.0,
            "cp_abort_if_mem_available_mb_below": 1200,
            "cp_init": "random",
            "cpu_num_threads": 2,
            "cpu_num_interop_threads": 1,
        },
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rank sweep config (TT or Tucker)")
    parser.add_argument(
        "--output",
        type=str,
        default="config.json",
        help="Path to output config file",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="TT",
        choices=["TT", "Tucker"],
        help="Decomposition method for the sweep",
    )
    args = parser.parse_args()

    config = build_config(method=args.method)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(config, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Wrote {output_path} with {len(config['experiments'])} experiments "
        f"({len(RANKS_20)} ranks x 2 conditions, method={args.method})."
    )


if __name__ == "__main__":
    main()
