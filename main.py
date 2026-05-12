"""
CNN Tensor Decomposition Evaluator — entry point only.

Orchestration lives in `src.experiments.runner.run_experiments_from_config`.
"""

import argparse

from src.experiments.runner import run_experiments_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN Tensor Decomposition Evaluator")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to experiment config JSON",
    )
    args = parser.parse_args()
    run_experiments_from_config(args.config)


if __name__ == "__main__":
    main()
