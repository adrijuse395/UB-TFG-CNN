"""
src/utils/logger.py

RunLogger: Manages a timestamped run directory for each execution.
Saves experiment input config and appends result rows to a CSV incrementally,
so no data is lost if the process is interrupted mid-run.
"""

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List


# Ordered list of CSV columns — must match keys in the dicts passed to log_result()
CSV_HEADERS = [
    "experiment_name",
    "method",
    "target_layers",
    "rank",
    "fine_tuning_enabled",
    "fine_tuning_phase",
    "fine_tuning_epochs",
    "fine_tuning_learning_rate",
    "fine_tuning_time_s",
    "fine_tuning_early_stopping",
    "fine_tuning_patience",
    "fine_tuning_min_improvement",
    "fine_tuning_monitor",
    "fine_tuning_best_epoch",
    "fine_tuning_stopped_early",
    "fine_tuning_last_val_loss",
    "fine_tuning_last_val_accuracy",
    "total_parameters",
    "compression_ratio",
    "compression_time_s",
    "macs_g",
    "latency_ms",
    "throughput_fps",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]


class RunLogger:
    """
    Creates a timestamped run directory and handles all disk I/O for an experiment run.

    Usage:
        logger = RunLogger(base_dir="runs", config=full_config_dict)
        logger.log_result(metrics_dict)  # called after each experiment
    """

    def __init__(self, base_dir: str, config: Dict[str, Any]):
        """
        Args:
            base_dir: Root directory where run folders are created (e.g. "runs").
            config:   Full experiment configuration dict (global_settings + experiments).
        """
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = os.path.join(base_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # --- Save input configuration ---
        input_config_path = os.path.join(self.run_dir, "input_config.json")
        config_to_save = {"run_id": self.run_id, **config}
        with open(input_config_path, "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, indent=4, ensure_ascii=False)

        # --- Initialize CSV with headers ---
        self.csv_path = os.path.join(self.run_dir, "results.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

        print(f"[Logger] Run directory created: {self.run_dir}")
        print(f"[Logger] Input config saved  : {input_config_path}")
        print(f"[Logger] Results CSV ready   : {self.csv_path}")

    def log_result(self, result: Dict[str, Any]) -> None:
        """
        Appends a single experiment result row to the CSV immediately.
        Only columns in CSV_HEADERS are written; unknown keys are ignored.

        Args:
            result: Dict with keys matching CSV_HEADERS.
        """
        row = {col: result.get(col, "") for col in CSV_HEADERS}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(row)
        print(f"[Logger] Result logged for '{result.get('experiment_name', '?')}'")

    @property
    def directory(self) -> str:
        """Returns the absolute path of this run's directory."""
        return os.path.abspath(self.run_dir)
