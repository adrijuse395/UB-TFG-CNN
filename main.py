"""
main.py — CNN Tensor Decomposition Evaluator

Execution flow:
  1. Load config.json
  2. Initialise RunLogger → creates runs/run_<timestamp>/ and saves input_config.json
  3. Load dataset
  4. Load pretrained baseline model, evaluate it, log result immediately
  5. For each experiment: deepcopy baseline, replace layers, evaluate, log result,
     then delete the compressed model and free GPU memory before moving on
"""

import argparse
import copy
import gc
import os
import time
from typing import Any, Dict

import torch

from src.data.factory import DatasetFactory
from src.decompositions.cp import CPDecomposedLayer
from src.decompositions.replacer import ModelReplacer
from src.decompositions.tt import TTDecomposedLayer
from src.decompositions.tucker import TuckerDecomposedLayer
from src.evaluation.metrics import ModelEvaluator
from src.models.factory import ModelFactory
from src.training.fine_tune import fine_tune_model
from src.utils.config import ConfigParser
from src.utils.logger import RunLogger


DECOMPOSITION_REGISTRY = {
    "Tucker": TuckerDecomposedLayer,
    "CP":     CPDecomposedLayer,
    "TT":     TTDecomposedLayer,
}


def _free_model(model, device: str) -> None:
    """Move model off GPU and release its memory."""
    try:
        model.cpu()
    except Exception:
        pass
    del model
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _attach_fine_tuning_metadata(
    result: Dict[str, Any],
    *,
    enabled: bool,
    phase: str,
    epochs: int,
    learning_rate: float,
    fine_tuning_time_s: float,
    early_stopping: bool = False,
    patience: int = 0,
    min_improvement: float = 0.0,
    monitor: str = "",
    best_epoch: int = 0,
    stopped_early: int = 0,
    last_val_loss: float = 0.0,
    last_val_accuracy: float = 0.0,
) -> Dict[str, Any]:
    result["fine_tuning_enabled"] = enabled
    result["fine_tuning_phase"] = phase
    result["fine_tuning_epochs"] = epochs if enabled else 0
    result["fine_tuning_learning_rate"] = learning_rate if enabled else 0.0
    result["fine_tuning_time_s"] = round(fine_tuning_time_s, 4)
    result["fine_tuning_early_stopping"] = early_stopping if enabled else False
    result["fine_tuning_patience"] = patience if enabled else 0
    result["fine_tuning_min_improvement"] = min_improvement if enabled else 0.0
    result["fine_tuning_monitor"] = monitor if enabled else ""
    result["fine_tuning_best_epoch"] = best_epoch if enabled else 0
    result["fine_tuning_stopped_early"] = stopped_early if enabled else 0
    result["fine_tuning_last_val_loss"] = last_val_loss if enabled else 0.0
    result["fine_tuning_last_val_accuracy"] = last_val_accuracy if enabled else 0.0
    return result


def main():
    parser = argparse.ArgumentParser(description="CNN Tensor Decomposition Evaluator")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to experiment config JSON")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load configuration
    # ------------------------------------------------------------------ #
    print(f"[*] Loading configuration from {args.config}...")
    if not os.path.exists(args.config):
        print(f"[Error] Configuration file '{args.config}' not found.")
        return

    config = ConfigParser.load_config(args.config)
    global_settings = config.get("global_settings", {})
    experiments     = config.get("experiments", [])
    resource_limits = ConfigParser.merge_resource_limits(global_settings)

    # Execution selector (new):
    # global_settings.execution = {"gpu": true/false, "on": "cpu"|"gpu"}
    # Backward compatible with legacy global_settings.use_gpu.
    execution = global_settings.get("execution", {}) or {}
    exec_gpu_enabled = bool(execution.get("gpu", global_settings.get("use_gpu", True)))
    exec_on = str(execution.get("on", "gpu")).strip().lower()
    if exec_on not in {"cpu", "gpu"}:
        exec_on = "gpu"
    if exec_on == "cpu":
        device = "cpu"
    else:
        device = "cuda" if (torch.cuda.is_available() and exec_gpu_enabled) else "cpu"
    print(f"[*] Using device: {device}")
    print(
        "[*] resource_limits:",
        f"max_rank={resource_limits['max_rank']}, "
        f"max_target_layers={resource_limits['max_target_layers_per_experiment']}, "
        f"max_batch_size={resource_limits['max_batch_size']}, "
        f"cp_parafac_n_iter_max={resource_limits['cp_parafac_n_iter_max']}, "
        f"cp_parafac_on_cpu={resource_limits['cp_parafac_on_cpu']}, "
        f"cp_normalize_factors={resource_limits['cp_normalize_factors']}, "
        f"cp_layer_timeout_s={resource_limits['cp_layer_timeout_s']}, "
        f"cp_mem_guard_mb={resource_limits['cp_abort_if_mem_available_mb_below']}, "
        f"cp_init={resource_limits['cp_init']}",
    )
    print(
        "[*] execution:",
        f"requested_on={exec_on}, gpu_enabled={exec_gpu_enabled}, resolved_device={device}"
    )
    torch.set_num_threads(max(1, resource_limits["cpu_num_threads"]))
    try:
        torch.set_num_interop_threads(max(1, resource_limits["cpu_num_interop_threads"]))
    except RuntimeError:
        # set_num_interop_threads can only be called once in some runtimes.
        pass

    # ------------------------------------------------------------------ #
    # 2. Initialise run logger (creates directory + input_config.json)
    # ------------------------------------------------------------------ #
    logger = RunLogger(base_dir="runs", config=config)

    # ------------------------------------------------------------------ #
    # 3. Load dataset
    # ------------------------------------------------------------------ #
    dataset_name = global_settings.get("dataset", "cifar10")
    batch_size   = min(
        int(global_settings.get("batch_size", 128)),
        resource_limits["max_batch_size"],
    )
    if batch_size < int(global_settings.get("batch_size", 128)):
        print(
            f"[!] batch_size clamped to {batch_size} (resource_limits.max_batch_size)."
        )
    print(f"[*] Loading Dataset: {dataset_name.upper()}...")
    train_loader, val_loader, test_loader = DatasetFactory.get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
    )
    input_shape = (1, 3, 32, 32) if dataset_name == "cifar10" else (1, 3, 224, 224)

    # ------------------------------------------------------------------ #
    # 4. Load pretrained baseline model + evaluate
    # ------------------------------------------------------------------ #
    model_name  = global_settings.get("model", "vgg11_bn")
    num_classes = 10 if dataset_name == "cifar10" else global_settings.get("num_classes", 10)
    pretrained  = global_settings.get("pretrained", True)

    print(f"[*] Instantiating Model: {model_name.upper()} (Pretrained: {pretrained})...")
    base_model = ModelFactory.get_model(model_name, num_classes=num_classes, pretrained=pretrained)

    print("\n--- Evaluating Baseline ---")
    baseline_evaluator = ModelEvaluator(
        experiment_name="Baseline",
        device=device,
        baseline_params=None,   # ratio will be 1.0
    )
    baseline_results = baseline_evaluator.evaluate_all(
        model=base_model,
        dataloader=test_loader,
        input_shape=input_shape,
        method="None",
        target_layers=None,
        rank=None,
        compression_time_s=0.0,
    )
    _attach_fine_tuning_metadata(
        baseline_results,
        enabled=False,
        phase="baseline",
        epochs=0,
        learning_rate=0.0,
        fine_tuning_time_s=0.0,
    )
    logger.log_result(baseline_results)
    baseline_params   = baseline_results["total_parameters"]

    # Baseline evaluation leaves weights on `device`; keep a CPU copy for deepcopy
    # so we do not duplicate the full model on GPU during compression (VRAM spike).
    base_model.cpu()
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 5. Run compression experiments
    # ------------------------------------------------------------------ #
    print(f"\n[*] Found {len(experiments)} experiments in config.")

    for exp in experiments:
        exp_name = exp.get("name", "Unnamed Experiment")
        method   = exp.get("method", "None")

        print(f"\n--- Running Experiment: {exp_name} (Method: {method}) ---")

        if method == "None":
            print("    [Skipping — Baseline already logged above]")
            continue

        if method not in DECOMPOSITION_REGISTRY:
            print(f"    [Error] Method '{method}' is not implemented. Skipping.")
            continue

        target_layers = exp.get("target_layers", [])
        rank          = exp.get("rank")
        fine_tuning_enabled = bool(exp.get("fine_tuning", exp.get("fine_tunning", False)))
        ft_epochs = max(1, int(exp.get("epochs", 1))) if fine_tuning_enabled else 0
        ft_lr = float(exp.get("learning_rate", 1e-4)) if fine_tuning_enabled else 0.0
        ft_early_stopping = bool(exp.get("early_stopping", True)) if fine_tuning_enabled else False
        ft_patience = max(1, int(exp.get("patience", 3))) if fine_tuning_enabled else 0
        ft_min_improvement = float(
            exp.get("min_improvement", exp.get("threshold", exp.get("threashold", 0.1)))
        ) if fine_tuning_enabled else 0.0
        ft_monitor = str(exp.get("monitor", "val_accuracy")) if fine_tuning_enabled else ""
        ft_max_train_batches_per_epoch = (
            max(1, int(exp.get("max_train_batches_per_epoch", 60)))
            if fine_tuning_enabled
            else 0
        )
        ft_max_val_batches_per_epoch = (
            max(1, int(exp.get("max_val_batches_per_epoch", 20)))
            if fine_tuning_enabled
            else 0
        )

        max_tl = resource_limits["max_target_layers_per_experiment"]
        if len(target_layers) > max_tl:
            print(
                f"    [Error] {len(target_layers)} target_layers exceeds "
                f"resource_limits.max_target_layers_per_experiment ({max_tl}). Skipping."
            )
            continue

        if not target_layers:
            print("    [Warning] No target_layers specified. Model unchanged.")

        rank_clamped = ConfigParser.clamp_rank_for_method(rank, method, resource_limits)
        if rank_clamped != rank:
            print(
                f"    [!] rank clamped by resource_limits: {rank!r} -> {rank_clamped!r}"
            )
        rank = rank_clamped

        compress_kw = {"rank": rank}
        if method in {"CP", "CP_ALS_LIGHT"}:
            compress_kw["parafac_n_iter_max"] = resource_limits["cp_parafac_n_iter_max"]
            compress_kw["parafac_tol"] = resource_limits["cp_parafac_tol"]
            compress_kw["cp_parafac_on_cpu"] = resource_limits["cp_parafac_on_cpu"]
            compress_kw["cp_memory_efficient_mttkrp"] = resource_limits[
                "cp_memory_efficient_mttkrp"
            ]
            compress_kw["cp_layer_timeout_s"] = resource_limits["cp_layer_timeout_s"]
            compress_kw["cp_abort_if_mem_available_mb_below"] = resource_limits[
                "cp_abort_if_mem_available_mb_below"
            ]
            compress_kw["cp_init"] = resource_limits["cp_init"]
            compress_kw["cp_normalize_factors"] = resource_limits["cp_normalize_factors"]
        if method == "CP_HOSVD":
            # Reuse same switch semantics: on_cpu=True keeps SVD work off GPU.
            compress_kw["cp_hosvd_on_cpu"] = resource_limits["cp_parafac_on_cpu"]

        # Deep-copy baseline weights into a fresh model (CPU-side tensors).
        current_model = copy.deepcopy(base_model)
        # Optional: run CP factorization directly on GPU when configured.
        if method in {"CP", "CP_ALS_LIGHT"} and (not compress_kw.get("cp_parafac_on_cpu", True)):
            if device.startswith("cuda") and torch.cuda.is_available():
                current_model.to(device)
            else:
                print(
                    "    [Warning] CP requested on GPU but CUDA is unavailable; "
                    "falling back to CPU factorization."
                )
                compress_kw["cp_parafac_on_cpu"] = True

        # Replace the target layers and time the operation
        t0 = time.perf_counter()
        ModelReplacer.replace_layers(
            module=current_model,
            decomposition_class=DECOMPOSITION_REGISTRY[method],
            target_layers=target_layers,
            **compress_kw,
        )
        compression_time_s = time.perf_counter() - t0
        print(f"    Layer replacement took {compression_time_s:.4f}s")

        # If fine-tuning is enabled, skip the intermediate compressed-only evaluation
        # to avoid duplicated work/log rows. We log only the final fine-tuned result.
        if not fine_tuning_enabled:
            evaluator = ModelEvaluator(
                experiment_name=exp_name,
                device=device,
                baseline_params=baseline_params,
            )
            results = evaluator.evaluate_all(
                model=current_model,
                dataloader=test_loader,
                input_shape=input_shape,
                method=method,
                target_layers=target_layers,
                rank=rank,
                compression_time_s=compression_time_s,
            )
            _attach_fine_tuning_metadata(
                results,
                enabled=False,
                phase="compressed",
                epochs=0,
                learning_rate=0.0,
                fine_tuning_time_s=0.0,
            )
            logger.log_result(results)

        if fine_tuning_enabled:
            print(
                f"    [FineTuning] Starting fine-tuning: epochs={ft_epochs}, "
                f"learning_rate={ft_lr}, early_stopping={ft_early_stopping}, "
                f"patience={ft_patience}, min_improvement={ft_min_improvement}, "
                f"monitor={ft_monitor}, max_train_batches={ft_max_train_batches_per_epoch}, "
                f"max_val_batches={ft_max_val_batches_per_epoch}"
            )
            ft_info = fine_tune_model(
                current_model,
                train_loader,
                val_loader,
                device=device,
                epochs=ft_epochs,
                learning_rate=ft_lr,
                early_stopping=ft_early_stopping,
                patience=ft_patience,
                min_improvement=ft_min_improvement,
                monitor=ft_monitor,
                max_train_batches_per_epoch=ft_max_train_batches_per_epoch,
                max_val_batches_per_epoch=ft_max_val_batches_per_epoch,
            )

            print("    [FineTuning] Re-evaluating model after fine-tuning...")
            ft_evaluator = ModelEvaluator(
                experiment_name=f"{exp_name} [fine_tuned]",
                device=device,
                baseline_params=baseline_params,
            )
            ft_results = ft_evaluator.evaluate_all(
                model=current_model,
                dataloader=test_loader,
                input_shape=input_shape,
                method=method,
                target_layers=target_layers,
                rank=rank,
                compression_time_s=compression_time_s,
            )
            _attach_fine_tuning_metadata(
                ft_results,
                enabled=True,
                phase="fine_tuned",
                epochs=ft_epochs,
                learning_rate=ft_lr,
                fine_tuning_time_s=float(ft_info["fine_tuning_time_s"]),
                early_stopping=ft_early_stopping,
                patience=ft_patience,
                min_improvement=ft_min_improvement,
                monitor=str(ft_info["fine_tuning_monitor"]),
                best_epoch=int(ft_info["fine_tuning_best_epoch"]),
                stopped_early=int(ft_info["fine_tuning_stopped_early"]),
                last_val_loss=float(ft_info["fine_tuning_last_val_loss"]),
                last_val_accuracy=float(ft_info["fine_tuning_last_val_accuracy"]),
            )
            logger.log_result(ft_results)

        # Release compressed model memory before next experiment
        _free_model(current_model, device)
        gc.collect()

    print(f"\n[*] All experiments complete. Results saved to: {logger.directory}")


if __name__ == "__main__":
    main()
