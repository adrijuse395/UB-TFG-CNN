"""
Runs the full CNN tensor-decomposition experiment pipeline from a JSON config.

`main.py` only parses CLI args and delegates here so orchestration stays testable
and separate from the entry point.

CSV logging: after layer replacement we log one row (experiment name may end in
`[compressed]` when FT is enabled). If fine-tuning is enabled, a second row is
logged after FT (`[fine_tuned]`). The analyzer infers display phase from names and
`fine_tuning_enabled` when needed.
"""

from __future__ import annotations

import copy
import gc
import os
import re
import time
from typing import Any, Dict, Optional

import torch

from src.data.factory import DatasetFactory
from src.decompositions.registry import DECOMPOSITION_REGISTRY
from src.decompositions.replacer import ModelReplacer
from src.evaluation.metrics import ModelEvaluator
from src.models.factory import ModelFactory
from src.training.fine_tune import fine_tune_model
from src.utils.config import ConfigParser
from src.utils.logger import RunLogger


def _experiment_base_name_for_ft_pair(exp_name: str) -> str:
    """
    Normaliza el nombre del experimento cuando el JSON lleva sufijo ' | ft'.

    Las filas CSV [compressed] / [fine_tuned] se construyen sobre esta base para que
    la fila pre-FT no arrastre '| ft' en el nombre.
    """
    return re.sub(r"\s*\|\s*ft\s*$", "", exp_name, flags=re.IGNORECASE).strip()


# Merged with `global_settings.fine_tuning` only — per-experiment FT hyperparameters are ignored.
DEFAULT_GLOBAL_FINE_TUNING: Dict[str, Any] = {
    "epochs": 3,
    "learning_rate": 1e-4,
    "early_stopping": True,
    "patience": 2,
    "min_improvement": 0.25,
    "monitor": "val_accuracy",
    "max_train_batches_per_epoch": 60,
    "max_val_batches_per_epoch": 20,
    "kfold": 1,
    "kfold_seed": 42,
}


def _resolved_fine_tuning_settings(global_settings: Dict[str, Any]) -> Dict[str, Any]:
    raw = global_settings.get("fine_tuning")
    user: Dict[str, Any] = raw if isinstance(raw, dict) else {}
    return {**DEFAULT_GLOBAL_FINE_TUNING, **user}


def _free_model(model, device: str) -> None:
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
    fine_tuning_time_s: float = 0.0,
) -> Dict[str, Any]:
    result["fine_tuning_enabled"] = enabled
    result["fine_tuning_time_s"] = round(fine_tuning_time_s, 4) if enabled else 0.0
    return result


def _resolve_device(global_settings: Dict[str, Any]) -> str:
    """
    Single switch: ``global_settings.use_gpu`` (default True).
    CUDA is used only when that flag is true *and* ``torch.cuda.is_available()``.
    """
    want_gpu = bool(global_settings.get("use_gpu", True))
    if want_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_compress_kwargs(method: str, rank: Any, method_params: Dict[str, Any]) -> Dict[str, Any]:
    """Rank plus all resolved method_params (parafac / CP-ALS / Tucker / TT knobs)."""
    return {"rank": rank, **(method_params or {})}


def _maybe_move_model_for_cp_compression(
    device: str,
    method: str,
    current_model: torch.nn.Module,
    compress_kw: Dict[str, Any],
) -> Dict[str, Any]:
    cp_compress_on_gpu = False
    if method == "CP":
        cp_compress_on_gpu = not compress_kw.get("cp_parafac_on_cpu", True)
    elif method == "CP_GD":
        cp_compress_on_gpu = not compress_kw.get("cp_gd_on_cpu", True)

    if cp_compress_on_gpu:
        if device.startswith("cuda") and torch.cuda.is_available():
            current_model.to(device)
        else:
            print(
                "    [Warning] CP-style decomposition requested on GPU but CUDA "
                "is unavailable; falling back to CPU factorization."
            )
            if method == "CP_GD":
                compress_kw["cp_gd_on_cpu"] = True
            else:
                compress_kw["cp_parafac_on_cpu"] = True
    return compress_kw


def run_experiments_from_config(config_path: str) -> Optional[str]:
    """
    Execute all experiments described by the JSON config.

    Returns the run directory path used by RunLogger, or None if the config file
    is missing (same behaviour as the former main() early exit).
    """
    print(f"[*] Loading configuration from {config_path}...")
    if not os.path.exists(config_path):
        print(f"[Error] Configuration file '{config_path}' not found.")
        return None

    config = ConfigParser.load_config(config_path)
    global_settings = config.get("global_settings", {})
    experiments = config.get("experiments", [])
    resource_limits_cfg = config.get(
        "resource_limits", global_settings.get("resource_limits", {})
    )
    resource_limits = ConfigParser.merge_resource_limits(resource_limits_cfg)

    device = _resolve_device(global_settings)
    print(f"[*] Using device: {device}")
    print(
        "[*] resource_limits:",
        f"max_rank={resource_limits['max_rank']}, "
        f"max_batch_size={resource_limits['max_batch_size']}, "
        "(method-specific params: method_defaults + experiment.method_params)",
    )
    print(
        "[*] use_gpu:",
        f"requested={bool(global_settings.get('use_gpu', True))}, "
        f"cuda_available={torch.cuda.is_available()}, resolved={device}",
    )

    logger = RunLogger(base_dir="runs", config=config)

    dataset_name = global_settings.get("dataset", "cifar10")
    requested_bs = int(global_settings.get("batch_size", 128))
    batch_size = min(requested_bs, resource_limits["max_batch_size"])
    if batch_size < requested_bs:
        print(f"[!] batch_size clamped to {batch_size} (resource_limits.max_batch_size).")

    print(f"[*] Loading Dataset: {dataset_name.upper()}...")
    train_loader, val_loader, test_loader, train_full_aug, train_full_eval = DatasetFactory.get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
    )
    input_shape = (1, 3, 32, 32) if dataset_name == "cifar10" else (1, 3, 224, 224)

    model_name = global_settings.get("model", "vgg11_bn")
    num_classes = 10 if dataset_name == "cifar10" else global_settings.get("num_classes", 10)
    pretrained = global_settings.get("pretrained", True)

    print(f"[*] Instantiating Model: {model_name.upper()} (Pretrained: {pretrained})...")
    base_model = ModelFactory.get_model(model_name, num_classes=num_classes, pretrained=pretrained)

    print("\n--- Evaluating Baseline ---")
    baseline_evaluator = ModelEvaluator(
        experiment_name="Baseline",
        device=device,
        baseline_params=None,
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
        fine_tuning_time_s=0.0,
    )
    logger.log_result(baseline_results)
    baseline_params = baseline_results["total_parameters"]

    base_model.cpu()
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[*] Found {len(experiments)} experiments in config.")

    for exp in experiments:
        exp_name = exp.get("name", "Unnamed Experiment")
        method = exp.get("method", "None")

        print(f"\n--- Running Experiment: {exp_name} (Method: {method}) ---")

        if method == "None":
            print("    [Skipping — Baseline already logged above]")
            continue

        if method not in DECOMPOSITION_REGISTRY:
            print(f"    [Error] Method '{method}' is not implemented. Skipping.")
            continue

        target_layers = exp.get("target_layers", [])
        rank = exp.get("rank")
        fine_tuning_enabled = bool(exp.get("fine_tuning", exp.get("fine_tunning", False)))
        ft_cfg = _resolved_fine_tuning_settings(global_settings)
        ft_epochs = max(1, int(ft_cfg["epochs"])) if fine_tuning_enabled else 0
        ft_lr = float(ft_cfg["learning_rate"]) if fine_tuning_enabled else 0.0
        ft_early_stopping = bool(ft_cfg["early_stopping"]) if fine_tuning_enabled else False
        ft_patience = max(1, int(ft_cfg["patience"])) if fine_tuning_enabled else 0
        ft_min_improvement = float(ft_cfg["min_improvement"]) if fine_tuning_enabled else 0.0
        ft_monitor = str(ft_cfg["monitor"]) if fine_tuning_enabled else ""
        ft_max_train_batches_per_epoch = (
            max(1, int(ft_cfg["max_train_batches_per_epoch"])) if fine_tuning_enabled else 0
        )
        ft_max_val_batches_per_epoch = (
            max(1, int(ft_cfg["max_val_batches_per_epoch"])) if fine_tuning_enabled else 0
        )
        ft_kfold = max(1, int(ft_cfg["kfold"])) if fine_tuning_enabled else 1
        ft_kfold_seed = int(ft_cfg["kfold_seed"]) if fine_tuning_enabled else 42

        if not target_layers:
            print("    [Warning] No target_layers specified. Model unchanged.")

        rank_clamped = ConfigParser.clamp_rank_for_method(rank, method, resource_limits)
        if rank_clamped != rank:
            print(f"    [!] rank clamped by resource_limits: {rank!r} -> {rank_clamped!r}")
        rank = rank_clamped

        method_params = ConfigParser.resolve_method_params(
            method=method,
            config=config,
            global_settings=global_settings,
            experiment=exp,
        )
        compress_kw = _build_compress_kwargs(method, rank, method_params)

        current_model = copy.deepcopy(base_model)
        compress_kw = _maybe_move_model_for_cp_compression(device, method, current_model, compress_kw)

        # Wall time for decomposition / layer replacement only (excludes test eval and fine-tuning).
        t0 = time.perf_counter()
        ModelReplacer.replace_layers(
            module=current_model,
            decomposition_class=DECOMPOSITION_REGISTRY[method],
            target_layers=target_layers,
            **compress_kw,
        )
        compression_time_s = time.perf_counter() - t0
        print(f"    Layer replacement took {compression_time_s:.4f}s")

        if fine_tuning_enabled:
            ft_pair_base = _experiment_base_name_for_ft_pair(exp_name)
            compressed_row_name = f"{ft_pair_base} [compressed]"
            fine_tuned_row_name = f"{ft_pair_base} [fine_tuned]"
        else:
            compressed_row_name = exp_name

        # Always log post-compression metrics (sin FT). Si fine_tuning está activo,
        # más abajo se añade una segunda fila con el modelo tras FT.
        print("    [Eval] Post-compression (sin fine-tuning)...")
        compressed_evaluator = ModelEvaluator(
            experiment_name=compressed_row_name,
            device=device,
            baseline_params=baseline_params,
        )
        compressed_results = compressed_evaluator.evaluate_all(
            model=current_model,
            dataloader=test_loader,
            input_shape=input_shape,
            method=method,
            target_layers=target_layers,
            rank=rank,
            compression_time_s=compression_time_s,
        )
        _attach_fine_tuning_metadata(
            compressed_results,
            enabled=False,
            fine_tuning_time_s=0.0,
        )
        logger.log_result(compressed_results)

        if fine_tuning_enabled:
            print(
                f"    [FineTuning] global_settings.fine_tuning → "
                f"epochs={ft_epochs}, lr={ft_lr}, early_stopping={ft_early_stopping}, "
                f"patience={ft_patience}, min_improvement={ft_min_improvement}, "
                f"monitor={ft_monitor}, max_train_batches={ft_max_train_batches_per_epoch}, "
                f"max_val_batches={ft_max_val_batches_per_epoch}, kfold={ft_kfold}, "
                f"kfold_seed={ft_kfold_seed}"
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
                kfold_splits=ft_kfold,
                train_full_aug=train_full_aug,
                train_full_eval=train_full_eval,
                kfold_seed=ft_kfold_seed,
                batch_size=batch_size,
                dataloader_num_workers=2,
                pin_memory=True,
            )

            print("    [FineTuning] Re-evaluating model after fine-tuning...")
            ft_evaluator = ModelEvaluator(
                experiment_name=fine_tuned_row_name,
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
                fine_tuning_time_s=float(ft_info["fine_tuning_time_s"]),
            )
            logger.log_result(ft_results)

        _free_model(current_model, device)
        gc.collect()

    out_dir = logger.directory
    print(f"\n[*] All experiments complete. Results saved to: {out_dir}")
    return out_dir
