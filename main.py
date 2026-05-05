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

import torch

from src.data.factory import DatasetFactory
from src.decompositions.cp import CPDecomposedLayer
from src.decompositions.replacer import ModelReplacer
from src.decompositions.tt import TTDecomposedLayer
from src.decompositions.tucker import TuckerDecomposedLayer
from src.evaluation.metrics import ModelEvaluator
from src.models.factory import ModelFactory
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

    device = "cuda" if torch.cuda.is_available() and global_settings.get("use_gpu", True) else "cpu"
    print(f"[*] Using device: {device}")
    print(
        "[*] resource_limits:",
        f"max_rank={resource_limits['max_rank']}, "
        f"max_target_layers={resource_limits['max_target_layers_per_experiment']}, "
        f"max_batch_size={resource_limits['max_batch_size']}, "
        f"cp_parafac_n_iter_max={resource_limits['cp_parafac_n_iter_max']}, "
        f"cp_parafac_on_cpu={resource_limits['cp_parafac_on_cpu']}",
    )

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
    _, _, test_loader = DatasetFactory.get_dataloaders(
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
        if method == "CP":
            compress_kw["parafac_n_iter_max"] = resource_limits["cp_parafac_n_iter_max"]
            compress_kw["parafac_tol"] = resource_limits["cp_parafac_tol"]
            compress_kw["cp_parafac_on_cpu"] = resource_limits["cp_parafac_on_cpu"]
            compress_kw["cp_memory_efficient_mttkrp"] = resource_limits[
                "cp_memory_efficient_mttkrp"
            ]

        # Deep-copy baseline weights into a fresh model (CPU-side tensors).
        current_model = copy.deepcopy(base_model)

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

        # Evaluate and log immediately — no accumulation in memory
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
        logger.log_result(results)

        # Release compressed model memory before next experiment
        _free_model(current_model, device)
        gc.collect()

    print(f"\n[*] All experiments complete. Results saved to: {logger.directory}")


if __name__ == "__main__":
    main()
