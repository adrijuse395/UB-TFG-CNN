"""
src/evaluation/metrics.py

ModelEvaluator: Full evaluation suite for a compressed/baseline CNN model.

Metrics computed:
  - total_parameters    : trainable parameter count
  - compression_ratio   : baseline_params / current_params
  - compression_time_s  : time to replace layers (external)
  - macs_g              : GigaMACs for one forward pass (via thop graph trace)
  - latency_ms          : mean latency per sample (batch=1, 100 warm runs)
  - throughput_fps      : samples/s on the full test set (batched)
  - accuracy            : top-1 accuracy (%) on test set
  - precision / recall / f1: macro-averaged (%) via sklearn
"""

import time
import json
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from thop import profile as thop_profile
from sklearn.metrics import precision_recall_fscore_support


class ModelEvaluator:
    """
    Stateful evaluator for a specific experiment.
    Produces a flat dict of all metrics, ready to be logged to CSV.
    """

    def __init__(
        self,
        experiment_name: str,
        device: str = "cpu",
        baseline_params: Optional[int] = None,
    ):
        """
        Args:
            experiment_name: Human-readable name of this experiment.
            device:          Torch device string ('cpu' or 'cuda').
            baseline_params: Parameter count of the original model.
                             Used to compute compression_ratio (None → ratio = 1.0).
        """
        self.experiment_name = experiment_name
        self.device          = device
        self.baseline_params = baseline_params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        input_shape: tuple = (1, 3, 32, 32),
        method: str = "None",
        target_layers=None,
        rank=None,
        compression_time_s: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Runs the full evaluation suite and returns a flat result dict.

        Args:
            model:              The (possibly compressed) model to evaluate.
            dataloader:         Test DataLoader.
            input_shape:        Shape for latency/MACs dummy input.
            method:             Decomposition method name (for logging).
            target_layers:      List of compressed layer names (for logging).
            rank:               Rank used in decomposition (for logging).
            compression_time_s: Time taken to replace layers, in seconds.

        Returns:
            Dict whose keys match CSV_HEADERS in logger.py.
        """
        print(f"    -> Evaluating '{self.experiment_name}'...")

        total_params = self._count_parameters(model)
        compression_ratio = (
            1.0
            if self.baseline_params is None
            else round(self.baseline_params / total_params, 4)
        )

        macs_g       = self._measure_macs(model, input_shape)
        latency_ms   = self._measure_latency(model, input_shape)
        throughput_fps, accuracy, precision, recall, f1 = self._evaluate_on_dataset(
            model, dataloader
        )

        # Serialise list/complex fields for CSV
        if target_layers is None:
            target_layers_str = "None"
        elif isinstance(target_layers, list):
            target_layers_str = json.dumps(target_layers)
        else:
            target_layers_str = str(target_layers)

        rank_str = (
            json.dumps(rank) if isinstance(rank, list)
            else str(rank) if rank is not None
            else "None"
        )

        results = {
            "experiment_name":    self.experiment_name,
            "method":             method,
            "target_layers":      target_layers_str,
            "rank":               rank_str,
            "total_parameters":   total_params,
            "compression_ratio":  compression_ratio,
            "compression_time_s": round(compression_time_s, 4),
            "macs_g":             round(macs_g, 4),
            "latency_ms":         round(latency_ms, 4),
            "throughput_fps":     round(throughput_fps, 2),
            "accuracy":           round(accuracy, 4),
            "precision":          round(precision, 4),
            "recall":             round(recall, 4),
            "f1_score":           round(f1, 4),
        }

        # Pretty-print
        print(f"       Parameters       : {total_params:,}")
        print(f"       Compression Ratio: {compression_ratio:.4f}x")
        print(f"       GMACs            : {macs_g:.4f}")
        print(f"       Latency          : {latency_ms:.2f} ms")
        print(f"       Throughput       : {throughput_fps:.0f} FPS")
        print(f"       Accuracy         : {accuracy:.2f}%")
        print(f"       Precision (macro): {precision:.2f}%")
        print(f"       Recall (macro)   : {recall:.2f}%")
        print(f"       F1 (macro)       : {f1:.2f}%")

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _measure_macs(self, model: nn.Module, input_shape: tuple) -> float:
        """
        Computes GigaMACs (10^9 multiply-accumulate operations) for one forward
        pass using thop, which traces the actual computation graph.

        This is distinct from parameter count: a Tucker-decomposed layer may have
        far fewer weights but its GMACs depend on the spatial size of the feature
        maps it operates on and on the intermediate rank dimensions — relationships
        that can't be deduced from parameter counts alone.
        """
        model.eval()
        model.to(self.device)
        dummy = torch.randn(input_shape).to(self.device)
        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return macs / 1e9   # convert to GMACs

    def _measure_latency(self, model: nn.Module, input_shape: tuple,
                         num_runs: int = 100) -> float:
        """Mean latency per single sample (ms), averaged over num_runs warm passes."""
        model.eval()
        model.to(self.device)
        dummy = torch.randn(input_shape).to(self.device)
        is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(dummy)
        if is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                model(dummy)
        if is_cuda:
            torch.cuda.synchronize()

        return ((time.perf_counter() - start) * 1000) / num_runs

    def _evaluate_on_dataset(
        self, model: nn.Module, dataloader: torch.utils.data.DataLoader
    ):
        """
        Single pass over the test set.
        Returns: throughput_fps, accuracy, precision, recall, f1  (% where applicable).
        """
        model.eval()
        model.to(self.device)
        is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()

        all_targets, all_predictions = [], []
        total_inference_time = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)

                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs = model(inputs)
                if is_cuda:
                    torch.cuda.synchronize()
                total_inference_time += time.perf_counter() - t0

                total_samples += targets.size(0)
                _, predicted = outputs.max(1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        throughput_fps = (
            total_samples / total_inference_time if total_inference_time > 0 else 0.0
        )
        accuracy = (
            100.0 * sum(p == t for p, t in zip(all_predictions, all_targets))
            / total_samples
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average="macro", zero_division=0
        )
        return throughput_fps, accuracy, precision * 100, recall * 100, f1 * 100
