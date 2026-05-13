"""
src/evaluation/metrics.py

ModelEvaluator: Full evaluation suite for a compressed/baseline CNN model.

Metrics (see evaluate_all):
  - total_parameters       : trainable parameter count (requires_grad=True only)
  - compression_ratio      : baseline_params / current_params (same counting rule)
  - compression_time_s     : wall time for layer replacement (measured in runner, not here)
  - model_memory_mb        : sum of parameter+buffer tensor bytes / 1 MiB (static footprint)
  - peak_inference_memory_mb: peak CUDA bytes allocated only during the batched test-set loop
                              (0 if not CUDA; see LIMITATIONS below)
  - test_eval_time_s       : wall time for the full test-loader pass (H2D, forwards, syncs,
                              moving predictions to CPU for metrics — not sklearn post-process)
  - macs_g                 : thop analytic op count / 1e9 (multiply-accumulate style for convs;
                              not hardware FLOPs; custom layers may be missing — see LIMITATIONS)
  - latency_ms             : mean wall time of one batch-1 forward after warmup (see LIMITATIONS)
  - throughput_fps         : test_set_samples / sum(per-batch wall times including H2D + forward)
  - accuracy / precision / recall / f1 : on the test set (macro for PRF)

LIMITATIONS (read before interpreting CSV):
  - **CPU peak RAM**: we do not record process RSS; `peak_inference_memory_mb` is CUDA-only.
  - **CUDA latency / throughput**: timings use `torch.cuda.synchronize()` so each interval
    measures completed GPU work (not queued launches). This removes the classic bug where
    many forwards are timed with a single sync at the end (mean latency would be meaningless).
  - **CPU latency**: `latency_ms` is still wall time on CPU — cache misses, thermal throttling,
    OS scheduling, and other hardware effects add variance that is *not* removed by averaging;
    use repeated runs / confidence intervals for rigorous comparisons.
  - **thop / macs_g**: static graph estimate for one input shape; BN/training effects ignored;
    decompositions using custom modules may be under-counted unless thop registers hooks.
  - **total_parameters**: excludes frozen (`requires_grad=False`) weights; compression_ratio
    follows the same rule — compare to "all tensors on disk" only if nothing is frozen.
"""

import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from thop import profile as thop_profile


def _format_target_layers(target_layers) -> str:
    if target_layers is None:
        return "None"
    if isinstance(target_layers, list):
        return "|".join(str(x) for x in target_layers)
    return str(target_layers)


def _format_rank(rank) -> str:
    if rank is None:
        return "None"
    if isinstance(rank, list):
        return "|".join(str(x) for x in rank)
    return str(rank)


def _model_static_memory_mb(model: nn.Module) -> float:
    """Sum of parameter and buffer storage (MiB), device-agnostic size."""
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        total += int(t.numel()) * int(t.element_size())
    return total / (1024.0 * 1024.0)


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
        self.experiment_name = experiment_name
        self.device = device
        self.baseline_params = baseline_params

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
        print(f"    -> Evaluating '{self.experiment_name}'...")

        total_params = self._count_parameters(model)
        compression_ratio = (
            1.0
            if self.baseline_params is None
            else round(self.baseline_params / total_params, 4)
        )

        model_mem_mb = round(_model_static_memory_mb(model), 4)

        macs_g = self._measure_macs(model, input_shape)
        latency_ms = self._measure_latency(model, input_shape)

        is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()

        throughput_fps, accuracy, precision, recall, f1, test_eval_time_s, peak_inf_mb = (
            self._evaluate_on_dataset(model, dataloader, record_cuda_peak=is_cuda)
        )

        results = {
            "experiment_name": self.experiment_name,
            "method": method,
            "target_layers": _format_target_layers(target_layers),
            "rank": _format_rank(rank),
            "total_parameters": total_params,
            "compression_ratio": compression_ratio,
            "compression_time_s": round(compression_time_s, 4),
            "model_memory_mb": model_mem_mb,
            "peak_inference_memory_mb": round(peak_inf_mb, 4) if is_cuda else 0.0,
            "test_eval_time_s": round(test_eval_time_s, 4),
            "macs_g": round(macs_g, 4),
            "latency_ms": round(latency_ms, 4),
            "throughput_fps": round(throughput_fps, 2),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

        print(f"       Parameters       : {total_params:,}")
        print(f"       Compression Ratio: {compression_ratio:.4f}x")
        print(f"       Model memory     : {model_mem_mb:.2f} MiB (params+buffers)")
        if is_cuda:
            print(f"       Peak CUDA (test) : {peak_inf_mb:.2f} MiB")
        print(f"       Test eval time   : {test_eval_time_s:.3f} s")
        print(f"       GMACs            : {macs_g:.4f}")
        print(f"       Latency          : {latency_ms:.2f} ms")
        print(f"       Throughput       : {throughput_fps:.0f} FPS")
        print(f"       Accuracy         : {accuracy:.2f}%")
        print(f"       Precision (macro): {precision:.2f}%")
        print(f"       Recall (macro)   : {recall:.2f}%")
        print(f"       F1 (macro)       : {f1:.2f}%")

        return results

    def _count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _measure_macs(self, model: nn.Module, input_shape: tuple) -> float:
        """
        thop `profile` returns an analytic multiply-accumulate-style op total for one forward
        on `input_shape` (layer hooks; conv uses MAC-like products — see thop source).
        Divided by 1e9 for a compact scalar (`macs_g`); not the same as hardware FLOP counters.
        """
        model.eval()
        model.to(self.device)
        dummy = torch.randn(input_shape).to(self.device)
        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return float(macs) / 1e9

    def _measure_latency(self, model: nn.Module, input_shape: tuple, num_runs: int = 100) -> float:
        """
        Mean wall time of one batch-1 forward in ms, averaged over `num_runs` after warmup.

        On CUDA, each timed iteration is bracketed with `torch.cuda.synchronize()` so the
        duration reflects completed work (not a stream of queued kernels ending in one sync).
        """
        model.eval()
        model.to(self.device)
        dummy = torch.randn(input_shape).to(self.device)
        is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()

        with torch.no_grad():
            for _ in range(10):
                model(dummy)
        if is_cuda:
            torch.cuda.synchronize()

        per_run_ms: list[float] = []
        with torch.no_grad():
            for _ in range(num_runs):
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy)
                if is_cuda:
                    torch.cuda.synchronize()
                per_run_ms.append((time.perf_counter() - t0) * 1000.0)
        return sum(per_run_ms) / float(len(per_run_ms))

    def _evaluate_on_dataset(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        *,
        record_cuda_peak: bool,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        One batched pass over the test loader for accuracy / throughput.

        throughput_fps = total_samples / sum(per-batch wall time), where each interval
        (after a leading device sync) includes host→device copies of the batch tensors
        plus the forward pass (then sync again on CUDA). Excludes dataloader dequeue
        before the timed region and CPU-side metric bookkeeping after the forward.

        Returns peak_inference_memory_mb as max CUDA allocated during this loop only,
        or 0.0 if not record_cuda_peak.
        """
        model.eval()
        model.to(self.device)
        is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        if record_cuda_peak:
            torch.cuda.reset_peak_memory_stats()

        all_targets, all_predictions = [], []
        total_inference_time = 0.0
        total_samples = 0

        t0 = time.perf_counter()
        with torch.no_grad():
            for inputs, targets in dataloader:
                if is_cuda:
                    torch.cuda.synchronize()
                t_batch = time.perf_counter()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                if is_cuda:
                    torch.cuda.synchronize()
                total_inference_time += time.perf_counter() - t_batch

                total_samples += targets.size(0)
                _, predicted = outputs.max(1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        test_eval_time_s = time.perf_counter() - t0

        throughput_fps = (
            total_samples / total_inference_time if total_inference_time > 0 else 0.0
        )
        accuracy = 100.0 * sum(p == t for p, t in zip(all_predictions, all_targets)) / max(1, total_samples)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average="macro", zero_division=0
        )

        peak_mb = 0.0
        if record_cuda_peak:
            peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        return throughput_fps, accuracy, precision * 100, recall * 100, f1 * 100, test_eval_time_s, peak_mb
