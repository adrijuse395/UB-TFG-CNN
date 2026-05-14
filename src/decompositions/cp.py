"""
CP (Parafac) decomposition module.

Layout (same contract as other methods):
  - CPDecomposedLayer(BaseDecomposedLayer)
  - compress() → _compress_conv2d | _compress_linear
"""

import gc
import time
import torch
import torch.nn as nn
from tensorly.decomposition import parafac
from typing import Union, List
from .base import BaseDecomposedLayer


class CPDecomposedLayer(BaseDecomposedLayer):
    """
    Implements CP (Canonical Polyadic) Decomposition.
    For Conv2d: Replaces with a sequence of 4 convolutions (Pointwise -> Depthwise Vertical -> Depthwise Horizontal -> Pointwise).
    For Linear: Equivalent to Truncated SVD (rank-R approximation).
    """

    def compress(self, layer: Union[nn.Conv2d, nn.Linear], **kwargs):
        rank = kwargs.get("rank")
        if rank is None:
            raise ValueError("CP decomposition requires a 'rank' parameter (int).")

        # Ensure rank is an integer for CP
        if isinstance(rank, list):
            rank = rank[0]

        parafac_n_iter_max = int(kwargs.get("parafac_n_iter_max", 60))
        parafac_tol = float(kwargs.get("parafac_tol", 1e-5))
        cp_parafac_on_cpu = bool(kwargs.get("cp_parafac_on_cpu", True))
        cp_layer_timeout_s = float(kwargs.get("cp_layer_timeout_s", 20.0))
        cp_abort_if_mem_available_mb_below = int(
            kwargs.get("cp_abort_if_mem_available_mb_below", 800)
        )
        cp_init = str(kwargs.get("cp_init", "random")).lower()
        if cp_init not in {"svd", "random"}:
            cp_init = "random"
        cp_normalize_factors = bool(kwargs.get("cp_normalize_factors", True))
        if isinstance(layer, nn.Conv2d):
            self._compress_conv2d(
                layer,
                rank,
                parafac_n_iter_max=parafac_n_iter_max,
                parafac_tol=parafac_tol,
                cp_parafac_on_cpu=cp_parafac_on_cpu,
                cp_layer_timeout_s=cp_layer_timeout_s,
                cp_abort_if_mem_available_mb_below=cp_abort_if_mem_available_mb_below,
                cp_init=cp_init,
                cp_normalize_factors=cp_normalize_factors,
            )
        elif isinstance(layer, nn.Linear):
            self._compress_linear(layer, rank)
        else:
            raise ValueError(f"CP decomposition not supported for {type(layer)}")

    def _compress_conv2d(
        self,
        layer: nn.Conv2d,
        rank: int,
        *,
        parafac_n_iter_max: int,
        parafac_tol: float,
        cp_parafac_on_cpu: bool,
        cp_layer_timeout_s: float,
        cp_abort_if_mem_available_mb_below: int,
        cp_init: str,
        cp_normalize_factors: bool,
    ):
        target_device = layer.weight.device
        target_dtype = layer.weight.dtype

        if cp_parafac_on_cpu:
            W = layer.weight.data.detach().cpu().float().contiguous()
        else:
            W = layer.weight.data.detach().float().contiguous()

        in_ch = int(layer.in_channels)
        out_ch = int(layer.out_channels)
        kh = int(layer.kernel_size[0])
        kw = int(layer.kernel_size[1])

        # Guard rail 1: CP factors have shapes (out_ch, R) and (in_ch, R) → need R <= min(in_ch, out_ch).
        rank_requested = max(1, int(rank))
        rank = min(rank_requested, in_ch, out_ch)
        if rank < rank_requested:
            print(
                f"    [CP] rank capped by channel dimensions: {rank_requested} -> {rank} "
                f"(layer {in_ch}->{out_ch}, k={kh}x{kw}; CP rank cannot exceed min(in_channels, out_channels))"
            )

        # Guard rail 2: enforce rank where CP still yields parameter compression.
        denom = max(1, in_ch + out_ch + kh + kw)
        max_rank_compression = max(1, int((in_ch * out_ch * kh * kw) / denom))
        if rank > max_rank_compression:
            print(
                f"    [CP] rank capped for compression: {rank} -> {max_rank_compression} "
                f"(layer {in_ch}->{out_ch}, k={kh}x{kw})"
            )
            rank = max_rank_compression

        mem_guard_triggered = {"value": False}
        timeout_triggered = {"value": False}
        t_start = time.perf_counter()

        def _mem_available_mb() -> int:
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            kb = int(line.split()[1])
                            return kb // 1024
            except Exception:
                return 10**9
            return 10**9

        # Pre-flight RAM guard: callback is per-iteration and may trigger too late.
        # This intentionally over-estimates ALS intermediates to fail fast on risky ranks.
        bytes_per_elem = 4  # float32
        tensor_bytes = int(layer.weight.numel()) * bytes_per_elem
        estimated_peak_mb = (tensor_bytes * max(8, min(20, rank // 4))) / (1024 * 1024)
        mem_avail_mb = _mem_available_mb()
        if mem_avail_mb < (cp_abort_if_mem_available_mb_below + int(estimated_peak_mb)):
            raise RuntimeError(
                "CP pre-flight memory guard: estimated ALS peak too high for current RAM "
                f"(avail={mem_avail_mb} MB, est_peak={estimated_peak_mb:.0f} MB, "
                f"floor={cp_abort_if_mem_available_mb_below} MB)."
            )

        # Prefer SVD init in heavy layers/ranks for stabler and usually shorter ALS runs.
        if cp_init == "random" and (in_ch >= 256 or out_ch >= 256 or rank >= 32):
            cp_init = "svd"
            print(
                f"    [CP] switching init to 'svd' for stability "
                f"(layer {in_ch}->{out_ch}, rank={rank})."
            )

        def _cp_callback(_cp_tensor, _rec_error):
            elapsed = time.perf_counter() - t_start
            if elapsed > cp_layer_timeout_s:
                timeout_triggered["value"] = True
                return True
            if _mem_available_mb() < cp_abort_if_mem_available_mb_below:
                mem_guard_triggered["value"] = True
                return True
            return False

        try:
            cp_weights, factors = parafac(
                W,
                rank=rank,
                init='random',
                n_iter_max=parafac_n_iter_max,
                tol=parafac_tol,
                normalize_factors=cp_normalize_factors,
                callback=_cp_callback,
            )
        finally:
            del W
            gc.collect()

        if timeout_triggered["value"]:
            raise RuntimeError(
                f"CP guard timeout reached ({cp_layer_timeout_s:.1f}s) for layer "
                f"{layer.in_channels}->{layer.out_channels}."
            )
        if mem_guard_triggered["value"]:
            raise RuntimeError(
                "CP memory guard triggered: available RAM dropped below "
                f"{cp_abort_if_mem_available_mb_below} MB."
            )

        # Factors:
        # f0: (out_channels, rank)
        # f1: (in_channels, rank)
        # f2: (kernel_h, rank)
        # f3: (kernel_w, rank)
        f_out, f_in, f_h, f_w = factors
        cw = cp_weights

        f_out = f_out.to(device=target_device, dtype=target_dtype)
        f_in = f_in.to(device=target_device, dtype=target_dtype)
        f_h = f_h.to(device=target_device, dtype=target_dtype)
        f_w = f_w.to(device=target_device, dtype=target_dtype)
        cw = cw.to(device=target_device, dtype=target_dtype)

        # 1. Pointwise Convolution (Compress Input Channels)
        # Weight shape: (rank, in_channels, 1, 1)
        layer1 = nn.Conv2d(layer.in_channels, rank, kernel_size=1, stride=1, padding=0, bias=False)
        layer1.weight.data = f_in.t().unsqueeze(-1).unsqueeze(-1)

        # 2. Depthwise Vertical Spatial Convolution
        # Weight shape: (rank, 1, kernel_h, 1)
        layer2 = nn.Conv2d(rank, rank, kernel_size=(layer.kernel_size[0], 1),
                           stride=(layer.stride[0], 1), padding=(layer.padding[0], 0),
                           groups=rank, bias=False)
        layer2.weight.data = f_h.t().unsqueeze(1).unsqueeze(-1)

        # 3. Depthwise Horizontal Spatial Convolution
        # Weight shape: (rank, 1, 1, kernel_w)
        layer3 = nn.Conv2d(rank, rank, kernel_size=(1, layer.kernel_size[1]),
                           stride=(1, layer.stride[1]), padding=(0, layer.padding[1]),
                           groups=rank, bias=False)
        layer3.weight.data = f_w.t().unsqueeze(1).unsqueeze(2)

        # 4. Pointwise Convolution (Expand Output Channels)
        # Include CP weights in this final layer
        # Weight shape: (out_channels, rank, 1, 1)
        layer4 = nn.Conv2d(rank, layer.out_channels, kernel_size=1, stride=1, padding=0, bias=layer.bias is not None)
        f_out_weighted = f_out * cw.unsqueeze(0)
        layer4.weight.data = f_out_weighted.unsqueeze(-1).unsqueeze(-1)

        if layer.bias is not None:
            layer4.bias.data = layer.bias.data

        self.compressed_ops = nn.Sequential(layer1, layer2, layer3, layer4)

        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    def _compress_linear(self, layer: nn.Linear, rank: int):
        """
        CP decomposition of a 2D tensor is basically SVD.
        """
        rank = min(rank, min(layer.in_features, layer.out_features))

        W = layer.weight.data
        U, S, V = torch.svd(W)

        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        V_trunc = V[:, :rank]

        first_layer = nn.Linear(layer.in_features, rank, bias=False)
        first_layer.weight.data = V_trunc.t()

        second_layer = nn.Linear(rank, layer.out_features, bias=layer.bias is not None)
        second_layer.weight.data = torch.mm(U_trunc, torch.diag(S_trunc))

        if layer.bias is not None:
            second_layer.bias.data = layer.bias.data

        self.compressed_ops = nn.Sequential(first_layer, second_layer)
