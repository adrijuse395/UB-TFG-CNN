"""
CP (Canonical Polyadic) decomposition for Conv2d / Linear.

Conv2d: CP-ALS in PyTorch (alternating least squares). Each mode update uses the
matricized tensor times Khatri-Rao via a single einsum (no dense J×R Khatri-Rao) and
K^T K as the Hadamard product of Gram matrices (R×R), matching the same CP-ALS family
as TensorLy's parafac but avoiding that memory path. Linear: truncated SVD.
"""

from __future__ import annotations

import gc
import time
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base import BaseDecomposedLayer


# --- CP-ALS helpers (4-way tensor W: out × in × kh × kw) ---------------------------------


def _cp_als_reconstruction_error(W: torch.Tensor, factors: List[torch.Tensor]) -> float:
    A0, A1, A2, A3 = factors
    W_hat = torch.einsum("or,ir,hr,wr->oihw", A0, A1, A2, A3)
    return float((W_hat - W).norm() / W.norm().clamp_min(1e-12))


def _cp_als_mttkrp(W: torch.Tensor, factors: List[torch.Tensor], skip: int) -> torch.Tensor:
    if skip == 0:
        _, A1, A2, A3 = factors
        return torch.einsum("oihw, ir, hr, wr -> or", W, A1, A2, A3)
    if skip == 1:
        A0, _, A2, A3 = factors
        return torch.einsum("oihw, or, hr, wr -> ir", W, A0, A2, A3)
    if skip == 2:
        A0, A1, _, A3 = factors
        return torch.einsum("oihw, or, ir, wr -> hr", W, A0, A1, A3)
    if skip == 3:
        A0, A1, A2, _ = factors
        return torch.einsum("oihw, or, ir, hr -> wr", W, A0, A1, A2)
    raise ValueError(f"skip must be 0..3, got {skip}")


def _cp_als_khatri_rao_gram(factors: List[torch.Tensor], skip: int, *, eps: float = 1e-8) -> torch.Tensor:
    R = factors[0].shape[1]
    dev, dt = factors[0].device, factors[0].dtype
    G = torch.ones(R, R, device=dev, dtype=dt)
    for j, Aj in enumerate(factors):
        if j == skip:
            continue
        G = G * (Aj.T @ Aj)
    return G + eps * torch.eye(R, device=dev, dtype=dt)


def _cp_als_lstsq_a_mk(MK: torch.Tensor, KtK: torch.Tensor) -> torch.Tensor:
    sol = torch.linalg.lstsq(KtK, MK.T, rcond=1e-10)
    return sol.solution.T


def _cp_als_random_init(
    out: int, inn: int, kh: int, kw: int, rank: int, dev: torch.device, dt: torch.dtype
) -> List[torch.Tensor]:
    s = 0.02
    return [
        torch.randn(out, rank, device=dev, dtype=dt) * s,
        torch.randn(inn, rank, device=dev, dtype=dt) * s,
        torch.randn(kh, rank, device=dev, dtype=dt) * s,
        torch.randn(kw, rank, device=dev, dtype=dt) * s,
    ]


def _cp_als_svd_init(W: torch.Tensor, rank: int) -> List[torch.Tensor]:
    out, inn, kh, kw = W.shape
    dev, dt = W.device, W.dtype

    def pad_cols(M: torch.Tensor) -> torch.Tensor:
        r0 = M.shape[1]
        if r0 >= rank:
            return M[:, :rank]
        z = torch.randn(M.shape[0], rank - r0, device=dev, dtype=dt) * 0.02
        return torch.cat([M, z], dim=1)

    m0 = W.reshape(out, -1)
    U, _, _ = torch.linalg.svd(m0, full_matrices=False)
    A0 = pad_cols(U[:, : min(rank, U.shape[1])])

    m1 = W.permute(1, 0, 2, 3).reshape(inn, -1)
    U, _, _ = torch.linalg.svd(m1, full_matrices=False)
    A1 = pad_cols(U[:, : min(rank, U.shape[1])])

    m2 = W.permute(2, 0, 1, 3).reshape(kh, -1)
    U, _, _ = torch.linalg.svd(m2, full_matrices=False)
    A2 = pad_cols(U[:, : min(rank, U.shape[1])])

    m3 = W.permute(3, 0, 1, 2).reshape(kw, -1)
    U, _, _ = torch.linalg.svd(m3, full_matrices=False)
    A3 = pad_cols(U[:, : min(rank, U.shape[1])])

    return [A0, A1, A2, A3]


def _cp_als_4way(
    W: torch.Tensor,
    rank: int,
    *,
    n_iter_max: int,
    tol: float,
    init: str,
    callback: Optional[Callable[[], bool]],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Returns (weights all-ones length R, factor list)."""
    if W.ndim != 4:
        raise ValueError("_cp_als_4way expects a 4D tensor.")
    out, inn, kh, kw = W.shape
    dev, dt = W.device, W.dtype
    R = int(rank)

    init_l = (init or "random").lower()
    if init_l == "svd":
        factors = _cp_als_svd_init(W, R)
    else:
        factors = _cp_als_random_init(out, inn, kh, kw, R, dev, dt)

    lam = torch.ones(R, device=dev, dtype=dt)
    prev_err: Optional[float] = None

    for _it in range(int(n_iter_max)):
        for mode in range(4):
            if callback is not None and callback():
                return lam, factors
            KtK = _cp_als_khatri_rao_gram(factors, mode)
            MK = _cp_als_mttkrp(W, factors, mode)
            factors[mode] = _cp_als_lstsq_a_mk(MK, KtK)

        err = _cp_als_reconstruction_error(W, factors)
        if prev_err is not None and abs(prev_err - err) < tol * max(1.0, prev_err):
            break
        prev_err = err

    return lam, factors


class CPDecomposedLayer(BaseDecomposedLayer):
    """
    CP decomposition.
    For Conv2d: CP-ALS (low-memory) then the usual 4-conv stack.
    For Linear: truncated SVD.
    """

    def compress(self, layer: Union[nn.Conv2d, nn.Linear], **kwargs):
        rank = kwargs.get("rank")
        if rank is None:
            raise ValueError("CP decomposition requires a 'rank' parameter (int).")

        if isinstance(rank, list):
            rank = rank[0]

        parafac_n_iter_max = int(kwargs.get("parafac_n_iter_max", 60))
        parafac_tol = float(kwargs.get("parafac_tol", 1e-5))
        cp_parafac_on_cpu = bool(kwargs.get("cp_parafac_on_cpu", True))
        cp_layer_timeout_s = float(kwargs.get("cp_layer_timeout_s", 300.0))
        cp_abort_if_mem_available_mb_below = int(
            kwargs.get("cp_abort_if_mem_available_mb_below", 800)
        )
        cp_init = str(kwargs.get("cp_init", "svd")).lower()
        if cp_init not in {"svd", "random"}:
            cp_init = "svd"
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

        rank = max(1, int(rank))

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

        bytes_per_elem = 4
        tensor_bytes = int(layer.weight.numel()) * bytes_per_elem
        estimated_peak_mb = (tensor_bytes * max(8, min(20, rank // 4))) / (1024 * 1024)
        mem_avail_mb = _mem_available_mb()
        if mem_avail_mb < (cp_abort_if_mem_available_mb_below + int(estimated_peak_mb)):
            raise RuntimeError(
                "CP pre-flight memory guard: estimated ALS peak too high for current RAM "
                f"(avail={mem_avail_mb} MB, est_peak={estimated_peak_mb:.0f} MB, "
                f"floor={cp_abort_if_mem_available_mb_below} MB)."
            )

        if cp_init == "random" and (in_ch >= 256 or out_ch >= 256 or rank >= 32):
            cp_init = "svd"
            print(
                f"    [CP] switching init to 'svd' for stability "
                f"(layer {in_ch}->{out_ch}, rank={rank})."
            )

        def _als_guard_cb() -> bool:
            if time.perf_counter() - t_start > cp_layer_timeout_s:
                timeout_triggered["value"] = True
                return True
            if _mem_available_mb() < cp_abort_if_mem_available_mb_below:
                mem_guard_triggered["value"] = True
                return True
            return False

        cw, factors = _cp_als_4way(
            W,
            rank,
            n_iter_max=parafac_n_iter_max,
            tol=parafac_tol,
            init=cp_init,
            callback=_als_guard_cb,
        )
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

        f_out, f_in, f_h, f_w = factors

        f_out = f_out.to(device=target_device, dtype=target_dtype)
        f_in = f_in.to(device=target_device, dtype=target_dtype)
        f_h = f_h.to(device=target_device, dtype=target_dtype)
        f_w = f_w.to(device=target_device, dtype=target_dtype)
        cw = cw.to(device=target_device, dtype=target_dtype)

        layer1 = nn.Conv2d(layer.in_channels, rank, kernel_size=1, stride=1, padding=0, bias=False)
        layer1.weight.data = f_in.t().unsqueeze(-1).unsqueeze(-1)

        layer2 = nn.Conv2d(rank, rank, kernel_size=(layer.kernel_size[0], 1),
                           stride=(layer.stride[0], 1), padding=(layer.padding[0], 0),
                           groups=rank, bias=False)
        layer2.weight.data = f_h.t().unsqueeze(1).unsqueeze(-1)

        layer3 = nn.Conv2d(rank, rank, kernel_size=(1, layer.kernel_size[1]),
                           stride=(1, layer.stride[1]), padding=(0, layer.padding[1]),
                           groups=rank, bias=False)
        layer3.weight.data = f_w.t().unsqueeze(1).unsqueeze(2)

        layer4 = nn.Conv2d(rank, layer.out_channels, kernel_size=1, stride=1, padding=0, bias=layer.bias is not None)
        f_out_weighted = f_out * cw.unsqueeze(0)
        layer4.weight.data = f_out_weighted.unsqueeze(-1).unsqueeze(-1)

        if layer.bias is not None:
            layer4.bias.data = layer.bias.data

        self.compressed_ops = nn.Sequential(layer1, layer2, layer3, layer4)

        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    def _compress_linear(self, layer: nn.Linear, rank: int):
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
