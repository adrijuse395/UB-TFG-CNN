import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from .base import BaseDecomposedLayer


class CPGradientDecomposedLayer(BaseDecomposedLayer):
    """
    CP-style factorization of Conv2d weights by minimizing Frobenius MSE with Adam.
    Avoids TensorLy parafac / ALS (no Khatri–Rao / MTTKRP spikes).

    Reconstruction: W_hat[o,i,h,w] = sum_r f_out[o,r]*f_in[i,r]*f_h[h,r]*f_w[w,r]
    (weights λ_r are absorbed into factors — equivalent parametrization).

    Same stacked Conv2d layout as CPDecomposedLayer after factors are fitted.
    Linear layers use truncated SVD (same as standard CP path).
    """

    def compress(self, layer: Union[nn.Conv2d, nn.Linear], **kwargs):
        rank = kwargs.get("rank")
        if rank is None:
            raise ValueError("CP_GD requires a 'rank' parameter (int).")
        if isinstance(rank, list):
            rank = rank[0]

        if isinstance(layer, nn.Conv2d):
            self._compress_conv2d(
                layer,
                int(rank),
                cp_gd_steps=int(kwargs.get("cp_gd_steps", 3000)),
                cp_gd_lr=float(kwargs.get("cp_gd_lr", 0.01)),
                cp_gd_on_cpu=bool(kwargs.get("cp_gd_on_cpu", True)),
                cp_gd_timeout_s=float(kwargs.get("cp_gd_timeout_s", 180.0)),
                cp_abort_if_mem_available_mb_below=int(
                    kwargs.get("cp_abort_if_mem_available_mb_below", 800)
                ),
                cp_gd_init=str(kwargs.get("cp_gd_init", "svd")).lower(),
                cp_gd_grad_clip=float(kwargs.get("cp_gd_grad_clip", 0.0)),
                cp_gd_scheduler_patience=int(
                    kwargs.get("cp_gd_scheduler_patience", 200)
                ),
            )
        elif isinstance(layer, nn.Linear):
            self._compress_linear(layer, int(rank))
        else:
            raise ValueError(f"CP_GD not supported for {type(layer)}")

    def _mem_available_mb(self) -> int:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb // 1024
        except Exception:
            pass
        return 10**9

    @staticmethod
    def _expand_factor_cols(mat: torch.Tensor, rank: int) -> torch.Tensor:
        """
        Ensure factor second dimension equals `rank`. Mode-n SVD may yield fewer
        than `rank` columns when the unfolding has a small leading dimension
        (e.g. kh=3 → at most 3 singular vectors for the height mode).
        Tile columns as a cheap fix so einsum shapes stay consistent.
        """
        if mat.shape[-1] == rank:
            return mat
        if mat.shape[-1] > rank:
            return mat[..., :rank]
        r0 = mat.shape[-1]
        if r0 == 0:
            raise RuntimeError("CP_GD: empty factor column dimension.")
        repeats = (rank + r0 - 1) // r0
        tiled = mat.repeat(1, repeats)[:, :rank]
        return tiled

    def _rank_caps_conv2d(self, layer: nn.Conv2d, rank: int) -> int:
        in_ch = int(layer.in_channels)
        out_ch = int(layer.out_channels)
        kh = int(layer.kernel_size[0])
        kw = int(layer.kernel_size[1])
        rank = max(1, min(int(rank), in_ch, out_ch))
        denom = max(1, in_ch + out_ch + kh + kw)
        max_rank_compression = max(1, int((in_ch * out_ch * kh * kw) / denom))
        if rank > max_rank_compression:
            print(
                f"    [CP_GD] rank capped for compression: {rank} -> {max_rank_compression} "
                f"(layer {in_ch}->{out_ch}, k={kh}x{kw})"
            )
            rank = max_rank_compression
        return rank

    def _init_factors_svd_modes(self, W: torch.Tensor, rank: int):
        """Per-mode left singular vectors on normalized W (cheap CP warm-start)."""
        out_ch, in_ch, kh, kw = W.shape
        dev = W.device

        m0 = W.reshape(out_ch, -1)
        U, _, _ = torch.linalg.svd(m0, full_matrices=False)
        n0 = min(U.shape[1], rank)
        f_out = self._expand_factor_cols(U[:, :n0], rank).to(dev)

        m1 = W.permute(1, 0, 2, 3).reshape(in_ch, -1)
        U, _, _ = torch.linalg.svd(m1, full_matrices=False)
        n1 = min(U.shape[1], rank)
        f_in = self._expand_factor_cols(U[:, :n1], rank).to(dev)

        m2 = W.permute(2, 0, 1, 3).reshape(kh, -1)
        U, _, _ = torch.linalg.svd(m2, full_matrices=False)
        n2 = min(U.shape[1], rank)
        f_h = self._expand_factor_cols(U[:, :n2], rank).to(dev)

        m3 = W.permute(3, 0, 1, 2).reshape(kw, -1)
        U, _, _ = torch.linalg.svd(m3, full_matrices=False)
        n3 = min(U.shape[1], rank)
        f_w = self._expand_factor_cols(U[:, :n3], rank).to(dev)

        return f_out, f_in, f_h, f_w

    @staticmethod
    def _scale_f_out_to_match_W_opt_norm(
        f_out: nn.Parameter,
        f_in: nn.Parameter,
        f_h: nn.Parameter,
        f_w: nn.Parameter,
        W_opt: torch.Tensor,
    ) -> None:
        """Absorb scalar into f_out so ||Σ_r f_out f_in f_h f_w||_F matches ||W_opt||_F."""
        with torch.no_grad():
            W_hat0 = torch.einsum("or,ir,hr,wr->oihw", f_out, f_in, f_h, f_w)
            scale = W_opt.norm() / W_hat0.norm().clamp_min(1e-12)
            f_out.mul_(scale)

    def _compress_conv2d(
        self,
        layer: nn.Conv2d,
        rank: int,
        *,
        cp_gd_steps: int,
        cp_gd_lr: float,
        cp_gd_on_cpu: bool,
        cp_gd_timeout_s: float,
        cp_abort_if_mem_available_mb_below: int,
        cp_gd_init: str,
        cp_gd_grad_clip: float,
        cp_gd_scheduler_patience: int,
    ):
        target_device = layer.weight.device
        target_dtype = layer.weight.dtype

        rank = self._rank_caps_conv2d(layer, rank)
        if cp_gd_init not in {"random", "svd"}:
            cp_gd_init = "random"

        work_dev = torch.device("cpu") if cp_gd_on_cpu else target_device
        W = layer.weight.data.detach().to(device=work_dev, dtype=torch.float32).contiguous()
        w_norm = W.norm().clamp_min(1e-12)
        W_opt = W / w_norm

        out_ch, in_ch, kh, kw = int(layer.out_channels), int(layer.in_channels), int(
            layer.kernel_size[0]
        ), int(layer.kernel_size[1])

        if cp_gd_init == "svd":
            f_o, f_i, f_kh, f_kw = self._init_factors_svd_modes(W_opt, rank)
            f_out = nn.Parameter(f_o.clone())
            f_in = nn.Parameter(f_i.clone())
            f_h = nn.Parameter(f_kh.clone())
            f_w = nn.Parameter(f_kw.clone())
        else:
            # Scale random factors so rank-1 outer products are not ~1e-6 at step 0:
            # entries of W_hat scale roughly as product of four factor scales; match
            # std(W_opt) with init_std ≈ target_std^(1/4) / rank^(1/8)-style scaling.
            target_std = float(W_opt.std().clamp_min(1e-8))
            init_std = (target_std / (rank ** 0.5)) ** 0.25
            f_out = nn.Parameter(torch.randn(out_ch, rank, device=work_dev) * init_std)
            f_in = nn.Parameter(torch.randn(in_ch, rank, device=work_dev) * init_std)
            f_h = nn.Parameter(torch.randn(kh, rank, device=work_dev) * init_std)
            f_w = nn.Parameter(torch.randn(kw, rank, device=work_dev) * init_std)

        self._scale_f_out_to_match_W_opt_norm(f_out, f_in, f_h, f_w, W_opt)

        clip = cp_gd_grad_clip if cp_gd_grad_clip > 0 else 10.0

        params = [f_out, f_in, f_h, f_w]
        opt = torch.optim.Adam(params, lr=cp_gd_lr)
        sched_patience = max(
            20, min(cp_gd_scheduler_patience, max(cp_gd_steps // 4, 50))
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=sched_patience,
            min_lr=1e-5,
        )

        t_start = time.perf_counter()

        for step in range(cp_gd_steps):
            if time.perf_counter() - t_start > cp_gd_timeout_s:
                print(f"    [CP_GD] timeout after {step} steps ({cp_gd_timeout_s:.1f}s)")
                break
            if step % 64 == 0 and self._mem_available_mb() < cp_abort_if_mem_available_mb_below:
                raise RuntimeError(
                    "CP_GD memory guard: available RAM dropped below "
                    f"{cp_abort_if_mem_available_mb_below} MB."
                )

            opt.zero_grad(set_to_none=True)
            W_hat = torch.einsum("or,ir,hr,wr->oihw", f_out, f_in, f_h, f_w)
            loss = F.mse_loss(W_hat, W_opt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            scheduler.step(float(loss.detach()))

        with torch.no_grad():
            W_hat = torch.einsum("or,ir,hr,wr->oihw", f_out, f_in, f_h, f_w)
            rel = (W_hat - W_opt).norm() / W_opt.norm().clamp_min(1e-12)
            print(f"    [CP_GD] relative Frobenius error ~ {float(rel):.4e}")

        # Fold global Frobenius scale into output-side factor (linear in f_out).
        f_o = (f_out.detach() * w_norm).to(device=target_device, dtype=target_dtype)
        f_i = f_in.detach().to(device=target_device, dtype=target_dtype)
        f_hi = f_h.detach().to(device=target_device, dtype=target_dtype)
        f_wi = f_w.detach().to(device=target_device, dtype=target_dtype)

        layer1 = nn.Conv2d(
            layer.in_channels, rank, kernel_size=1, stride=1, padding=0, bias=False
        )
        layer1.weight.data = f_i.t().unsqueeze(-1).unsqueeze(-1)

        layer2 = nn.Conv2d(
            rank,
            rank,
            kernel_size=(layer.kernel_size[0], 1),
            stride=(layer.stride[0], 1),
            padding=(layer.padding[0], 0),
            groups=rank,
            bias=False,
        )
        layer2.weight.data = f_hi.t().unsqueeze(1).unsqueeze(-1)

        layer3 = nn.Conv2d(
            rank,
            rank,
            kernel_size=(1, layer.kernel_size[1]),
            stride=(1, layer.stride[1]),
            padding=(0, layer.padding[1]),
            groups=rank,
            bias=False,
        )
        layer3.weight.data = f_wi.t().unsqueeze(1).unsqueeze(2)

        layer4 = nn.Conv2d(
            rank,
            layer.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=layer.bias is not None,
        )
        layer4.weight.data = f_o.unsqueeze(-1).unsqueeze(-1)

        if layer.bias is not None:
            layer4.bias.data = layer.bias.data

        self.compressed_ops = nn.Sequential(layer1, layer2, layer3, layer4)

        del W, W_opt, W_hat, opt, scheduler
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    def _compress_linear(self, layer: nn.Linear, rank: int):
        rank = min(rank, min(layer.in_features, layer.out_features))
        W = layer.weight.data
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vh_trunc = Vh[:rank, :]

        first_layer = nn.Linear(layer.in_features, rank, bias=False)
        first_layer.weight.data = Vh_trunc

        second_layer = nn.Linear(rank, layer.out_features, bias=layer.bias is not None)
        second_layer.weight.data = U_trunc * S_trunc.unsqueeze(0)

        if layer.bias is not None:
            second_layer.bias.data = layer.bias.data

        self.compressed_ops = nn.Sequential(first_layer, second_layer)
