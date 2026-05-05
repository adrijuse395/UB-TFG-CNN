import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import partial_tucker
from typing import Union, List
from .base import BaseDecomposedLayer

tl.set_backend('pytorch')

class TuckerDecomposedLayer(BaseDecomposedLayer):
    """
    Implements Tucker-2 Decomposition specifically for Conv2d layers.
    It decomposes the channel dimensions (modes 0 and 1) while leaving spatial dimensions intact.
    For Linear layers, it falls back to Truncated SVD.
    """

    def compress(self, layer: Union[nn.Conv2d, nn.Linear], **kwargs):
        rank = kwargs.get("rank")
        if rank is None:
            raise ValueError("Tucker decomposition requires a 'rank' parameter (e.g., [16, 16] or 16).")
            
        if isinstance(layer, nn.Conv2d):
            self._compress_conv2d(layer, rank)
        elif isinstance(layer, nn.Linear):
            self._compress_linear(layer, rank)
        else:
            raise ValueError(f"Tucker decomposition not supported for {type(layer)}")

    def _compress_conv2d(self, layer: nn.Conv2d, rank: Union[int, List[int]]):
        """
        Decomposes a Conv2d layer into a sequential block:
        Conv2d(1x1) -> Conv2d(kxk) -> Conv2d(1x1)
        """
        # Parse rank
        if isinstance(rank, int):
            # Same rank for both input and output channels
            ranks = [rank, rank]
        else:
            ranks = rank
            
        # Ensure ranks don't exceed actual dimensions
        ranks[0] = min(ranks[0], layer.out_channels) # R_out
        ranks[1] = min(ranks[1], layer.in_channels)  # R_in

        # Extract weights
        W = layer.weight.data
        
        from tensorly.decomposition import tucker
        
        # We perform full tucker, but setting spatial ranks to full size
        ranks = [ranks[0], ranks[1], layer.kernel_size[0], layer.kernel_size[1]]
        core, factors = tucker(W, rank=ranks, init='svd')
        
        # Absorb the spatial factors back into the core to keep them uncompressed
        core = tl.tenalg.multi_mode_dot(core, [factors[2], factors[3]], modes=[2, 3])
        
        last_factor = factors[0]  # Shape: (out_channels, R_out)
        first_factor = factors[1] # Shape: (in_channels, R_in)

        # 1. Pointwise Convolution (Compress Input Channels)
        # Weight shape must be (R_in, in_channels, 1, 1)
        first_layer = nn.Conv2d(layer.in_channels, ranks[1], kernel_size=1, stride=1, padding=0, bias=False)
        first_layer.weight.data = torch.transpose(first_factor, 1, 0).unsqueeze(-1).unsqueeze(-1)

        # 2. Core Spatial Convolution
        # Weight shape must be (R_out, R_in, k_h, k_w)
        core_layer = nn.Conv2d(ranks[1], ranks[0], kernel_size=layer.kernel_size, 
                               stride=layer.stride, padding=layer.padding, bias=False)
        core_layer.weight.data = core

        # 3. Pointwise Convolution (Expand Output Channels)
        # Weight shape must be (out_channels, R_out, 1, 1)
        last_layer = nn.Conv2d(ranks[0], layer.out_channels, kernel_size=1, stride=1, padding=0, bias=layer.bias is not None)
        last_layer.weight.data = last_factor.unsqueeze(-1).unsqueeze(-1)
        
        if layer.bias is not None:
            last_layer.bias.data = layer.bias.data

        # Assemble the sequential block
        self.compressed_ops = nn.Sequential(first_layer, core_layer, last_layer)

    def _compress_linear(self, layer: nn.Linear, rank: int):
        """
        Decomposes a Linear layer using Truncated SVD (equivalent to Tucker on 2D).
        Linear(in, out) -> Linear(in, rank) -> Linear(rank, out)
        """
        if isinstance(rank, list):
            rank = rank[0]
            
        rank = min(rank, min(layer.in_features, layer.out_features))
        
        W = layer.weight.data
        U, S, V = torch.svd(W)
        
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        V_trunc = V[:, :rank]
        
        # W ~ U * S * V^T = (U * S) * V^T
        # First layer computes V^T * X
        first_layer = nn.Linear(layer.in_features, rank, bias=False)
        first_layer.weight.data = V_trunc.t()
        
        # Second layer computes (U * S) * (V^T * X)
        second_layer = nn.Linear(rank, layer.out_features, bias=layer.bias is not None)
        second_layer.weight.data = torch.mm(U_trunc, torch.diag(S_trunc))
        
        if layer.bias is not None:
            second_layer.bias.data = layer.bias.data
            
        self.compressed_ops = nn.Sequential(first_layer, second_layer)
