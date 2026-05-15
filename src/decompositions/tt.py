"""
TT decomposition module.

Layout (same contract as cp/tucker/cp_gradient):
  - TTDecomposedLayer(BaseDecomposedLayer)
  - compress() → _compress_conv2d | _compress_linear
  - TT-specific forward modules live at the bottom of this file (private helpers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
from tensorly.decomposition import tensor_train
from typing import Union, List
from .base import BaseDecomposedLayer

class TTDecomposedLayer(BaseDecomposedLayer):
    """
    Implements TT (Tensor Train) Decomposition.
    Since native PyTorch does not support TT-convolutions efficiently without custom kernels,
    we store the TT cores as learnable parameters and reconstruct the weight tensor 
    on the fly during the forward pass. This demonstrates parameter compression.
    """
    
    def compress(self, layer: Union[nn.Conv2d, nn.Linear], **kwargs):
        rank = kwargs.get("rank")
        if rank is None:
            raise ValueError("TT decomposition requires a 'rank' parameter.")
            
        if isinstance(layer, nn.Conv2d):
            self._compress_conv2d(layer, rank)
        elif isinstance(layer, nn.Linear):
            self._compress_linear(layer, rank)
        else:
            raise ValueError(f"TT decomposition not supported for {type(layer)}")

    def _compress_conv2d(self, layer: nn.Conv2d, rank: Union[int, List[int]]):
        # W shape: (out_channels, in_channels, k_h, k_w)
        W = layer.weight.data
        
        # TT ranks list must be length 5: [1, r1, r2, r3, 1]
        if isinstance(rank, int):
            ranks = [1, rank, rank, rank, 1]
        elif isinstance(rank, list) and len(rank) == 3:
            ranks = [1, rank[0], rank[1], rank[2], 1]
        else:
            ranks = rank
            
        factors = tensor_train(W, rank=ranks)
        
        # We need a custom module to handle the on-the-fly reconstruction
        self.compressed_ops = _TTConv2dModule(factors, layer)

    def _compress_linear(self, layer: nn.Linear, rank: int):
        W = layer.weight.data
        if isinstance(rank, int):
            ranks = [1, rank, 1]
        else:
            ranks = rank
            
        factors = tensor_train(W, rank=ranks)
        self.compressed_ops = _TTLinearModule(factors, layer)


# --- TT-specific forward modules (not decomposition entry points) -------------
class _TTConv2dModule(nn.Module):
    def __init__(self, factors, original_layer):
        super().__init__()
        # Register factors as parameters
        self.factors = nn.ParameterList([nn.Parameter(f) for f in factors])
        
        # Store original conv properties
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
        
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Reconstruct the weight tensor by multiplying TT cores
        # factors are:
        # 0: (1, out_channels, r1)
        # 1: (r1, in_channels, r2)
        # 2: (r2, k_h, r3)
        # 3: (r3, k_w, 1)
        import tensorly as tl
        
        core = self.factors[0]
        for i in range(1, len(self.factors)):
            # contraction: core(..., r_i) * factor_i(r_i, dim, r_{i+1}) -> (..., dim, r_{i+1})
            core = tl.tenalg.tensordot(core, self.factors[i], modes=([-1], [0]))
            
        # The reconstructed core shape is (1, out_channels, in_channels, k_h, k_w, 1)
        # We need to squeeze the outer dummy ranks
        W_rec = core.squeeze(0).squeeze(-1)
        
        return F.conv2d(x, W_rec, self.bias, self.stride, self.padding, self.dilation, self.groups)


class _TTLinearModule(nn.Module):
    def __init__(self, factors, original_layer):
        super().__init__()
        self.factors = nn.ParameterList([nn.Parameter(f) for f in factors])
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        import tensorly as tl
        core = self.factors[0]
        for i in range(1, len(self.factors)):
            core = tl.tenalg.tensordot(core, self.factors[i], modes=([-1], [0]))
        W_rec = core.squeeze(0).squeeze(-1)
        return F.linear(x, W_rec, self.bias)
