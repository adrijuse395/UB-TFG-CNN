"""
SVD decomposition module.

Layout:
  - SVDDecomposedLayer(BaseDecomposedLayer)
  - compress() → _compress_conv2d | _compress_linear
"""

import torch
import torch.nn as nn
from typing import Union
from .base import BaseDecomposedLayer


class SVDDecomposedLayer(BaseDecomposedLayer):
    """
    Implements Truncated SVD Decomposition for both Linear and Conv2d layers.
    For Conv2d, it unfolds the spatial dimensions into a 2D matrix (mode-0 unfolding),
    applies SVD, and replaces the layer with two sequential convolutions.
    This serves as a classical 2D baseline against multi-dimensional tensor methods.
    """

    def compress(self, layer: Union[nn.Conv2d, nn.Linear], **kwargs):
        rank = kwargs.get("rank")
        if rank is None:
            raise ValueError("SVD decomposition requires a 'rank' parameter (int).")
        
        if isinstance(rank, list):
            rank = rank[0]
            
        if isinstance(layer, nn.Conv2d):
            self._compress_conv2d(layer, int(rank))
        elif isinstance(layer, nn.Linear):
            self._compress_linear(layer, int(rank))
        else:
            raise ValueError(f"SVD decomposition not supported for {type(layer)}")

    def _compress_conv2d(self, layer: nn.Conv2d, rank: int):
        """
        Decomposes a Conv2d layer into a sequential block using standard 2D SVD:
        Conv2d(in, rank, kernel_size=(kh, kw)) -> Conv2d(rank, out, kernel_size=1)
        """
        W = layer.weight.data
        out_ch, in_ch, kh, kw = W.shape
        
        # Unfold to (out_ch, in_ch * kh * kw)
        W_mat = W.reshape(out_ch, -1)
        
        # SVD
        U, S, Vh = torch.linalg.svd(W_mat, full_matrices=False)
        
        # Enforce maximum possible rank given the matrix dimensions
        max_valid_rank = min(out_ch, in_ch * kh * kw)
        
        # Cap rank so the decomposed layer is actually smaller than the original.
        # SVD params = rank * (out_ch + in_ch*kh*kw); original = out_ch * in_ch*kh*kw
        # Compression holds when: rank < out_ch * in_ch*kh*kw / (out_ch + in_ch*kh*kw)
        denom = max(1, out_ch + in_ch * kh * kw)
        max_rank_compression = max(1, int((out_ch * in_ch * kh * kw) / denom))
        
        effective_rank = min(rank, max_valid_rank)
        if effective_rank > max_rank_compression:
            print(f"    [SVD] rank capped for compression: {effective_rank} -> {max_rank_compression}")
            effective_rank = max_rank_compression

        U_trunc = U[:, :effective_rank]
        S_trunc = S[:effective_rank]
        Vh_trunc = Vh[:effective_rank, :]
        
        # Calculate reconstruction error (Frobenius norm)
        W_mat_hat = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
        error = torch.norm(W_mat - W_mat_hat) / torch.norm(W_mat)
        self.reconstruction_error = error.item()
        
        # Layer 1: computes S * Vh
        # Vh_trunc has shape (rank, in_ch * kh * kw). We multiply by S and reshape to original spatial dims
        W1 = (S_trunc.unsqueeze(-1) * Vh_trunc).reshape(effective_rank, in_ch, kh, kw)
        layer1 = nn.Conv2d(
            in_ch, effective_rank, 
            kernel_size=layer.kernel_size, 
            stride=layer.stride, 
            padding=layer.padding, 
            dilation=layer.dilation,
            groups=layer.groups,
            bias=False
        )
        layer1.weight.data = W1

        # Layer 2: computes U
        # U_trunc has shape (out_ch, rank). We reshape to 1x1 conv weights
        W2 = U_trunc.reshape(out_ch, effective_rank, 1, 1)
        layer2 = nn.Conv2d(
            effective_rank, out_ch, 
            kernel_size=1, stride=1, padding=0, 
            bias=layer.bias is not None
        )
        layer2.weight.data = W2

        if layer.bias is not None:
            layer2.bias.data = layer.bias.data

        # Assemble the sequential block
        self.compressed_ops = nn.Sequential(layer1, layer2)

    def _compress_linear(self, layer: nn.Linear, rank: int):
        """
        Decomposes a Linear layer using Truncated SVD.
        Linear(in, out) -> Linear(in, rank) -> Linear(rank, out)
        """
        rank = min(rank, min(layer.in_features, layer.out_features))
        
        W = layer.weight.data
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vh_trunc = Vh[:rank, :]
        
        # Calculate reconstruction error (Frobenius norm)
        W_hat = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
        error = torch.norm(W - W_hat) / torch.norm(W)
        self.reconstruction_error = error.item()
        
        first_layer = nn.Linear(layer.in_features, rank, bias=False)
        first_layer.weight.data = Vh_trunc
        
        second_layer = nn.Linear(rank, layer.out_features, bias=layer.bias is not None)
        second_layer.weight.data = U_trunc * S_trunc.unsqueeze(0)
        
        if layer.bias is not None:
            second_layer.bias.data = layer.bias.data
            
        self.compressed_ops = nn.Sequential(first_layer, second_layer)
