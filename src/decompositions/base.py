import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Union

class BaseDecomposedLayer(nn.Module, ABC):
    """
    Abstract base class for tensor-compressed layers.
    Can be used to compress both Conv2d and Linear layers.
    """
    def __init__(self):
        super(BaseDecomposedLayer, self).__init__()
        self.compressed_ops = None
        self.original_type = None

    @abstractmethod
    def compress(self, layer: Union[nn.Conv2d, nn.Linear], **kwargs):
        """
        Extracts the weight tensor from the original layer, applies tensor decomposition,
        and populates `self.compressed_ops` with the resulting lighter operations.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass. Delegates to the compressed sequential operations.
        """
        if self.compressed_ops is None:
            raise RuntimeError("The layer has not been compressed yet. Call compress() first.")
        return self.compressed_ops(x)

    @classmethod
    def from_layer(cls, layer: Union[nn.Conv2d, nn.Linear], **kwargs) -> 'BaseDecomposedLayer':
        """
        Factory method to instantiate a decomposed layer and apply compression in one step.
        """
        instance = cls()
        instance.original_type = type(layer)
        instance.compress(layer, **kwargs)
        return instance
