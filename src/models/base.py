"""Abstract base for model builders registered in `src.models.registry`."""

from abc import ABC, abstractmethod

import torch.nn as nn


class ModelDefinition(ABC):
    """
    One concrete subclass per architecture. Register with @register_model(\"key\").

    Subclasses implement build(); no shared state — configuration stays in build args.
    """

    @classmethod
    @abstractmethod
    def build(cls, num_classes: int, pretrained: bool, **kwargs) -> nn.Module:
        """Return a fresh nn.Module ready for training or evaluation."""
