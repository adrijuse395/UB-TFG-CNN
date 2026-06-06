"""
Model instantiation via registered ModelDefinition subclasses.

Adding a model:
  1. Create src/models/<name>.py with @register_model(\"key\") class ... ModelDefinition.
  2. Import that module below so registration runs at import time.
"""

from typing import Any

import torch.nn as nn

from .registry import get_definition

# Side-effect imports: populate registry
from . import resnet18  # noqa: F401
from . import resnet20  # noqa: F401
from . import vgg11_bn  # noqa: F401


class ModelFactory:
    """Thin façade over the model registry."""

    @staticmethod
    def get_model(
        model_name: str,
        num_classes: int = 10,
        pretrained: bool = False,
        **kwargs: Any,
    ) -> nn.Module:
        definition = get_definition(model_name)
        return definition.build(num_classes=num_classes, pretrained=pretrained, **kwargs)
