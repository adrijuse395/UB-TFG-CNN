import torch.nn as nn
from torchvision import models

from .base import ModelDefinition
from .registry import register_model


@register_model("resnet18")
class ResNet18Definition(ModelDefinition):
    @classmethod
    def build(cls, num_classes: int, pretrained: bool, **kwargs) -> nn.Module:
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
