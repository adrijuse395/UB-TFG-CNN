import torch
import torch.nn as nn
from torchvision import models

from .base import ModelDefinition
from .registry import register_model

# Datasets supported by chenyaofo/pytorch-cifar-models for this architecture
_CIFAR_HUB_VARIANTS = {
    10:  "cifar10_vgg11_bn",
    100: "cifar100_vgg11_bn",
}


@register_model("vgg11_bn")
class VGG11BNDefinition(ModelDefinition):
    @classmethod
    def build(cls, num_classes: int, pretrained: bool, **kwargs) -> nn.Module:
        hub_model = _CIFAR_HUB_VARIANTS.get(num_classes)
        if pretrained and hub_model is not None:
            print(f"       [ModelFactory] Downloading native CIFAR-{num_classes} VGG11 from torch hub...")
            return torch.hub.load(
                "chenyaofo/pytorch-cifar-models",
                hub_model,
                pretrained=True,
                trust_repo=True,
            )

        model = models.vgg11_bn(
            weights=models.VGG11_BN_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        return model

