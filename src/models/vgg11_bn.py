import torch
import torch.nn as nn
from torchvision import models

from .base import ModelDefinition
from .registry import register_model


@register_model("vgg11_bn")
class VGG11BNDefinition(ModelDefinition):
    @classmethod
    def build(cls, num_classes: int, pretrained: bool, **kwargs) -> nn.Module:
        if pretrained and num_classes == 10:
            print("       [ModelFactory] Downloading native CIFAR-10 VGG11 from torch hub...")
            return torch.hub.load(
                "chenyaofo/pytorch-cifar-models",
                "cifar10_vgg11_bn",
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
