import torch
import torch.nn as nn

from .base import ModelDefinition
from .registry import register_model

# Datasets supported by chenyaofo/pytorch-cifar-models for this architecture
_CIFAR_HUB_VARIANTS = {
    10:  "cifar10_resnet20",
    100: "cifar100_resnet20",
}


@register_model("resnet20")
class ResNet20Definition(ModelDefinition):
    @classmethod
    def build(cls, num_classes: int, pretrained: bool, **kwargs) -> nn.Module:
        hub_model = _CIFAR_HUB_VARIANTS.get(num_classes)
        if pretrained and hub_model is not None:
            print(f"       [ModelFactory] Downloading native CIFAR-{num_classes} ResNet20 from torch hub...")
            # Prevent interactive prompt on Kaggle/Colab
            torch.hub.trusted_list.append("chenyaofo/pytorch-cifar-models")
            return torch.hub.load(
                "chenyaofo/pytorch-cifar-models",
                hub_model,
                pretrained=True,
            )

        raise NotImplementedError(
            f"Pretrained ResNet20 is only supported for CIFAR-10/100 (num_classes=10/100)."
        )
