import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any

class ModelFactory:
    """
    Factory class to instantiate PyTorch models.
    Designed to easily add new models in the future with < 10 lines of code.
    """
    
    @staticmethod
    def get_model(model_name: str, num_classes: int = 10, pretrained: bool = False, **kwargs: Dict[str, Any]) -> nn.Module:
        """
        Returns a PyTorch model based on the provided name.
        
        Args:
            model_name (str): The name of the architecture (e.g., 'vgg11_bn').
            num_classes (int): Number of output classes (e.g., 10 for CIFAR-10).
            pretrained (bool): Whether to load weights pretrained on ImageNet.
            
        Returns:
            nn.Module: The instantiated PyTorch model.
        """
        model_name = model_name.lower()
        
        if model_name == "vgg11_bn":
            if pretrained and num_classes == 10:
                # Use a natively trained CIFAR-10 model from torch hub (yields ~92% accuracy)
                print("       [ModelFactory] Downloading native CIFAR-10 VGG11 from torch hub...")
                model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=True, trust_repo=True)
                return model
            else:
                # Fallback to torchvision ImageNet model, adapted for arbitrary classes
                model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1 if pretrained else None)
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
            
        elif model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            # Adapt final fully connected layer
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
            
        else:
            raise ValueError(f"Model '{model_name}' is not supported. Please add it to the ModelFactory.")
