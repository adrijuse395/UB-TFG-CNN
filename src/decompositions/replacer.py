import torch
import torch.nn as nn
from typing import Dict, Any, Type
from .base import BaseDecomposedLayer

class ModelReplacer:
    """
    Utility class to replace standard PyTorch layers (Conv2d, Linear)
    with their decomposed equivalents (e.g., Tucker, CP, TT).
    """

    @staticmethod
    def replace_layers(module: nn.Module, decomposition_class: Type[BaseDecomposedLayer], 
                       target_layers: list, **kwargs):
        """
        Recursively searches the module for layers whose name matches one of the targets,
        and replaces them with an instance of `decomposition_class`.
        
        Args:
            module (nn.Module): The root PyTorch model.
            decomposition_class (Type[BaseDecomposedLayer]): The class to instantiate (e.g., TuckerDecomposedLayer).
            target_layers (list): List of layer names to target (e.g., ['features.0', 'classifier.3']).
            **kwargs: Arguments to pass to the decomposition (e.g., rank=[16, 16]).
        """
        ModelReplacer._replace_recursive(module, decomposition_class, target_layers, current_path="", **kwargs)

    @staticmethod
    def _replace_recursive(module: nn.Module, decomposition_class: Type[BaseDecomposedLayer], 
                           target_layers: list, current_path: str, **kwargs):
        
        for name, child in module.named_children():
            # Build the full path of the layer, e.g., "features.0"
            full_name = f"{current_path}.{name}" if current_path else name
            
            if full_name in target_layers:
                # Target matched. Validate type.
                if isinstance(child, (nn.Conv2d, nn.Linear)):
                    print(f"    [Replacer] Replacing '{full_name}' ({type(child).__name__}) with {decomposition_class.__name__}...")
                    
                    # Create the decomposed layer
                    decomposed_layer = decomposition_class.from_layer(child, **kwargs)
                    
                    # Swap the module
                    setattr(module, name, decomposed_layer)
                else:
                    print(f"    [Warning] Target '{full_name}' is a {type(child).__name__}, which is not supported for decomposition. Skipping.")
            else:
                # Recurse deeper into blocks (like Sequential, BasicBlock, etc.)
                ModelReplacer._replace_recursive(child, decomposition_class, target_layers, full_name, **kwargs)
