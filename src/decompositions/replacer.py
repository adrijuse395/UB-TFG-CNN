import torch.nn as nn
from typing import List, Type

from .base import BaseDecomposedLayer


class ModelReplacer:
    """
    Utility class to replace standard PyTorch layers (Conv2d, Linear)
    with their decomposed equivalents (e.g., Tucker, CP, TT).

    Every name in ``target_layers`` must exist, be Conv2d/Linear, compress successfully,
    and end as a ``BaseDecomposedLayer``; otherwise an error is raised (no partial runs).
    """

    @staticmethod
    def _module_at_path(root: nn.Module, path: str) -> nn.Module:
        if not path or not str(path).strip():
            raise ValueError("target_layers entry must be a non-empty dotted path.")
        cur: nn.Module = root
        for part in path.split("."):
            if not hasattr(cur, part):
                raise ValueError(
                    f"Target layer '{path}' not found: attribute '{part}' missing on {type(cur).__name__}."
                )
            child = getattr(cur, part)
            if not isinstance(child, nn.Module):
                raise ValueError(
                    f"Target layer '{path}' is invalid: '{part}' is not an nn.Module ({type(child).__name__})."
                )
            cur = child
        return cur

    @staticmethod
    def _validate_targets(root: nn.Module, target_layers: List[str]) -> None:
        for path in target_layers:
            layer = ModelReplacer._module_at_path(root, path)
            if not isinstance(layer, (nn.Conv2d, nn.Linear)):
                raise ValueError(
                    f"Target layer '{path}' is {type(layer).__name__}; only Conv2d and Linear can be decomposed."
                )

    @staticmethod
    def _assert_all_targets_decomposed(root: nn.Module, target_layers: List[str]) -> None:
        for path in target_layers:
            layer = ModelReplacer._module_at_path(root, path)
            if not isinstance(layer, BaseDecomposedLayer):
                raise RuntimeError(
                    f"Layer '{path}' was not replaced with a decomposed layer (still {type(layer).__name__}). "
                    "Check target path spelling and that the layer is reachable in named_children."
                )

    @staticmethod
    def replace_layers(
        module: nn.Module,
        decomposition_class: Type[BaseDecomposedLayer],
        target_layers: list,
        **kwargs,
    ):
        """
        Recursively searches the module for layers whose name matches one of the targets,
        and replaces them with an instance of `decomposition_class`.

        Args:
            module (nn.Module): The root PyTorch model.
            decomposition_class (Type[BaseDecomposedLayer]): The class to instantiate (e.g., TuckerDecomposedLayer).
            target_layers (list): List of layer names to target (e.g., ['features.0', 'classifier.3']).
            **kwargs: Arguments to pass to the decomposition (e.g., rank=[16, 16]).
        """
        paths = [str(p).strip() for p in (target_layers or []) if str(p).strip()]
        if not paths:
            return
        ModelReplacer._validate_targets(module, paths)
        ModelReplacer._replace_recursive(module, decomposition_class, paths, current_path="", **kwargs)
        ModelReplacer._assert_all_targets_decomposed(module, paths)

    @staticmethod
    def _replace_recursive(
        module: nn.Module,
        decomposition_class: Type[BaseDecomposedLayer],
        target_layers: list,
        current_path: str,
        **kwargs,
    ):
        for name, child in module.named_children():
            full_name = f"{current_path}.{name}" if current_path else name

            if full_name in target_layers:
                if isinstance(child, (nn.Conv2d, nn.Linear)):
                    print(
                        f"    [Replacer] Replacing '{full_name}' ({type(child).__name__}) "
                        f"with {decomposition_class.__name__}..."
                    )
                    decomposed_layer = decomposition_class.from_layer(child, **kwargs)
                    setattr(module, name, decomposed_layer)
                else:
                    raise ValueError(
                        f"Target '{full_name}' is {type(child).__name__}; expected Conv2d or Linear "
                        "(pre-validation should have caught this — inconsistent model?)."
                    )
            else:
                ModelReplacer._replace_recursive(child, decomposition_class, target_layers, full_name, **kwargs)
