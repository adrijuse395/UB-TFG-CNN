"""Model name → ModelDefinition registry (decorator-based)."""

from typing import Dict, Tuple, Type

from .base import ModelDefinition

_REGISTRY: Dict[str, Type[ModelDefinition]] = {}


def register_model(name: str):
    """Decorator: register this ModelDefinition subclass under `name` (case-insensitive)."""

    def decorator(cls: Type[ModelDefinition]):
        key = name.strip().lower()
        if key in _REGISTRY:
            raise ValueError(f"Duplicate model registration for '{key}'.")
        _REGISTRY[key] = cls
        return cls

    return decorator


def get_definition(model_name: str) -> Type[ModelDefinition]:
    key = str(model_name).strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"Model '{model_name}' is not supported. Registered models: {available}"
        )
    return _REGISTRY[key]


def list_registered_models() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY.keys()))
