"""
Central registry for tensor decomposition layer classes.

Config JSON references methods by name (e.g. Tucker, CP).
Adding a new decomposition: implement BaseDecomposedLayer, import it here, and
add one entry to DECOMPOSITION_REGISTRY.
"""

from typing import Dict, Tuple, Type

from .base import BaseDecomposedLayer
from .cp import CPDecomposedLayer
from .cp_gradient import CPGradientDecomposedLayer
from .tt import TTDecomposedLayer
from .tucker import TuckerDecomposedLayer
from .svd import SVDDecomposedLayer

DECOMPOSITION_REGISTRY: Dict[str, Type[BaseDecomposedLayer]] = {
    "Tucker": TuckerDecomposedLayer,
    "CP": CPDecomposedLayer,
    "CP_GD": CPGradientDecomposedLayer,
    "TT": TTDecomposedLayer,
    "SVD": SVDDecomposedLayer,
}


def get_decomposition_class(method: str) -> Type[BaseDecomposedLayer]:
    """Return the layer class for a config method name, or raise KeyError."""
    return DECOMPOSITION_REGISTRY[method]


def list_decomposition_methods() -> Tuple[str, ...]:
    return tuple(sorted(DECOMPOSITION_REGISTRY.keys()))
