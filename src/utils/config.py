import json
import os
from typing import Any, Dict, List, Optional, Union

# Defaults merged with optional global_settings["resource_limits"].
# Keeps “config bomb” experiments from exhausting RAM/VRAM or running huge CP-ALS.
DEFAULT_RESOURCE_LIMITS: Dict[str, Any] = {
    "max_rank": 512,
    "max_target_layers_per_experiment": 64,
    "max_batch_size": 256,
    "cp_parafac_n_iter_max": 60,
    "cp_parafac_tol": 1e-5,
    # Run CP-ALS on CPU even if the layer sits on CUDA (avoids VRAM spikes during ALS).
    "cp_parafac_on_cpu": True,
    # TensorLy: explicit Khatri–Rao MTTKRP uses less RAM than the default fast path.
    "cp_memory_efficient_mttkrp": True,
    "cp_normalize_factors": True,
    # Safety controls for CP layer decomposition.
    "cp_layer_timeout_s": 20.0,
    "cp_abort_if_mem_available_mb_below": 800,
    "cp_init": "random",
    # Keep CPU pressure bounded on laptops.
    "cpu_num_threads": 2,
    "cpu_num_interop_threads": 1,
}


class ConfigParser:
    """
    Parses experimental configurations from JSON files.
    Allows for structured and reproducible experiments.
    """

    @staticmethod
    def merge_resource_limits(global_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns resource_limits with defaults, overridden by global_settings["resource_limits"].
        """
        user = global_settings.get("resource_limits") or {}
        merged = {**DEFAULT_RESOURCE_LIMITS, **user}
        merged["max_rank"] = int(merged["max_rank"])
        merged["max_target_layers_per_experiment"] = int(
            merged["max_target_layers_per_experiment"]
        )
        merged["max_batch_size"] = int(merged["max_batch_size"])
        merged["cp_parafac_n_iter_max"] = int(merged["cp_parafac_n_iter_max"])
        merged["cp_parafac_tol"] = float(merged["cp_parafac_tol"])
        merged["cp_parafac_on_cpu"] = bool(merged["cp_parafac_on_cpu"])
        merged["cp_memory_efficient_mttkrp"] = bool(merged["cp_memory_efficient_mttkrp"])
        merged["cp_normalize_factors"] = bool(merged["cp_normalize_factors"])
        merged["cp_layer_timeout_s"] = float(merged["cp_layer_timeout_s"])
        merged["cp_abort_if_mem_available_mb_below"] = int(
            merged["cp_abort_if_mem_available_mb_below"]
        )
        merged["cp_init"] = str(merged["cp_init"]).strip().lower()
        if merged["cp_init"] not in {"svd", "random"}:
            merged["cp_init"] = "random"
        merged["cpu_num_threads"] = int(merged["cpu_num_threads"])
        merged["cpu_num_interop_threads"] = int(merged["cpu_num_interop_threads"])
        return merged

    @staticmethod
    def clamp_rank_for_method(
        rank: Optional[Union[int, List[int]]], method: str, limits: Dict[str, Any]
    ) -> Optional[Union[int, List[int]]]:
        """Clamp rank(s) to resource_limits.max_rank (method-specific shapes preserved)."""
        if rank is None:
            return None
        max_r = max(1, limits["max_rank"])

        if method == "Tucker":
            if isinstance(rank, list):
                return [max(1, min(int(x), max_r)) for x in rank]
            return max(1, min(int(rank), max_r))

        if method in {"CP", "CP_HOSVD", "CP_ALS_LIGHT"}:
            if isinstance(rank, list):
                if not rank:
                    raise ValueError(f"{method} rank list is empty.")
                return [max(1, min(int(rank[0]), max_r))]
            return max(1, min(int(rank), max_r))

        if method == "TT":
            if isinstance(rank, list):
                if len(rank) == 3:
                    return [max(1, min(int(x), max_r)) for x in rank]
                # Interpret as full TT rank list [1, r1, r2, r3, 1] or similar
                return [max(1, min(int(x), max_r)) for x in rank]
            return max(1, min(int(rank), max_r))

        return rank

    @staticmethod
    def load_config(filepath: str) -> Dict[str, Any]:
        """
        Loads and validates a JSON configuration file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in {filepath}: {e}")
                
        # Basic validation
        required_keys = ["global_settings", "experiments"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in configuration.")

        return config
