import json
import os
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# resource_limits — only global quotas.
# ---------------------------------------------------------------------------
DEFAULT_RESOURCE_LIMITS: Dict[str, Any] = {
    "max_rank": 512,
    "max_batch_size": 256,
}

# ---------------------------------------------------------------------------
# method_defaults — method-specific knobs, independently defined per method.
# JSON shape:
#   "method_defaults": {
#       "CP": {...},
#       "CP_GD": {...}
#   }
# and experiment-level overrides with:
#   "method_params": {...}
# ---------------------------------------------------------------------------
DEFAULT_METHOD_PARAMS: Dict[str, Dict[str, Any]] = {
    "CP": {},
    "CP_GD": {
        "cp_gd_steps": 3000,
        "cp_gd_lr": 0.05,
        "cp_gd_on_cpu": True,
        "cp_gd_init": "svd",
        "cp_gd_scheduler_patience": 200,
    },
}

class ConfigParser:
    """
    Parses experimental configurations from JSON files.
    Allows for structured and reproducible experiments.
    """

    @staticmethod
    def merge_resource_limits(
        user_resource_limits: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge and coerce only global resource limit keys."""
        user = user_resource_limits or {}
        merged = {**DEFAULT_RESOURCE_LIMITS, **{k: v for k, v in user.items() if k in DEFAULT_RESOURCE_LIMITS}}
        merged["max_rank"] = int(merged["max_rank"])
        merged["max_batch_size"] = int(merged["max_batch_size"])
        return merged

    @staticmethod
    def _coerce_method_params(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(params)
        if method == "CP_GD":
            p["cp_gd_steps"] = int(p.get("cp_gd_steps", 3000))
            p["cp_gd_lr"] = float(p.get("cp_gd_lr", 0.05))
            p["cp_gd_on_cpu"] = bool(p.get("cp_gd_on_cpu", True))
            p["cp_gd_init"] = str(p.get("cp_gd_init", "svd")).strip().lower()
            if p["cp_gd_init"] not in {"svd", "random"}:
                p["cp_gd_init"] = "svd"
            p["cp_gd_scheduler_patience"] = int(p.get("cp_gd_scheduler_patience", 200))
        return p

    @staticmethod
    def resolve_method_params(
        *,
        method: str,
        config: Dict[str, Any],
        global_settings: Dict[str, Any],
        experiment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve method params with precedence:
        defaults < config.method_defaults[method] < global_settings.method_defaults[method]
        < experiment.method_params

        No legacy mapping: keep config surface explicit and minimal.
        """
        base = dict(DEFAULT_METHOD_PARAMS.get(method, {}))

        cfg_defs = (config.get("method_defaults") or {}).get(method, {})
        gs_defs = (global_settings.get("method_defaults") or {}).get(method, {})
        exp_defs = experiment.get("method_params") or {}

        merged = {**base, **cfg_defs, **gs_defs, **exp_defs}

        return ConfigParser._coerce_method_params(method, merged)

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

        if method in {"CP", "CP_GD"}:
            if isinstance(rank, list):
                if not rank:
                    raise ValueError(f"{method} rank list is empty.")
                return [max(1, min(int(rank[0]), max_r))]
            return max(1, min(int(rank), max_r))

        if method == "TT":
            if isinstance(rank, list):
                if len(rank) == 3:
                    return [max(1, min(int(x), max_r)) for x in rank]
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

        with open(filepath, "r") as f:
            try:
                cfg = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in {filepath}: {e}") from e

        required_keys = ["global_settings", "experiments"]
        for key in required_keys:
            if key not in cfg:
                raise ValueError(f"Missing required key '{key}' in configuration.")

        return cfg
