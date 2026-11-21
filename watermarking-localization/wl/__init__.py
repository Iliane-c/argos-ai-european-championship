"""Core constants, path helpers, and hierarchical YAML config loader for wl."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_KIT_DIR = ROOT_DIR / "kit"
DEFAULT_CONFIGS_DIR = ROOT_DIR / "configs"

METHODS = ("pixelseal", "trustmark", "stegastamp")
METHOD_TO_ID = {"pixelseal": 0, "trustmark": 1, "stegastamp": 2}
ID_TO_METHOD = {0: "pixelseal", 1: "trustmark", 2: "stegastamp"}
METHOD_SHORT = {"pixelseal": "PS", "trustmark": "TM", "stegastamp": "SS"}
NBITS = {"pixelseal": 256, "trustmark": 100, "stegastamp": 100}
MAX_BITS = 256
NATIVE_RES = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}
MAX_ASPECT = {"pixelseal": 16.0, "trustmark": 2.0, "stegastamp": 16.0}

IMG_SIZE = 512
GRID = 128
CELL = 4


def ensure_kit_on_path(kit_dir: str | Path | None = None) -> Path:
    """Add starter kit directory to sys.path if missing so `wmloc` imports work anywhere."""
    resolved = Path(kit_dir) if kit_dir is not None else DEFAULT_KIT_DIR
    resolved_str = str(resolved.resolve())
    if resolved_str not in sys.path:
        sys.path.insert(0, resolved_str)
    return resolved


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary into a deep copy of base."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_override_value(raw: str) -> Any:
    """Parse a CLI override value string with YAML typing."""
    if raw == "":
        return ""
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply dotted `section.key=value` overrides to a config dictionary."""
    if not overrides:
        return cfg
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected key=value override, got: {item!r}")
        key_path, raw_val = item.split("=", 1)
        parts = [p for p in key_path.strip().split(".") if p]
        if not parts:
            raise ValueError(f"Invalid override key path: {item!r}")
        cursor = out
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = parse_override_value(raw_val)
    return out


def _resolve_config_path(ref: str | Path, search_dirs: list[Path]) -> Path:
    p = Path(ref)
    candidates = [p] if p.is_absolute() else []
    for d in search_dirs:
        candidates.append(d / p)
        if p.suffix != ".yaml":
            candidates.append(d / f"{p}.yaml")
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"Config not found: {ref} (searched {[str(d) for d in search_dirs]})")


def _load_yaml_chain(path: Path, search_dirs: list[Path], seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Cyclic config parent reference involving {resolved}")
    seen.add(resolved)
    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {resolved} must deserialize to a mapping")

    parent_ref = data.get("parent")
    # Every config implicitly inherits base.yaml unless it is base.yaml itself or sets an explicit parent chain.
    if parent_ref and str(parent_ref) not in ("none", "null", "—"):
        parent_path = _resolve_config_path(str(parent_ref), [resolved.parent, *search_dirs])
        base_cfg = _load_yaml_chain(parent_path, search_dirs, seen)
    else:
        base_path = _resolve_config_path("base.yaml", [resolved.parent, *search_dirs])
        if base_path != resolved:
            base_cfg = _load_yaml_chain(base_path, search_dirs, seen)
        else:
            base_cfg = {}
    return deep_merge(base_cfg, data)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load experiment YAML with base/parent inheritance and dotted CLI overrides."""
    search_dirs = [Path.cwd(), DEFAULT_CONFIGS_DIR, ROOT_DIR]
    cfg_path = _resolve_config_path(path, search_dirs)
    cfg = _load_yaml_chain(cfg_path, [cfg_path.parent, DEFAULT_CONFIGS_DIR], seen=set())
    cfg = apply_overrides(cfg, overrides)
    cfg.setdefault("id", cfg_path.stem)
    return cfg
