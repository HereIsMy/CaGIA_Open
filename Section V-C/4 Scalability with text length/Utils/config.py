from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "cola": {"num_labels": 2},
    "sst": {"num_labels": 5},
    "yelp": {"num_labels": 2},
    "imdb": {"num_labels": 2},
}


def _simple_yaml_value(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") or value.startswith("{"):
        return json.loads(value.replace("'", '"'))
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


def _fallback_yaml_load(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, value = raw_line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            node: Dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _simple_yaml_value(value)
    return root


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
        return _resolve_named_references(data or {})
    except ImportError:
        return _resolve_named_references(_fallback_yaml_load(text))


def _lookup_case_insensitive(mapping: Dict[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    lowered = key.lower()
    for candidate, value in mapping.items():
        if str(candidate).lower() == lowered:
            return value
    raise KeyError(key)


def _resolve_named_references(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = cfg.setdefault("model", {})
    model_name = model_cfg.get("name") or model_cfg.get("name_or_path")
    model_paths = cfg.get("model_path", {}) or {}
    if isinstance(model_name, str) and model_paths:
        try:
            model_cfg["name"] = model_name
            model_cfg["name_or_path"] = _lookup_case_insensitive(model_paths, model_name)
        except KeyError:
            pass

    data_cfg = cfg.setdefault("data", {})
    dataset_name = data_cfg.get("name") or data_cfg.get("dataset") or data_cfg.get("path")
    data_paths = cfg.get("data_path", {}) or {}
    if isinstance(dataset_name, str) and data_paths:
        try:
            data_cfg["name"] = dataset_name
            data_cfg["path"] = _lookup_case_insensitive(data_paths, dataset_name)
            defaults = DATASET_DEFAULTS.get(dataset_name.lower(), {})
            for key, value in defaults.items():
                data_cfg.setdefault(key, value)
        except KeyError:
            pass
    elif isinstance(data_cfg.get("name"), str):
        defaults = DATASET_DEFAULTS.get(str(data_cfg["name"]).lower(), {})
        for key, value in defaults.items():
            data_cfg.setdefault(key, value)

    resolved_name = str(data_cfg.get("name") or "").lower()
    dataset_defaults = DATASET_DEFAULTS.get(resolved_name, {})
    if "num_labels" in dataset_defaults:
        data_cfg["num_labels"] = dataset_defaults["num_labels"]
        model_cfg["num_labels"] = dataset_defaults["num_labels"]
    elif "num_labels" in data_cfg:
        model_cfg["num_labels"] = data_cfg["num_labels"]
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedSGD gradient inversion attacks")
    parser.add_argument("--config", nargs="+", default=["config.yaml"], help="One or more config.yaml paths")
    return parser.parse_args()
