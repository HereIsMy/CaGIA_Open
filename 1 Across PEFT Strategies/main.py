from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from Utils.config import _resolve_named_references, load_config, parse_args
from Utils.federated import run_fedsgd, set_seed
from Utils.metrics import recompute_metrics_from_results


def _as_sweep_values(value: Any) -> List[Any]:
    return value if isinstance(value, list) else [value]


def _expand_dataset_configs(cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    raw_path = data_cfg.get("path")
    if not isinstance(raw_path, list):
        yield cfg
        return
    for alias in raw_path:
        expanded_cfg = copy.deepcopy(cfg)
        expanded_cfg.setdefault("data", {})["path"] = alias
        expanded_cfg["data"]["name"] = alias
        _resolve_named_references(expanded_cfg)
        yield expanded_cfg


def _expand_batch_size_configs(cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    batch_sizes = _as_sweep_values(cfg.get("data", {}).get("batch_size", 1))
    for batch_size in batch_sizes:
        expanded_cfg = copy.deepcopy(cfg)
        expanded_cfg.setdefault("data", {})["batch_size"] = int(batch_size)
        yield expanded_cfg


def _expand_peft_configs(cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    raw_peft = cfg.get("selected_peft", "partial")
    if not isinstance(raw_peft, list):
        yield cfg
        return
    for peft_method in raw_peft:
        expanded_cfg = copy.deepcopy(cfg)
        expanded_cfg["selected_peft"] = str(peft_method)
        yield expanded_cfg


def _expand_configs(cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for dataset_cfg in _expand_dataset_configs(cfg):
        for batch_cfg in _expand_batch_size_configs(dataset_cfg):
            for peft_cfg in _expand_peft_configs(batch_cfg):
                yield peft_cfg


def main() -> None:
    args = parse_args()
    for config_path in args.config:
        print(f"\nRunning config: {config_path}")
        cfg = load_config(config_path)

        results_path = cfg.get("output", {}).get("path", "results/results.json")
        recompute_metrics_from_results(results_path, verbose=True)

        summary_path = cfg.get("output", {}).get("summary_path", "results/summary.json")
        if summary_path:
            from Utils.result_summary import rebuild_summary_from_results
            rebuild_summary_from_results(results_path, summary_path)
            print(f"[Summary] 已根据新指标重建 {summary_path}")

        expanded_configs = list(_expand_configs(cfg))
        for run_idx, run_cfg in enumerate(expanded_configs, start=1):
            if len(expanded_configs) > 1:
                data_name = run_cfg.get("data", {}).get("name", "?")
                batch_size = run_cfg["data"]["batch_size"]
                print(f"\nSweep {run_idx}/{len(expanded_configs)}: dataset={data_name}, batch_size={batch_size}")
            set_seed(int(run_cfg.get("seed", 1)))
            device = torch.device(run_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
            run_fedsgd(run_cfg, device)


if __name__ == "__main__":
    main()
