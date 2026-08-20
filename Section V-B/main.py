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
    """将 data.path 列表展开为多个单数据集配置，并解析每个数据集的 name/path/num_labels。

    支持两种写法：
      - data.path: CoLA              （字符串，单数据集，保持原行为）
      - data.path: [CoLA, SST, Yelp] （列表，多数据集，逐个展开）
    """
    data_cfg = cfg.get("data", {})
    raw_path = data_cfg.get("path")
    if not isinstance(raw_path, list):
        yield cfg
        return
    for alias in raw_path:
        expanded_cfg = copy.deepcopy(cfg)
        expanded_cfg.setdefault("data", {})["path"] = alias
        expanded_cfg["data"]["name"] = alias
        # 重新解析命名引用，将该数据集的真实路径、num_labels 等填入
        _resolve_named_references(expanded_cfg)
        yield expanded_cfg


def _expand_batch_size_configs(cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    batch_sizes = _as_sweep_values(cfg.get("data", {}).get("batch_size", 1))
    for batch_size in batch_sizes:
        expanded_cfg = copy.deepcopy(cfg)
        expanded_cfg.setdefault("data", {})["batch_size"] = int(batch_size)
        yield expanded_cfg


def _expand_configs(cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """先展开数据集，再展开 batch_size。"""
    for dataset_cfg in _expand_dataset_configs(cfg):
        for batch_cfg in _expand_batch_size_configs(dataset_cfg):
            yield batch_cfg


def main() -> None:
    args = parse_args()
    for config_path in args.config:
        print(f"\nRunning config: {config_path}")
        cfg = load_config(config_path)

        # 每次运行 main.py 时，先遍历 results.json 重新计算已有结果的指标（不重跑实验）。
        # 这样在指标计算逻辑变更后（如 ROUGE-1/2 去重策略调整），
        # 旧记录的 metrics 会被同步更新；后续 has_completed 检测到 experiment_key 命中会跳过攻击。
        results_path = cfg.get("output", {}).get("path", "results/results.json")
        recompute_metrics_from_results(results_path, verbose=True)

        # 重新计算指标后，重建 summary.json 以反映最新的指标值
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
