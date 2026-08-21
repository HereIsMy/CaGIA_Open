from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, rows: Iterable[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        return {"sentence": row["sentence"], "labels": torch.tensor(int(row["label"]), dtype=torch.long)}


def _coerce_binary_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        numeric = float(value)
        if numeric in {0.0, 1.0}:
            return int(numeric)
        return 1 if numeric > 0 else 0
    except (TypeError, ValueError):
        text = str(value).strip().lower()
        return 1 if text in {"1", "true", "positive", "pos"} else 0


def _filter_valid_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [row for row in rows if row.get("sentence")]
    return rows


def load_cola_dataset(path: str) -> TextClassificationDataset:
    path_obj = Path(path)
    rows: List[Dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if not parts or len(parts) < 4:
                continue
            label = parts[1]
            sentence = parts[3]
            rows.append({"label": _coerce_binary_label(label), "sentence": sentence})
    rows = _filter_valid_rows(rows)
    if not rows:
        raise ValueError(f"No CoLA rows loaded from {os.fspath(path_obj)}.")
    return TextClassificationDataset(rows)


def load_sst_dataset(path: str) -> TextClassificationDataset:
    path_obj = Path(path)
    rows: List[Dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row_idx, parts in enumerate(reader):
            if row_idx == 0:
                continue
            if not parts or len(parts) < 2:
                continue
            sentence = parts[0]
            label = parts[1]
            rows.append({"label": int(label), "sentence": sentence})
    rows = _filter_valid_rows(rows)
    if not rows:
        raise ValueError(f"No SST rows loaded from {os.fspath(path_obj)}.")
    return TextClassificationDataset(rows)


def load_yelp_dataset(path: str) -> TextClassificationDataset:
    path_obj = Path(path)
    rows: List[Dict[str, Any]] = []
    for line in path_obj.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        text = item.get("text") or item.get("sentence") or item.get("review")
        if not text:
            continue
        if "stars" in item:
            label = 1 if float(item["stars"]) >= 4 else 0
        else:
            label = 1 if int(item.get("compliment_count", 0)) > 0 else 0
        rows.append({"label": label, "sentence": text})
    rows = _filter_valid_rows(rows)
    if not rows:
        raise ValueError(f"No Yelp rows loaded from {os.fspath(path_obj)}.")
    return TextClassificationDataset(rows)


def _iter_imdb_rows(path_obj: Path) -> Iterable[Dict[str, Any]]:
    for label, dirname in ((1, "pos"), (0, "neg")):
        folder = path_obj / dirname
        if not folder.exists():
            continue
        for file_path in sorted(folder.glob("*.txt")):
            yield {"label": label, "sentence": file_path.read_text(encoding="utf-8", errors="ignore").strip()}


def load_imdb_dataset(path: str) -> TextClassificationDataset:
    path_obj = Path(path)
    rows = list(_iter_imdb_rows(path_obj))
    rows = _filter_valid_rows(rows)
    if not rows:
        raise ValueError(f"No IMDB rows loaded from {os.fspath(path_obj)}.")
    return TextClassificationDataset(rows)


_DATASET_LOADERS = {
    "cola": load_cola_dataset,
    "sst": load_sst_dataset,
    "yelp": load_yelp_dataset,
    "imdb": load_imdb_dataset,
}


def load_dataset(cfg: Dict[str, Any]) -> TextClassificationDataset:
    data_cfg = cfg.get("data", {})
    name = str(data_cfg.get("name") or data_cfg.get("dataset") or "").lower()
    if name not in _DATASET_LOADERS:
        raise ValueError(
            f"Unsupported dataset '{name}'. Supported: {list(_DATASET_LOADERS.keys())}. "
            "请在 data_path 中配置数据集名称，并在 data.path 处使用别名。"
        )
    path = data_cfg["path"]
    return _DATASET_LOADERS[name](path)


def collate_text_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sentence": [item["sentence"] for item in batch],
        "labels": torch.stack([item["labels"] for item in batch]),
    }
