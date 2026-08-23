from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _model_display_name(model_path: str) -> str:
    normalized = str(model_path).replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] if normalized else str(model_path)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rebuild_summary_from_results(results_path: str, summary_path: str) -> None:
    results_file = Path(results_path)
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    if not results_file.exists() or results_file.stat().st_size == 0:
        summary_file.write_text("[]", encoding="utf-8")
        return

    records = json.loads(results_file.read_text(encoding="utf-8"))

    grouped: Dict[Tuple[str, str, str, int, str], List[Dict[str, Any]]] = {}
    skipped = 0
    for record in records:
        model_name = record.get("model_name")
        peft_method = record.get("peft_method")
        method_name = record.get("method_name")
        batch_size = record.get("batch_size")
        metrics = record.get("metrics")
        if not all(v is not None for v in [model_name, peft_method, method_name, batch_size]) or not metrics:
            skipped += 1
            continue

        data_name = str(record.get("data_name", ""))
        key = (str(model_name), str(peft_method), str(method_name).lower(), int(batch_size), data_name)
        details = metrics.get("sentence_details", [])
        correct_tokens = [float(d.get("correct_tokens", 0.0)) for d in details]
        grouped.setdefault(key, []).append(
            {
                "f1": float(metrics.get("mean_f1_score", 0.0)),
                "rouge_l": float(metrics.get("mean_rouge_l", 0.0)),
                "correct_token_count": _mean(correct_tokens),
            }
        )

    rows = []
    for (model_name, peft_method, method_name, batch_size, data_name), group in grouped.items():
        rows.append(
            {
                "model_name": model_name,
                "peft_method": peft_method,
                "reconstruction_method": method_name,
                "batch_size": batch_size,
                "data_name": data_name,
                "mean_f1": _mean([r["f1"] for r in group]),
                "mean_rouge_l": _mean([r["rouge_l"] for r in group]),
                "mean_correct_token_count": _mean([r["correct_token_count"] for r in group]),
            }
        )

    rows.sort(
        key=lambda item: (
            item["model_name"],
            item["peft_method"],
            item["reconstruction_method"],
            item["batch_size"],
            item.get("data_name", ""),
        )
    )
    summary_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if skipped:
        pass


class ResultSummaryWriter:

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_records: List[Dict[str, Any]] = []

    @staticmethod
    def _row_key(row: Dict[str, Any]) -> Tuple[str, str, str, int, str]:
        return (
            str(row["model_name"]),
            str(row["peft_method"]),
            str(row["reconstruction_method"]),
            int(row["batch_size"]),
            str(row.get("data_name", "")),
        )

    def _load_existing_rows(self) -> List[Dict[str, Any]]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        if self.path.suffix.lower() == ".json":
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        for row in rows:
            if "mean_correct_token_count" not in row and "mean_reconstructed_token_count" in row:
                row["mean_correct_token_count"] = row.pop("mean_reconstructed_token_count")
            row.setdefault("data_name", "")
        return rows

    def append_batch(self, cfg: Dict[str, Any], method_name: str, metrics: Dict[str, Any]) -> None:
        details = metrics.get("sentence_details", [])
        correct_token_counts = [float(item.get("correct_tokens", 0.0)) for item in details]
        self.batch_records.append(
            {
                "model_name": _model_display_name(cfg.get("model", {}).get("name_or_path", "")),
                "peft_method": cfg.get("selected_peft", "partial"),
                "reconstruction_method": method_name,
                "batch_size": int(cfg.get("data", {}).get("batch_size", 1)),
                "data_name": cfg.get("data", {}).get("name") or cfg.get("data", {}).get("path") or "",
                "f1": float(metrics.get("mean_f1_score", 0.0)),
                "rouge_l": float(metrics.get("mean_rouge_l", 0.0)),
                "correct_token_count": _mean(correct_token_counts),
            }
        )

    def save(self) -> None:
        grouped: Dict[Tuple[str, str, str, int, str], List[Dict[str, Any]]] = {}
        for record in self.batch_records:
            key = (
                record["model_name"],
                record["peft_method"],
                record["reconstruction_method"],
                record["batch_size"],
                record["data_name"],
            )
            grouped.setdefault(key, []).append(record)

        rows = []
        for (model_name, peft_method, reconstruction_method, batch_size, data_name), records in grouped.items():
            rows.append(
                {
                    "model_name": model_name,
                    "peft_method": peft_method,
                    "reconstruction_method": reconstruction_method,
                    "batch_size": batch_size,
                    "data_name": data_name,
                    "mean_f1": _mean([record["f1"] for record in records]),
                    "mean_rouge_l": _mean([record["rouge_l"] for record in records]),
                    "mean_correct_token_count": _mean([record["correct_token_count"] for record in records]),
                }
            )

        new_other_keys = {
            (r["model_name"], r["peft_method"], r["reconstruction_method"], r["batch_size"])
            for r in rows
        }
        legacy_rows: List[Dict[str, Any]] = []
        for row in self._load_existing_rows():
            if not row.get("data_name"):
                other_key = (
                    str(row["model_name"]),
                    str(row["peft_method"]),
                    str(row["reconstruction_method"]),
                    int(row["batch_size"]),
                )
                if other_key in new_other_keys:
                    continue
            legacy_rows.append(row)

        merged = {self._row_key(row): row for row in legacy_rows}
        for row in rows:
            merged[self._row_key(row)] = row
        rows = list(merged.values())
        rows.sort(key=lambda item: (item["model_name"], item["peft_method"], item["reconstruction_method"], item["batch_size"], item.get("data_name", "")))
        if self.path.suffix.lower() == ".json":
            self.path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            return

        fieldnames = [
            "model_name",
            "peft_method",
            "reconstruction_method",
            "batch_size",
            "data_name",
            "mean_f1",
            "mean_rouge_l",
            "mean_correct_token_count",
        ]
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
