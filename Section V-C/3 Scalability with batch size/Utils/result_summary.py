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

    grouped: Dict[Tuple[str, str, str, int, str, Any], List[Dict[str, Any]]] = {}
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
        fixed_length = record.get("fixed_length")
        key = (str(model_name), str(peft_method), str(method_name).lower(), int(batch_size), data_name, fixed_length)
        grouped.setdefault(key, []).append(
            {
                "f1": float(metrics.get("mean_f1_score", 0.0)),
                "rouge_l": float(metrics.get("mean_rouge_l", 0.0)),
                "rouge_1": float(metrics.get("mean_rouge_1", 0.0)),
                "rouge_2": float(metrics.get("mean_rouge_2", 0.0)),
                "token_accuracy": float(metrics.get("mean_token_accuracy", 0.0)),
                "mean_correct_tokens": float(metrics.get("mean_correct_tokens", 0.0)),
                "mean_recon_token_length": float(metrics.get("mean_recon_token_length", 0.0)),
            }
        )

    rows = []
    for (model_name, peft_method, method_name, batch_size, data_name, fixed_length), group in grouped.items():
        rows.append(
            {
                "model_name": model_name,
                "peft_method": peft_method,
                "reconstruction_method": method_name,
                "batch_size": batch_size,
                "data_name": data_name,
                "fixed_length": fixed_length,
                "mean_f1": _mean([r["f1"] for r in group]),
                "mean_rouge_l": _mean([r["rouge_l"] for r in group]),
                "mean_rouge_1": _mean([r["rouge_1"] for r in group]),
                "mean_rouge_2": _mean([r["rouge_2"] for r in group]),
                "mean_token_accuracy": _mean([r["token_accuracy"] for r in group]),
                "mean_recon_token_count": _mean([r["mean_correct_tokens"] for r in group]),
                "mean_recon_token_length": _mean([r["mean_recon_token_length"] for r in group]),
            }
        )

    rows.sort(
        key=lambda item: (
            item["model_name"],
            item["peft_method"],
            item["reconstruction_method"],
            item["batch_size"],
            item.get("data_name", ""),
            item.get("fixed_length") if item.get("fixed_length") is not None else -1,
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
    def _row_key(row: Dict[str, Any]) -> Tuple[str, str, str, int, str, Any]:
        return (
            str(row["model_name"]),
            str(row["peft_method"]),
            str(row["reconstruction_method"]),
            int(row["batch_size"]),
            str(row.get("data_name", "")),
            row.get("fixed_length"),
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
            if "mean_recon_token_count" not in row and "mean_reconstructed_token_count" in row:
                row["mean_recon_token_count"] = row.pop("mean_reconstructed_token_count")
            row.setdefault("data_name", "")
        return rows

    def append_batch(self, cfg: Dict[str, Any], method_name: str, metrics: Dict[str, Any]) -> None:
        data_cfg = cfg.get("data", {})
        fixed_length = data_cfg.get("fixed_length")
        self.batch_records.append(
            {
                "model_name": _model_display_name(cfg.get("model", {}).get("name_or_path", "")),
                "peft_method": cfg.get("selected_peft", "partial"),
                "reconstruction_method": method_name,
                "batch_size": int(data_cfg.get("batch_size", 1)),
                "data_name": data_cfg.get("name") or data_cfg.get("path") or "",
                "fixed_length": int(fixed_length) if fixed_length is not None else None,
                "f1": float(metrics.get("mean_f1_score", 0.0)),
                "rouge_l": float(metrics.get("mean_rouge_l", 0.0)),
                "rouge_1": float(metrics.get("mean_rouge_1", 0.0)),
                "rouge_2": float(metrics.get("mean_rouge_2", 0.0)),
                "token_accuracy": float(metrics.get("mean_token_accuracy", 0.0)),
                "mean_correct_tokens": float(metrics.get("mean_correct_tokens", 0.0)),
                "mean_recon_token_length": float(metrics.get("mean_recon_token_length", 0.0)),
            }
        )

    def save(self) -> None:
        grouped: Dict[Tuple[str, str, str, int, str, Any], List[Dict[str, Any]]] = {}
        for record in self.batch_records:
            key = (
                record["model_name"],
                record["peft_method"],
                record["reconstruction_method"],
                record["batch_size"],
                record["data_name"],
                record.get("fixed_length"),
            )
            grouped.setdefault(key, []).append(record)

        rows = []
        for (model_name, peft_method, reconstruction_method, batch_size, data_name, fixed_length), records in grouped.items():
            rows.append(
                {
                    "model_name": model_name,
                    "peft_method": peft_method,
                    "reconstruction_method": reconstruction_method,
                    "batch_size": batch_size,
                    "data_name": data_name,
                    "fixed_length": fixed_length,
                    "mean_f1": _mean([record["f1"] for record in records]),
                    "mean_rouge_l": _mean([record["rouge_l"] for record in records]),
                    "mean_rouge_1": _mean([record["rouge_1"] for record in records]),
                    "mean_rouge_2": _mean([record["rouge_2"] for record in records]),
                    "mean_token_accuracy": _mean([record["token_accuracy"] for record in records]),
                    "mean_recon_token_count": _mean([record["mean_correct_tokens"] for record in records]),
                    "mean_recon_token_length": _mean([record["mean_recon_token_length"] for record in records]),
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
        rows.sort(key=lambda item: (item["model_name"], item["peft_method"], item["reconstruction_method"], item["batch_size"], item.get("data_name", ""), item.get("fixed_length") if item.get("fixed_length") is not None else -1))
        if self.path.suffix.lower() == ".json":
            self.path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            return

        fieldnames = [
            "model_name",
            "peft_method",
            "reconstruction_method",
            "batch_size",
            "data_name",
            "fixed_length",
            "mean_f1",
            "mean_rouge_l",
            "mean_rouge_1",
            "mean_rouge_2",
            "mean_token_accuracy",
            "mean_recon_token_count",
            "mean_recon_token_length",
        ]
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
