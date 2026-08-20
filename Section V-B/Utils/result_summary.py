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


def recalculate_metrics_in_results(results_path: str) -> None:
    """遍历 results.json 中的记录，仅重新计算指标，不重跑实验。

    每次运行 main.py 时调用：读取已落盘的重建结果（reference_text 与
    attack.reconstructed_text），用最新的指标计算方法重新计算 metrics 字段
    并写回 results.json。这样修改指标算法后无需重跑攻击即可刷新历史结果。
    """
    from .metrics import calculate_batch_metrics

    results_file = Path(results_path)
    if not results_file.exists() or results_file.stat().st_size == 0:
        return

    records = json.loads(results_file.read_text(encoding="utf-8"))
    updated = 0
    skipped = 0
    for record in records:
        reference_text = record.get("reference_text", [])
        attack = record.get("attack", {}) or {}
        reconstructed_text = attack.get("reconstructed_text", [])
        # 缺少参考文本或重建结果，无法重算指标
        if not reference_text:
            skipped += 1
            continue
        metrics = calculate_batch_metrics(reference_text, reconstructed_text)
        record["metrics"] = metrics
        updated += 1

    if updated:
        results_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Metrics] 已重新计算 {updated} 条记录的指标（跳过 {skipped} 条缺少参考文本的记录）。")


def rebuild_summary_from_results(results_path: str, summary_path: str) -> None:
    """从 results.json 重建 summary.json。

    每次运行末尾调用，用 results.json 中的全部记录重新生成 summary.json，
    保证 summary 始终是 results 的准确派生视图。
    缺少 model_name/peft_method/batch_size 的旧记录会被跳过（重跑后会回填）。
    """
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
        # 跳过缺少必要字段的旧记录（重跑后会回填，再重建即可纳入）
        if not all(v is not None for v in [model_name, peft_method, method_name, batch_size]) or not metrics:
            skipped += 1
            continue

        data_name = str(record.get("data_name", ""))
        key = (str(model_name), str(peft_method), str(method_name).lower(), int(batch_size), data_name)
        details = metrics.get("sentence_details", [])
        # 计算该batch正确重建的token总数（所有句子的correct_tokens求和）
        batch_recon_token_count = sum(float(d.get("correct_tokens", 0.0)) for d in details)
        grouped.setdefault(key, []).append(
            {
                "f1": float(metrics.get("mean_f1_score", 0.0)),
                "rouge_l": float(metrics.get("mean_rouge_l", 0.0)),
                "rouge_1": float(metrics.get("mean_rouge_1", 0.0)),
                "rouge_2": float(metrics.get("mean_rouge_2", 0.0)),
                "token_accuracy": float(metrics.get("mean_token_accuracy", 0.0)),
                "recon_token_count": batch_recon_token_count,
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
                "mean_rouge_1": _mean([r["rouge_1"] for r in group]),
                "mean_rouge_2": _mean([r["rouge_2"] for r in group]),
                "mean_token_accuracy": _mean([r["token_accuracy"] for r in group]),
                "mean_recon_token_count": _mean([r["recon_token_count"] for r in group]),
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
        print(f"[Summary] 跳过 {skipped} 条缺少 model_name/peft_method/batch_size 的旧记录，重跑后会自动回填。")


class ResultSummaryWriter:
    """Maintain final method-level averages from per-batch attack metrics."""

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
            # 兼容旧字段名
            if "mean_recon_token_count" not in row:
                if "mean_correct_token_count" in row:
                    row["mean_recon_token_count"] = row.pop("mean_correct_token_count")
                elif "mean_reconstructed_token_count" in row:
                    row["mean_recon_token_count"] = row.pop("mean_reconstructed_token_count")
            # 兼容旧结果：补齐 data_name 字段（缺失则置空，由 save() 进一步处理）
            row.setdefault("data_name", "")
        return rows

    def append_batch(self, cfg: Dict[str, Any], method_name: str, metrics: Dict[str, Any]) -> None:
        details = metrics.get("sentence_details", [])
        # 计算该batch正确重建的token总数（所有句子的correct_tokens求和）
        batch_recon_token_count = sum(float(item.get("correct_tokens", 0.0)) for item in details)
        self.batch_records.append(
            {
                "model_name": _model_display_name(cfg.get("model", {}).get("name_or_path", "")),
                "peft_method": cfg.get("selected_peft", "partial"),
                "reconstruction_method": method_name,
                "batch_size": int(cfg.get("data", {}).get("batch_size", 1)),
                "data_name": cfg.get("data", {}).get("name") or cfg.get("data", {}).get("path") or "",
                "f1": float(metrics.get("mean_f1_score", 0.0)),
                "rouge_l": float(metrics.get("mean_rouge_l", 0.0)),
                "rouge_1": float(metrics.get("mean_rouge_1", 0.0)),
                "rouge_2": float(metrics.get("mean_rouge_2", 0.0)),
                "token_accuracy": float(metrics.get("mean_token_accuracy", 0.0)),
                "recon_token_count": batch_recon_token_count,
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
                    "mean_rouge_1": _mean([record["rouge_1"] for record in records]),
                    "mean_rouge_2": _mean([record["rouge_2"] for record in records]),
                    "mean_token_accuracy": _mean([record["token_accuracy"] for record in records]),
                    "mean_recon_token_count": _mean([record["recon_token_count"] for record in records]),
                }
            )

        # 加载旧行并兼容回填：
        #  - 旧行缺少 data_name（记为空串）；
        #  - 若本次运行已产出同 (model, peft, method, batch_size) 的新行，则旧行视为已被覆盖，丢弃；
        #  - 否则保留旧行（未重跑的历史数据），data_name 留空。
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
                    # 旧行已被本次新行覆盖，丢弃
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
            "mean_rouge_1",
            "mean_rouge_2",
            "mean_token_accuracy",
            "mean_correct_token_count",
        ]
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)