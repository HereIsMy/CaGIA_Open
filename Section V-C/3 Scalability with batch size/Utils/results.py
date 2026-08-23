from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def _model_display_name(model_path: str) -> str:
    normalized = str(model_path).replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] if normalized else str(model_path)


def _json_default(value: Any) -> str:
    return str(value)


def make_record_meta(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {})
    fixed_length = data_cfg.get("fixed_length")
    return {
        "model_name": _model_display_name(cfg.get("model", {}).get("name_or_path", "")),
        "peft_method": cfg.get("selected_peft", "partial"),
        "batch_size": int(data_cfg.get("batch_size", 1)),
        "data_name": data_cfg.get("name") or data_cfg.get("path"),
        "fixed_length": int(fixed_length) if fixed_length is not None else None,
    }


def make_experiment_key(cfg: Dict[str, Any], method_name: str, round_idx: int, client_idx: int, batch_idx: int) -> str:
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    key_payload = {
        "model_name": _model_display_name(cfg.get("model", {}).get("name_or_path", "")),
        "data_name": data_cfg.get("name") or data_cfg.get("path"),
        "selected_peft": cfg.get("selected_peft", "partial"),
        "batch_size": int(data_cfg.get("batch_size", 1)),
        "max_length": data_cfg.get("max_length"),
        "fixed_length": data_cfg.get("fixed_length"),
        "method_name": method_name.lower(),
        "method_config": cfg.get("methods", {}).get(method_name, {}),
        "round": int(round_idx),
        "client": int(client_idx),
        "batch_idx": int(batch_idx),
        "transformer_layer_index": run_cfg.get("transformer_layer_index"),
    }
    encoded = json.dumps(key_payload, ensure_ascii=False, sort_keys=True, default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class JsonResultWriter:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        if self.path.exists() and self.path.stat().st_size > 0:
            self.records = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def _same_reference_text(record: Dict[str, Any], raw_batch) -> bool:
        if raw_batch is None or "sentence" not in raw_batch:
            return True
        return record.get("reference_text", []) == raw_batch["sentence"]

    def _backfill(self, record: Dict[str, Any], meta: Dict[str, Any]) -> None:
        changed = False
        for key, value in meta.items():
            if value and not record.get(key):
                record[key] = value
                changed = True
        if changed:
            self.path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")

    def has_completed(self, experiment_key: str, method_name: str, round_idx: int, client_idx: int, batch_idx: int, raw_batch=None, meta: Dict[str, Any] | None = None) -> bool:
        for record in self.records:
            if record.get("experiment_key") == experiment_key:
                if meta:
                    self._backfill(record, meta)
                return True
            if record.get("experiment_key") is not None:
                continue
            if str(record.get("method_name", "")).lower() != method_name.lower():
                continue
            if int(record.get("round", -1)) != int(round_idx):
                continue
            if int(record.get("client", -1)) != int(client_idx):
                continue
            if "batch_idx" in record and int(record.get("batch_idx", -1)) != int(batch_idx):
                continue
            if self._same_reference_text(record, raw_batch):
                if meta:
                    self._backfill(record, meta)
                return True
        return False

    def append(self, record: Dict[str, Any]) -> None:
        self.records.append(record)
        self.path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")


def recalculate_metrics_in_results(results_path: str, tokenizer=None) -> int:
    from .metrics import calculate_batch_metrics

    path = Path(results_path)
    if not path.exists() or path.stat().st_size == 0:
        return 0

    records = json.loads(path.read_text(encoding="utf-8"))
    updated = 0
    for record in records:
        ref_text = record.get("reference_text")
        attack = record.get("attack") or {}
        recon_text = attack.get("reconstructed_text")
        if not ref_text or recon_text is None:
            continue
        new_metrics = calculate_batch_metrics(ref_text, recon_text)
        record["metrics"] = new_metrics
        updated += 1

    if updated > 0:
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return updated
