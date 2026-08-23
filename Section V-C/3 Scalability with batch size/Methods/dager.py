from __future__ import annotations

from typing import Dict

from .common import AttackContext, dager_reconstruct


def run_attack(global_model, uploaded_gradients, peft_config, tokenizer, **kwargs) -> Dict:
    ctx = AttackContext(global_model, uploaded_gradients, peft_config, tokenizer, kwargs.pop("device"))
    cfg = dict(kwargs)
    cfg.setdefault("gradient_projection", "full")
    result = dager_reconstruct(ctx, cfg)
    result["method"] = "DAGER"
    return result
