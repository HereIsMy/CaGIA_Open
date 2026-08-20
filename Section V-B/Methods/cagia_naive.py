from __future__ import annotations

from typing import Dict

from .common import AttackContext, linear_reconstruct


def run_attack(global_model, uploaded_gradients, peft_config, tokenizer, **kwargs) -> Dict:
    ctx = AttackContext(global_model, uploaded_gradients, peft_config, tokenizer, kwargs.pop("device"))
    cfg = dict(kwargs)
    cfg.setdefault("gradient_projection", "v")
    result = linear_reconstruct(ctx, cfg, optimized=False)
    result["method"] = "CaGIA-Naive"
    return result