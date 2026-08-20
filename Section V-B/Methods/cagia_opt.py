from __future__ import annotations

from typing import Dict

from .common import AttackContext, linear_reconstruct


def run_attack(global_model, uploaded_gradients, peft_config, tokenizer, **kwargs) -> Dict:
    ctx = AttackContext(global_model, uploaded_gradients, peft_config, tokenizer, kwargs.pop("device"))
    cfg = dict(kwargs)
    cfg.setdefault("gradient_projection", "full")
    cfg.setdefault("tolerance", 100)
    cfg.setdefault("error_ratio_threshold", 5.0)
    cfg.setdefault("top_k", 200)
    result = linear_reconstruct(ctx, cfg, optimized=True)
    result["method"] = "CaGIA-Opt"
    return result