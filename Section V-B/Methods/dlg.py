from __future__ import annotations

from typing import Dict

from .gradient_matching import gradient_matching_attack


def run_attack(global_model, uploaded_gradients, peft_config, tokenizer, **kwargs) -> Dict:
    kwargs.setdefault("gradient_norm", "l2")
    kwargs.setdefault("l1_weight", 0.0)
    return gradient_matching_attack(global_model, uploaded_gradients, peft_config, tokenizer, kwargs, "DLG")