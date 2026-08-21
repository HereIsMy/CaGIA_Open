from __future__ import annotations

from typing import Dict

from .gradient_matching import gradient_matching_attack


def _resolve_gradient_keyword(uploaded_gradients: Dict, peft_config: Dict, requested_keyword):
    candidates = []
    if requested_keyword:
        candidates.append(str(requested_keyword))
    candidates.extend(str(keyword) for keyword in peft_config.get("train_keywords", []) if keyword)
    target_module = peft_config.get("target_module")
    if target_module:
        candidates.append(str(target_module))

    method = str(peft_config.get("method", "partial")).lower()
    defaults = {
        "adapter": "adapter",
        "lora": "lora_layer",
        "partial": "attn.c_attn",
        "mlp": "mlp.c_fc",
    }
    if method in defaults:
        candidates.append(defaults[method])

    seen = set()
    ordered_candidates = []
    for keyword in candidates:
        if keyword not in seen:
            seen.add(keyword)
            ordered_candidates.append(keyword)

    for keyword in ordered_candidates:
        if any(keyword in name for name in uploaded_gradients):
            return keyword
    return requested_keyword


def run_attack(global_model, uploaded_gradients, peft_config, tokenizer, **kwargs) -> Dict:
    kwargs.setdefault("gradient_norm", "l2")
    kwargs.setdefault("l1_weight", kwargs.get("alpha", 0.0))
    kwargs["gradient_keyword"] = _resolve_gradient_keyword(
        uploaded_gradients,
        peft_config,
        kwargs.get("gradient_keyword"),
    )

    result = gradient_matching_attack(global_model, uploaded_gradients, peft_config, tokenizer, kwargs, "Partial-Gradient")
    result["partial_gradient_keyword"] = kwargs.get("gradient_keyword")
    result["layer_indices"] = kwargs.get("layer_indices")
    return result
