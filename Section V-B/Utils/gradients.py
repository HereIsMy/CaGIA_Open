from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import torch


TensorDict = Dict[str, torch.Tensor]


_TRANSFORMER_LAYER_PATTERNS = [
    r"(?:^|\.)(?:h|layers|layer|blocks|block)\.{index}(?:\.|$)",
    r"(?:^|\.)encoder\.layer\.{index}(?:\.|$)",
    r"(?:^|\.)decoder\.layer\.{index}(?:\.|$)",
]


def trainable_named_parameters(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def collect_gradients(model: torch.nn.Module) -> TensorDict:
    grads: TensorDict = {}
    for name, param in trainable_named_parameters(model):
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def is_transformer_layer_parameter(name: str, layer_index: int) -> bool:
    escaped_index = re.escape(str(int(layer_index)))
    return any(re.search(pattern.format(index=escaped_index), name) for pattern in _TRANSFORMER_LAYER_PATTERNS)


def filter_tensor_dict_by_transformer_layer(tensors: TensorDict, layer_index: int | None) -> TensorDict:
    if layer_index is None:
        return tensors
    return {name: tensor for name, tensor in tensors.items() if is_transformer_layer_parameter(name, int(layer_index))}


def state_delta(before: TensorDict, model: torch.nn.Module) -> TensorDict:
    delta: TensorDict = {}
    current = model.state_dict()
    for name, old_value in before.items():
        if name in current:
            delta[name] = current[name].detach().clone() - old_value
    return delta


def snapshot_trainable_state(model: torch.nn.Module) -> TensorDict:
    state = model.state_dict()
    return {name: state[name].detach().clone() for name, _ in trainable_named_parameters(model) if name in state}


def average_tensor_dicts(updates: Iterable[TensorDict]) -> TensorDict:
    updates = list(updates)
    if not updates:
        return {}
    keys = set.intersection(*(set(update.keys()) for update in updates))
    return {key: torch.stack([update[key] for update in updates], dim=0).mean(dim=0) for key in keys}


def apply_average_update(model: torch.nn.Module, avg_update: TensorDict, lr: float, update_type: str) -> None:
    with torch.no_grad():
        named_params = dict(model.named_parameters())
        for name, update in avg_update.items():
            if name not in named_params:
                continue
            if update_type == "gradient":
                named_params[name].add_(update.to(named_params[name].device), alpha=-lr)
            else:
                named_params[name].add_(update.to(named_params[name].device))


def gradients_from_upload(upload: TensorDict, lr: float, update_type: str) -> TensorDict:
    if update_type == "gradient":
        return upload
    scale = -1.0 / max(lr, 1e-12)
    return {name: value * scale for name, value in upload.items()}
