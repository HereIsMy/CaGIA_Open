from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


class ActivationStore:
    def __init__(self) -> None:
        self.values: List[torch.Tensor] = []
        # 指定只保存序列中哪个位置的激活，避免保存完整序列浪费显存。
        # None 表示保存完整序列；整数表示序列中对应索引（支持负数）。
        self.token_index = None

    def clear(self) -> None:
        self.values.clear()

    def hook(self, _module, inputs, _output) -> None:
        if inputs:
            x = inputs[0].detach()
            # 仅保存目标位置的激活 (batch, hidden_dim)，避免保存完整序列 (batch, seq_len, hidden_dim)
            if self.token_index is not None:
                x = x[:, self.token_index, :]
            self.values.append(x)


def iter_transformer_blocks(model: torch.nn.Module):
    candidates = [
        "h",
        "layers",
        "transformer.h",
        "model.layers",
        "encoder.layer",
        "encoder.layers",
        "decoder.layers",
        "transformer.encoder.layers",
        "transformer.decoder.layers",
    ]
    for attr in candidates:
        node = model
        ok = True
        for part in attr.split("."):
            if not hasattr(node, part):
                ok = False
                break
            node = getattr(node, part)
        if ok:
            return list(node)
    raise AttributeError("Cannot find transformer blocks; configure model_utils for this architecture.")


def get_module_by_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    if not path:
        return root
    node = root
    for part in path.split("."):
        if not part:
            continue
        node = getattr(node, part)
    return node


def has_module_path(root: torch.nn.Module, path: str) -> bool:
    try:
        get_module_by_path(root, path)
        return True
    except AttributeError:
        return False


def first_existing_module_path(root: torch.nn.Module, paths: Sequence[str]) -> str:
    for path in paths:
        if has_module_path(root, path):
            return path
    raise AttributeError(f"Cannot find any target module path in this block: {list(paths)}")


def register_attention_input_hooks(model: torch.nn.Module, module_path: str) -> Tuple[ActivationStore, List]:
    store = ActivationStore()
    handles = []
    for block in iter_transformer_blocks(model):
        module = get_module_by_path(block, module_path)
        handles.append(module.register_forward_hook(store.hook))
    return store, handles


def remove_hooks(handles: List) -> None:
    for handle in handles:
        handle.remove()
