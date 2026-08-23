from __future__ import annotations

import types
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from .hooks import first_existing_module_path, get_module_by_path, iter_transformer_blocks
from .gradients import is_transformer_layer_parameter


MODEL_MODULE_ALIASES = {
    "attention_input": ["attn.c_attn", "self_attn.q_proj"],
    "attention_output": ["attn.c_proj", "self_attn.o_proj"],
    "mlp_input": ["mlp.c_fc", "mlp.gate_proj", "mlp.up_proj"],
    "mlp_output": ["mlp.c_proj", "mlp.down_proj"],
}


MODULE_ALIAS_BY_NAME = {
    "attn.c_attn": "attention_input",
    "c_attn": "attention_input",
    "q_proj": "attention_input",
    "self_attn.q_proj": "attention_input",
    "attn.c_proj": "attention_output",
    "self_attn.o_proj": "attention_output",
    "o_proj": "attention_output",
    "mlp.c_fc": "mlp_input",
    "c_fc": "mlp_input",
    "mlp.gate_proj": "mlp_input",
    "gate_proj": "mlp_input",
    "mlp.up_proj": "mlp_input",
    "up_proj": "mlp_input",
    "mlp.c_proj": "mlp_output",
    "mlp.down_proj": "mlp_output",
    "down_proj": "mlp_output",
}


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden)


class BottleneckAdapter(nn.Module):
    def __init__(self, in_features: int, bottleneck: int, out_features: int | None = None, activation: bool = True):
        super().__init__()
        out_features = out_features or in_features
        self.adapter_down = nn.Linear(in_features, bottleneck)
        self.adapter_up = nn.Linear(bottleneck, out_features)
        self.act = nn.ReLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter_up(self.act(self.adapter_down(x)))


def _hidden_size(model: torch.nn.Module) -> int:
    cfg = model.config
    hidden = getattr(cfg, "n_embd", None)
    if hidden is None:
        hidden = getattr(cfg, "hidden_size")
    return int(hidden)


def resolve_layer_range(model: torch.nn.Module, layer_start: int, layer_end: int) -> tuple[int, int]:
    blocks = iter_transformer_blocks(model)
    total_layers = len(blocks)
    start = max(0, int(layer_start))
    end = total_layers if int(layer_end) < 0 else min(int(layer_end), total_layers)
    if end < start:
        end = start
    return start, end


def resolve_target_module_path(model: torch.nn.Module, target_path: str) -> str:
    blocks = iter_transformer_blocks(model)
    if not blocks:
        raise AttributeError("Cannot resolve target module without transformer blocks.")
    alias = MODULE_ALIAS_BY_NAME.get(str(target_path), str(target_path))
    candidates = MODEL_MODULE_ALIASES.get(alias, [str(target_path)])
    return first_existing_module_path(blocks[0], candidates)


def _linear_out_features(module: nn.Module, hidden: int) -> int:
    if hasattr(module, "nf"):
        return int(module.nf)
    if hasattr(module, "out_features"):
        return int(module.out_features)
    if hasattr(module, "weight") and getattr(module.weight, "ndim", 0) >= 2:
        return int(module.weight.shape[0])
    return hidden


def _linear_in_features(module: nn.Module, hidden: int) -> int:
    if hasattr(module, "in_features"):
        return int(module.in_features)
    if hasattr(module, "weight") and getattr(module.weight, "ndim", 0) >= 2:
        if hasattr(module, "nf"):
            return int(module.weight.shape[0])
        return int(module.weight.shape[1])
    return hidden


def add_classification_head(model: torch.nn.Module, num_labels: int) -> None:
    if hasattr(model, "classification_head"):
        return
    model.classification_head = ClassificationHead(_hidden_size(model), num_labels).to(next(model.parameters()).device)
    original_forward = model.forward

    def forward_with_head(self, input_ids=None, *args, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        kwargs["output_hidden_states"] = True
        outputs = original_forward(input_ids=input_ids, *args, **kwargs)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        elif isinstance(outputs, (tuple, list)):
            if len(outputs) > 1 and isinstance(outputs[1], tuple):
                hidden = outputs[1][-1]
            else:
                hidden = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs
        if attention_mask is None:
            pooled = hidden[:, -1, :]
        else:
            last_indices = attention_mask.to(hidden.device).sum(dim=1).clamp(min=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_indices]
        logits = self.classification_head(pooled)
        return logits

    model.forward = types.MethodType(forward_with_head, model)


def insert_lora(model: torch.nn.Module, target_path: str, reduction_factor: float, layer_start: int, layer_end: int) -> None:
    hidden = _hidden_size(model)
    device = next(model.parameters()).device
    target_path = resolve_target_module_path(model, target_path)
    layer_start, layer_end = resolve_layer_range(model, layer_start, layer_end)
    for block in iter_transformer_blocks(model)[layer_start:layer_end]:
        module = get_module_by_path(block, target_path)
        in_features = _linear_in_features(module, hidden)
        if target_path.endswith("c_attn"):
            out_features = hidden * 3
        else:
            out_features = _linear_out_features(module, hidden)
        module.lora_layer = BottleneckAdapter(in_features, max(1, int(in_features / reduction_factor)), int(out_features), activation=False).to(device)
        module.forward_original = module.forward

        def forward_with_lora(self, hidden_states, *args, **kwargs):
            return self.forward_original(hidden_states, *args, **kwargs) + self.lora_layer(hidden_states)

        module.forward = types.MethodType(forward_with_lora, module)


def insert_adapters(model: torch.nn.Module, reduction_factor: float, layer_start: int, layer_end: int) -> None:
    hidden = _hidden_size(model)
    device = next(model.parameters()).device
    layer_start, layer_end = resolve_layer_range(model, layer_start, layer_end)
    for block in iter_transformer_blocks(model)[layer_start:layer_end]:
        block.adapter = BottleneckAdapter(hidden, max(1, int(hidden / reduction_factor)), hidden).to(device)
        block.forward_original = block.forward

        def forward_with_adapter(self, *args, **kwargs):
            output = self.forward_original(*args, **kwargs)
            hidden_states = output[0]
            return (hidden_states + self.adapter(hidden_states),) + tuple(output[1:])

        block.forward = types.MethodType(forward_with_adapter, block)


def configure_peft(model: torch.nn.Module, peft_cfg: Dict, num_labels: int) -> List[torch.nn.Parameter]:
    add_classification_head(model, num_labels)
    method = str(peft_cfg.get("method", "partial")).lower()

    try:
        from .peft_config import get_default_peft_config
        default_config = get_default_peft_config(method)
    except Exception:
        default_config = {}

    target_path = peft_cfg.get("target_module", default_config.get("target_module", "attn.c_attn"))
    reduction_factor = float(peft_cfg.get("reduction_factor", default_config.get("reduction_factor", 64)))
    layer_start = int(peft_cfg.get("layer_start", default_config.get("layer_start", 0)))
    layer_end = int(peft_cfg.get("layer_end", default_config.get("layer_end", 10**9)))
    train_keywords = peft_cfg.get("train_keywords", default_config.get("train_keywords"))
    resolved_target_path = resolve_target_module_path(model, target_path) if method in {"partial", "lora", "mlp"} else target_path

    peft_cfg["resolved_target_module"] = resolved_target_path
    peft_cfg["resolved_layer_start"], peft_cfg["resolved_layer_end"] = resolve_layer_range(model, layer_start, layer_end)

    if method == "lora":
        insert_lora(model, resolved_target_path, reduction_factor, layer_start, layer_end)
    elif method == "adapter":
        insert_adapters(model, reduction_factor, layer_start, layer_end)

    for param in model.parameters():
        param.requires_grad = False

    if train_keywords is None:
        train_keywords = {
            "lora": ["lora_layer", "classification_head"],
            "adapter": ["adapter", "classification_head"],
            "mlp": [resolved_target_path, "classification_head"],
            "partial": [resolved_target_path, "classification_head"],
        }[method]
    else:
        train_keywords = [resolved_target_path if str(keyword) == str(target_path) else keyword for keyword in train_keywords]
    peft_cfg["train_keywords"] = train_keywords

    trainable = []
    for name, param in model.named_parameters():
        if not any(keyword in name for keyword in train_keywords):
            continue
        if "classification_head" not in name:
            layer_start, layer_end = peft_cfg["resolved_layer_start"], peft_cfg["resolved_layer_end"]
            if not any(is_transformer_layer_parameter(name, index) for index in range(layer_start, layer_end)):
                continue
        if any(keyword in name for keyword in train_keywords):
            param.requires_grad = True
            trainable.append(param)
    return trainable


def load_model_and_tokenizer(model_name: str, num_labels: int, peft_cfg: Dict, device: torch.device):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_path = _resolve_model_name_or_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_name_lower = str(model_name).lower()
    model_kwargs = {}
    if "qwen" in model_name_lower or "llama" in model_name_lower:
        model_kwargs["attn_implementation"] = "eager"
        model_cls = AutoModelForCausalLM
    else:
        model_cls = AutoModel

    config = AutoConfig.from_pretrained(model_path)
    if "qwen" in model_name_lower or "llama" in model_name_lower:
        config._attn_implementation = "eager"
    if "14b" in model_name_lower and hasattr(config, "num_hidden_layers") and config.num_hidden_layers >= 40:
        original_layers = config.num_hidden_layers
        config.num_hidden_layers = original_layers // 2 - 1
        model_kwargs["config"] = config

    model = model_cls.from_pretrained(model_path, **model_kwargs).to(device)
    configure_peft(model, peft_cfg, num_labels)
    return model, tokenizer


def _resolve_model_name_or_path(name_or_path: str) -> str:
    path = Path(name_or_path).expanduser()
    if path.exists():
        return str(path)
    looks_like_path = any(sep in name_or_path for sep in ("/", "\\")) or ":" in name_or_path
    if looks_like_path:
        raise FileNotFoundError(
            "model.name_or_path points to a local path that does not exist: "
            f"{name_or_path}. Please set it to an existing model folder, or use a Hub id such as 'gpt2'."
        )
    return name_or_path
