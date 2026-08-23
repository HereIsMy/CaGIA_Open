
PEFT_METHODS_CONFIG = {
    "partial": {
        "name": "部分模块微调",
        "description": "只微调模型中的特定模块",
        "trainable_modules": {
            "attn.c_attn": {
                "name": "注意力输入投影",
                "description": "自注意力机制的输入投影层",
                "gradient_keyword": "attn.c_attn",
                "hook_module": "attn.c_attn",
                "parameters": {
                    "weight": True,
                    "bias": True
                }
            },
            "attn.c_proj": {
                "name": "注意力输出投影",
                "description": "自注意力机制的输出投影层",
                "gradient_keyword": "attn.c_proj",
                "hook_module": "attn.c_proj",
                "parameters": {
                    "weight": True,
                    "bias": True
                }
            },
            "mlp.c_fc": {
                "name": "MLP前馈层",
                "description": "多层感知机的前馈层",
                "gradient_keyword": "mlp.c_fc",
                "hook_module": "mlp.c_fc",
                "parameters": {
                    "weight": True,
                    "bias": True
                }
            },
            "mlp.c_proj": {
                "name": "MLP输出层",
                "description": "多层感知机的输出层",
                "gradient_keyword": "mlp.c_proj",
                "hook_module": "mlp.c_proj",
                "parameters": {
                    "weight": True,
                    "bias": True
                }
            },
            "classification_head": {
                "name": "分类头",
                "description": "最终分类层",
                "gradient_keyword": "classification_head",
                "hook_module": "classification_head",
                "parameters": {
                    "weight": True,
                    "bias": True
                }
            }
        },
        "default_config": {
            "target_module": "attn.c_attn",
            "train_keywords": ["attn.c_attn", "classification_head"],
            "layer_start": 0,
            "layer_end": -1
        }
    },

    "lora": {
        "name": "LoRA微调",
        "description": "低秩适应微调方法",
        "trainable_modules": {
            "lora.c_attn": {
                "name": "LoRA注意力输入投影",
                "description": "在注意力输入投影层添加LoRA适配器",
                "gradient_keyword": "lora_layer",
                "hook_module": "attn.c_attn",
                "parameters": {
                    "adapter_down.weight": True,
                    "adapter_down.bias": True,
                    "adapter_up.weight": True,
                    "adapter_up.bias": True
                },
                "lora_config": {
                    "in_features": "embedding_dim",
                    "adapter_size": "rank_factor",
                    "out_features": "embedding_dim*3"
                }
            },
            "lora.c_proj": {
                "name": "LoRA注意力输出投影",
                "description": "在注意力输出投影层添加LoRA适配器",
                "gradient_keyword": "lora_layer",
                "hook_module": "attn.c_proj",
                "parameters": {
                    "adapter_down.weight": True,
                    "adapter_down.bias": True,
                    "adapter_up.weight": True,
                    "adapter_up.bias": True
                },
                "lora_config": {
                    "in_features": "embedding_dim",
                    "adapter_size": "rank_factor",
                    "out_features": "embedding_dim"
                }
            },
            "lora.c_fc": {
                "name": "LoRA MLP前馈层",
                "description": "在MLP前馈层添加LoRA适配器",
                "gradient_keyword": "lora_layer",
                "hook_module": "mlp.c_fc",
                "parameters": {
                    "adapter_down.weight": True,
                    "adapter_down.bias": True,
                    "adapter_up.weight": True,
                    "adapter_up.bias": True
                },
                "lora_config": {
                    "in_features": "embedding_dim",
                    "adapter_size": "rank_factor",
                    "out_features": "embedding_dim*4"
                }
            },
            "lora.c_proj_mlp": {
                "name": "LoRA MLP输出层",
                "description": "在MLP输出层添加LoRA适配器",
                "gradient_keyword": "lora_layer",
                "hook_module": "mlp.c_proj",
                "parameters": {
                    "adapter_down.weight": True,
                    "adapter_down.bias": True,
                    "adapter_up.weight": True,
                    "adapter_up.bias": True
                },
                "lora_config": {
                    "in_features": "embedding_dim*4",
                    "adapter_size": "rank_factor",
                    "out_features": "embedding_dim"
                }
            },
            "classification_head": {
                "name": "分类头",
                "description": "最终分类层",
                "gradient_keyword": "classification_head",
                "hook_module": "classification_head",
                "parameters": {
                    "weight": True,
                    "bias": True
                }
            }
        },
        "default_config": {
            "target_module": "attn.c_attn",
            "train_keywords": ["lora_layer", "classification_head"],
            "reduction_factor": 64,
            "layer_start": 0,
            "layer_end": -1
        }
    },

    "adapter": {
        "name": "Adapter微调",
        "description": "在Transformer层后添加适配器",
        "trainable_modules": {
            "adapter": {
                "name": "Transformer层适配器",
                "description": "在每个Transformer层后添加适配器",
                "gradient_keyword": "adapter",
                "hook_module": "adapter",
                "parameters": {
                    "adapter_down.weight": True,
                    "adapter_down.bias": True,
                    "adapter_up.weight": True,
                    "adapter_up.bias": True
                },
                "adapter_config": {
                    "in_features": "embedding_dim",
                    "adapter_size": "embedding_dim/reduction_factor",
                    "out_features": "embedding_dim"
                }
            },
            "classification_head": {
                "name": "分类头",
                "description": "最终分类层",
                "gradient_keyword": "classification_head",
                "hook_module": "classification_head",
                "parameters": {
                    "weight": True,
                    "bias": True
                }
            }
        },
        "default_config": {
            "target_module": "adapter",
            "train_keywords": ["adapter", "classification_head"],
            "reduction_factor": 64,
            "layer_start": 0,
            "layer_end": -1
        }
    },

    "mlp": {
        "name": "MLP前馈层微调",
        "description": "只微调Transformer块中的MLP输入投影层",
        "trainable_modules": {
            "mlp.c_fc": {
                "name": "MLP前馈层",
                "description": "多层感知机的输入投影层",
                "gradient_keyword": "mlp.c_fc",
                "hook_module": "mlp.c_fc",
                "parameters": {
                    "weight": True,
                    "bias": True
                }
            },
            "classification_head": {
                "name": "分类头",
                "description": "最终分类层",
                "gradient_keyword": "classification_head",
                "hook_module": "classification_head",
                "parameters": {
                    "weight": True,
                    "bias": True
                }
            }
        },
        "default_config": {
            "target_module": "mlp.c_fc",
            "train_keywords": ["mlp.c_fc", "classification_head"],
            "layer_start": 0,
            "layer_end": -1
        }
    }
}


def get_peft_method_config(method_name: str) -> dict:
    method_name = method_name.lower()
    if method_name not in PEFT_METHODS_CONFIG:
        raise ValueError(f"未知的PEFT方法: {method_name}. 可用方法: {list(PEFT_METHODS_CONFIG.keys())}")

    return PEFT_METHODS_CONFIG[method_name]


def get_trainable_modules(method_name: str) -> dict:
    config = get_peft_method_config(method_name)
    return config["trainable_modules"]


def get_module_config(method_name: str, module_name: str) -> dict:
    trainable_modules = get_trainable_modules(method_name)

    if module_name not in trainable_modules:
        available_modules = list(trainable_modules.keys())
        raise ValueError(f"未知的模块: {module_name}. 可用模块: {available_modules}")

    return trainable_modules[module_name]


def get_gradient_keyword(method_name: str, module_name: str) -> str:
    module_config = get_module_config(method_name, module_name)
    return module_config["gradient_keyword"]


def get_hook_module(method_name: str, module_name: str) -> str:
    module_config = get_module_config(method_name, module_name)
    return module_config["hook_module"]


def get_default_peft_config(method_name: str) -> dict:
    config = get_peft_method_config(method_name)
    return config["default_config"]


def list_peft_methods() -> list:
    return list(PEFT_METHODS_CONFIG.keys())


def list_trainable_modules(method_name: str) -> list:
    trainable_modules = get_trainable_modules(method_name)
    return list(trainable_modules.keys())


def validate_peft_config(peft_config: dict) -> bool:
    if "method" not in peft_config:
        return False

    method_name = peft_config["method"]
    if method_name not in PEFT_METHODS_CONFIG:
        return False

    if "target_module" in peft_config:
        trainable_modules = get_trainable_modules(method_name)
        hook_modules = {module.get("hook_module") for module in trainable_modules.values()}
        target_module = peft_config["target_module"]
        if target_module not in trainable_modules and target_module not in hook_modules:
            return False

    return True


def get_peft_method_info(method_name: str) -> dict:
    config = get_peft_method_config(method_name)

    return {
        "name": config["name"],
        "description": config["description"],
        "trainable_modules": list_trainable_modules(method_name),
        "default_config": config["default_config"]
    }



_MODEL_PEFT_CONFIG_CACHE = None


def _load_model_peft_yaml() -> dict:
    global _MODEL_PEFT_CONFIG_CACHE
    if _MODEL_PEFT_CONFIG_CACHE is not None:
        return _MODEL_PEFT_CONFIG_CACHE
    from pathlib import Path
    yaml_path = Path(__file__).resolve().parent / "model_peft.yaml"
    if not yaml_path.exists():
        _MODEL_PEFT_CONFIG_CACHE = {}
        return _MODEL_PEFT_CONFIG_CACHE
    text = yaml_path.read_text(encoding="utf-8")
    try:
        import yaml
        data = yaml.safe_load(text) or {}
    except ImportError:
        from .config import _fallback_yaml_load
        data = _fallback_yaml_load(text)
    _MODEL_PEFT_CONFIG_CACHE = data
    return data


def load_model_peft_config(model_name: str, method_name: str) -> dict:
    from .config import _lookup_case_insensitive

    data = _load_model_peft_yaml()
    if not data:
        raise KeyError("model_peft.yaml 为空或不存在，无法解析模型 PEFT 配置。")
    model_cfg = _lookup_case_insensitive(data, str(model_name))
    return _lookup_case_insensitive(model_cfg, str(method_name))
