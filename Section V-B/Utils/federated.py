from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

from Methods import METHODS
from .data import collate_text_batch, load_dataset
from .gradients import apply_average_update, average_tensor_dicts, collect_gradients, filter_tensor_dict_by_transformer_layer, gradients_from_upload, snapshot_trainable_state, state_delta
from .model_utils import load_model_and_tokenizer
from .result_summary import rebuild_summary_from_results, recalculate_metrics_in_results
from .results import JsonResultWriter, make_experiment_key, make_record_meta


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_clients(dataset, num_clients: int) -> List[Subset]:
    indices = list(range(len(dataset)))
    shards = [indices[i::num_clients] for i in range(num_clients)]
    return [Subset(dataset, shard) for shard in shards]


def build_client_loaders(cfg: Dict):
    dataset = load_dataset(cfg)
    clients = split_clients(dataset, int(cfg["federated"].get("num_clients", 1)))
    return [
        DataLoader(
            client_data,
            batch_size=int(cfg["data"].get("batch_size", 1)),
            shuffle=bool(cfg["data"].get("shuffle", False)),
            collate_fn=collate_text_batch,
        )
        for client_data in clients
    ]


def move_batch_to_device(tokenized: Dict[str, torch.Tensor], labels: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    batch = {key: value.to(device) for key, value in tokenized.items()}
    batch["labels"] = labels.to(device)
    return batch


def forward_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    labels = batch.pop("labels")
    logits = model(**batch)
    batch["labels"] = labels
    return torch.nn.CrossEntropyLoss()(logits, labels)


def train_one_client(client_model, tokenizer, client_loader, cfg: Dict, device: torch.device):
    local_batches = int(cfg["federated"].get("local_batches", 1))
    lr = float(cfg["federated"].get("client_lr", 1e-5))
    optimizer = torch.optim.SGD([p for p in client_model.parameters() if p.requires_grad], lr=lr)
    # Match the original attack setting: keep the language model deterministic
    # while enabling gradients only on PEFT/classification modules.
    client_model.eval()
    for module in client_model.modules():
        if any(param.requires_grad for param in module.parameters(recurse=False)):
            module.train()
    before = snapshot_trainable_state(client_model)
    last_batch = None

    optimizer.zero_grad()
    for batch_idx, raw_batch in enumerate(client_loader):
        if batch_idx >= local_batches:
            break
        tokenized = tokenizer(
            raw_batch["sentence"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg["data"].get("max_length"),
        )
        batch = move_batch_to_device(tokenized, raw_batch["labels"], device)
        loss = forward_loss(client_model, batch)
        loss.backward()
        last_batch = raw_batch
        if local_batches > 1:
            optimizer.step()
            optimizer.zero_grad()

    if local_batches == 1:
        return collect_gradients(client_model), "gradient", last_batch
    return state_delta(before, client_model), "delta", last_batch


def collect_batch_gradient(client_model, tokenizer, raw_batch, cfg: Dict, device: torch.device):
    """Collect gradients for one batch without applying any model update.

    调用方负责在调用前从 CPU 快照恢复权重，调用后同样由调用方恢复。
    本函数只做 forward + backward，不修改模型权重。
    """
    client_model.eval()
    for module in client_model.modules():
        if any(param.requires_grad for param in module.parameters(recurse=False)):
            module.train()
    client_model.zero_grad(set_to_none=True)
    tokenized = tokenizer(
        raw_batch["sentence"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg["data"].get("max_length"),
    )
    batch = move_batch_to_device(tokenized, raw_batch["labels"], device)
    loss = forward_loss(client_model, batch)
    loss.backward()
    return collect_gradients(client_model), "gradient", raw_batch


def _get_active_peft_config(cfg: Dict) -> Dict:
    """根据selected_peft获取当前激活的PEFT配置"""
    if "_active_peft_config" in cfg:
        return cfg["_active_peft_config"]
    selected_peft = cfg.get("selected_peft", "partial")
    peft_config = dict(cfg.get("peft", {}).get(selected_peft, {}))
    peft_config["method"] = selected_peft
    # 根据模型名从 Utils/model_peft.yaml 注入 target_module 和 train_keywords，
    # 这样不同模型结构（GPT2 vs Llama3/Qwen2.5）会使用各自的模块路径，
    # 而 config.yaml 的 peft 部分只需保留 reduction_factor 等参数。
    model_name = cfg.get("model", {}).get("name")
    if model_name:
        from .peft_config import load_model_peft_config
        try:
            model_peft = load_model_peft_config(model_name, selected_peft)
            peft_config["target_module"] = model_peft["target_module"]
            peft_config["train_keywords"] = list(model_peft["train_keywords"])
        except KeyError:
            pass
    return peft_config


_BATCH_AWARE_GRADIENT_MATCHING_METHODS = {"grab", "dlg", "tag", "lamp", "partial-gradient", "partial_gradient", "partial"}


def run_attack_for_upload(method_name: str, global_model, upload, update_type: str, cfg: Dict, tokenizer, device: torch.device, writer: JsonResultWriter, round_idx: int, client_idx: int, batch_idx: int, raw_batch) -> None:
    import time
    from .metrics import calculate_batch_metrics, print_metrics_summary
    
    method = METHODS[method_name]
    attack_cfg = dict(cfg.get("methods", {}).get(method_name, {}))
    
    # 添加显存管理配置
    memory_cfg = cfg.get("memory", {})
    attack_cfg.update(memory_cfg)
    
    # 获取当前激活的PEFT配置
    peft_config = _get_active_peft_config(cfg)
    
    # 设置gradient_keyword，优先从peft_config获取
    if "gradient_keyword" not in attack_cfg:
        train_keywords = peft_config.get("train_keywords")
        if train_keywords:
            attack_cfg["gradient_keyword"] = train_keywords[0]
    
    # 设置hook_module，优先从peft_config获取
    if "hook_module" not in attack_cfg:
        attack_cfg["hook_module"] = peft_config.get("resolved_target_module", peft_config.get("target_module", "attn.c_attn"))
    
    gradients = gradients_from_upload(upload, float(cfg["federated"].get("client_lr", 1e-5)), update_type)
    transformer_layer_index = cfg.get("run", {}).get("transformer_layer_index")
    gradients = filter_tensor_dict_by_transformer_layer(gradients, transformer_layer_index)
    attack_cfg["transformer_layer_index"] = transformer_layer_index
    
    # 从 raw_batch 获取真实的 batch size 和文本长度
    # 对于梯度匹配/优化重建方法（GRAB, DLG, TAG, LAMP, Partial-Gradient），需要根据实际batch信息初始化
    if raw_batch is not None and "sentence" in raw_batch:
        sentences = raw_batch["sentence"]
        actual_batch_size = len(sentences)
        
        # 使用与训练时相同的 tokenizer 参数计算每条文本的真实长度
        # padding=True 会将所有文本 padding 到 batch 中最长文本的长度
        max_length = cfg["data"].get("max_length")
        tokenized = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        # 从 attention_mask 获取真实长度（包含特殊token，与训练时一致）
        actual_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
        # seq_len 应该是 batch 中最长文本的长度（与 padding 策略一致）
        actual_seq_len = max(actual_lengths)
        
        # GRAB/DLG/TAG/LAMP/Partial-Gradient 的优化变量形状必须与产生梯度的真实 batch 一致。
        # 这些方法配置中常有 batch_size: 1 作为默认值，若不覆盖会只初始化并解码一条文本。
        if method_name in _BATCH_AWARE_GRADIENT_MATCHING_METHODS or "batch_size" not in attack_cfg:
            attack_cfg["batch_size"] = actual_batch_size
            attack_cfg["seq_len"] = actual_seq_len  # 使用实际最长长度
            attack_cfg["individual_lengths"] = actual_lengths
            print(f"[Batch Info] 实际batch大小: {actual_batch_size}, seq_len: {actual_seq_len}, 文本长度: {actual_lengths}")
    
    # 打印方法名称
    print(f"\n{'=' * 60}")
    print(f"Running {method_name.upper()} attack...")
    if transformer_layer_index is not None:
        print(f"Using gradients from transformer layer index {int(transformer_layer_index)} only ({len(gradients)} tensors).")
    print(f"{'=' * 60}")
    
    # 统计运行时间
    start_time = time.time()
    
    result = method(
        global_model=global_model,
        uploaded_gradients=gradients,
        peft_config=peft_config,
        tokenizer=tokenizer,
        device=device,
        num_labels=int(cfg["model"].get("num_labels", 2)),
        **attack_cfg,
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 打印运行时间
    print(f"\n{method_name.upper()} 运行时间: {elapsed_time:.2f} 秒")
    
    # 计算评估指标
    metrics = None
    if raw_batch is not None and "sentence" in raw_batch:
        ref_sentences = raw_batch["sentence"]
        recon_sentences = result.get("reconstructed_text", [])
        
        # 计算评估指标
        metrics = calculate_batch_metrics(ref_sentences, recon_sentences)

        # 打印评估指标
        print_metrics_summary(metrics, method_name.upper())

    record = {
        "experiment_key": make_experiment_key(cfg, method_name, round_idx, client_idx, batch_idx),
        "round": round_idx,
        "client": client_idx,
        "batch_idx": batch_idx,
        "upload_type": update_type,
        "reference_text": [] if raw_batch is None else raw_batch["sentence"],
        "attack": result,
        "method_name": method_name,
        "elapsed_time": elapsed_time,
        "metrics": metrics,
    }
    # 写入 model_name/peft_method/batch_size/data_name，供 summary 重建使用
    record.update(make_record_meta(cfg))
    writer.append(record)


def run_fedsgd(cfg: Dict, device: torch.device) -> None:
    print("[DEBUG] Building client loaders...", flush=True)
    loaders = build_client_loaders(cfg)
    print(f"[DEBUG] Loaders built: {len(loaders)} clients", flush=True)
    num_labels = int(cfg["model"].get("num_labels", 2))
    peft_config = _get_active_peft_config(cfg)
    print(f"[DEBUG] PEFT config: {peft_config}", flush=True)

    # 只加载一个模型到 GPU，GPU 上始终只维护一个模型实例
    global_model, tokenizer = load_model_and_tokenizer(cfg["model"]["name_or_path"], num_labels, peft_config, device)
    cfg["_active_peft_config"] = peft_config

    # 在 CPU 上保存原始权重快照，用于每次梯度收集前恢复
    cpu_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
    print("[DEBUG] Model loaded (GPU). CPU snapshot saved.", flush=True)

    results_path = cfg.get("output", {}).get("path", "New/results/results.json")
    summary_path = cfg.get("output", {}).get("summary_path", "New/results/summary.json")

    # 每次运行时，先用最新指标算法重算已有结果的指标，不重跑实验
    recalculate_metrics_in_results(results_path)

    writer = JsonResultWriter(results_path)
    methods = [name.lower() for name in cfg.get("run", {}).get("methods", ["cagia-opt"])]
    attack_client = int(cfg.get("run", {}).get("attack_client", 0))
    reconstruction_batches = int(cfg.get("run", {}).get("reconstruction_batches", 1))
    print(f"[DEBUG] methods={methods}, attack_client={attack_client}, reconstruction_batches={reconstruction_batches}", flush=True)

    for round_idx in range(int(cfg["federated"].get("rounds", 1))):
        attack_loader = loaders[attack_client]
        for batch_idx, raw_batch in enumerate(attack_loader):
            if batch_idx >= reconstruction_batches:
                break
            print(f"[DEBUG] Processing batch {batch_idx + 1}/{reconstruction_batches}", flush=True)
            pending_methods = []
            record_meta = make_record_meta(cfg)
            for method_name in methods:
                experiment_key = make_experiment_key(cfg, method_name, round_idx, attack_client, batch_idx)
                if writer.has_completed(experiment_key, method_name, round_idx, attack_client, batch_idx, raw_batch, meta=record_meta):
                    print(f"[Skip] 已存在结果: round={round_idx}, client={attack_client}, batch={batch_idx}, method={method_name}")
                else:
                    pending_methods.append(method_name)

            if not pending_methods:
                print(f"[Skip] batch {batch_idx + 1}/{reconstruction_batches} 的所有方法均已完成，跳过梯度收集。")
                continue

            # 从 CPU 快照恢复权重，确保梯度收集使用原始模型
            global_model.load_state_dict(cpu_state)
            print("[DEBUG] Collecting batch gradient (forward+backward)...", flush=True)
            upload, upload_type, raw_batch = collect_batch_gradient(global_model, tokenizer, raw_batch, cfg, device)
            print(f"[DEBUG] Gradient collected. upload_type={upload_type}, #tensors={len(upload)}", flush=True)
            # 恢复原始权重，确保攻击阶段使用未修改的模型
            global_model.load_state_dict(cpu_state)
            print(f"\nReconstructing batch {batch_idx + 1}/{reconstruction_batches} with a fixed model.")
            for method_name in pending_methods:
                print(f"[DEBUG] Running attack: {method_name}", flush=True)
                run_attack_for_upload(method_name, global_model, upload, upload_type, cfg, tokenizer, device, writer, round_idx, attack_client, batch_idx, raw_batch)
                print(f"[DEBUG] Attack {method_name} done.", flush=True)

    # 每次运行末尾从 results.json 重建 summary.json，保证 summary 始终是 results 的派生视图
    if summary_path:
        rebuild_summary_from_results(cfg.get("output", {}).get("path", "New/results/results.json"), summary_path)
        print(f"[Summary] 已从 results 重建 {summary_path}")
