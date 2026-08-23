from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

from Methods import METHODS
from .data import collate_text_batch, load_dataset
from .gradients import apply_average_update, average_tensor_dicts, collect_gradients, filter_tensor_dict_by_transformer_layer, gradients_from_upload, snapshot_trainable_state, state_delta
from .model_utils import load_model_and_tokenizer
from .result_summary import rebuild_summary_from_results
from .results import JsonResultWriter, make_experiment_key, make_record_meta, recalculate_metrics_in_results


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


def build_client_loaders(cfg: Dict, dataset=None):
    if dataset is None:
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
    if "_active_peft_config" in cfg:
        return cfg["_active_peft_config"]
    selected_peft = cfg.get("selected_peft", "partial")
    peft_config = dict(cfg.get("peft", {}).get(selected_peft, {}))
    peft_config["method"] = selected_peft
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

    memory_cfg = cfg.get("memory", {})
    attack_cfg.update(memory_cfg)

    peft_config = _get_active_peft_config(cfg)

    if "gradient_keyword" not in attack_cfg:
        train_keywords = peft_config.get("train_keywords")
        if train_keywords:
            attack_cfg["gradient_keyword"] = train_keywords[0]

    if "hook_module" not in attack_cfg:
        attack_cfg["hook_module"] = peft_config.get("resolved_target_module", peft_config.get("target_module", "attn.c_attn"))

    gradients = gradients_from_upload(upload, float(cfg["federated"].get("client_lr", 1e-5)), update_type)
    transformer_layer_index = cfg.get("run", {}).get("transformer_layer_index")
    gradients = filter_tensor_dict_by_transformer_layer(gradients, transformer_layer_index)
    attack_cfg["transformer_layer_index"] = transformer_layer_index

    if raw_batch is not None and "sentence" in raw_batch:
        sentences = raw_batch["sentence"]
        actual_batch_size = len(sentences)

        max_length = cfg["data"].get("max_length")
        tokenized = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        actual_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
        actual_seq_len = max(actual_lengths)

        if method_name in _BATCH_AWARE_GRADIENT_MATCHING_METHODS or "batch_size" not in attack_cfg:
            attack_cfg["batch_size"] = actual_batch_size
            attack_cfg["seq_len"] = actual_seq_len
            attack_cfg["individual_lengths"] = actual_lengths

    print(f"\n{'=' * 60}")
    print(f"Running {method_name.upper()} attack...")
    if transformer_layer_index is not None:
        print(f"Using gradients from transformer layer index {int(transformer_layer_index)} only ({len(gradients)} tensors).")
    print(f"{'=' * 60}")

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

    print(f"\n{method_name.upper()} 运行时间: {elapsed_time:.2f} 秒")

    metrics = None
    if raw_batch is not None and "sentence" in raw_batch:
        ref_sentences = raw_batch["sentence"]
        recon_sentences = result.get("reconstructed_text", [])

        metrics = calculate_batch_metrics(ref_sentences, recon_sentences)

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
    record.update(make_record_meta(cfg))
    writer.append(record)


def run_fedsgd(cfg: Dict, device: torch.device) -> None:
    dataset = load_dataset(cfg)
    loaders = build_client_loaders(cfg, dataset=dataset)
    num_labels = int(cfg["model"].get("num_labels", 2))
    peft_config = _get_active_peft_config(cfg)

    global_model, tokenizer = load_model_and_tokenizer(cfg["model"]["name_or_path"], num_labels, peft_config, device)
    cfg["_active_peft_config"] = peft_config

    cpu_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}

    results_path = cfg.get("output", {}).get("path", "New/results/results.json")
    summary_path = cfg.get("output", {}).get("summary_path", "New/results/summary.json")


    writer = JsonResultWriter(results_path)
    methods = [name.lower() for name in cfg.get("run", {}).get("methods", ["cagia-opt"])]
    attack_client = int(cfg.get("run", {}).get("attack_client", 0))
    reconstruction_batches = int(cfg.get("run", {}).get("reconstruction_batches", 1))

    for round_idx in range(int(cfg["federated"].get("rounds", 1))):
        attack_loader = loaders[attack_client]
        for batch_idx, raw_batch in enumerate(attack_loader):
            if batch_idx >= reconstruction_batches:
                break
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

            global_model.load_state_dict(cpu_state)
            upload, upload_type, raw_batch = collect_batch_gradient(global_model, tokenizer, raw_batch, cfg, device)
            global_model.load_state_dict(cpu_state)
            print(f"\nReconstructing batch {batch_idx + 1}/{reconstruction_batches} with a fixed model.")
            for method_name in pending_methods:
                run_attack_for_upload(method_name, global_model, upload, upload_type, cfg, tokenizer, device, writer, round_idx, attack_client, batch_idx, raw_batch)

    if summary_path:
        rebuild_summary_from_results(cfg.get("output", {}).get("path", "New/results/results.json"), summary_path)
        print(f"[Summary] 已从 results 重建 {summary_path}")
