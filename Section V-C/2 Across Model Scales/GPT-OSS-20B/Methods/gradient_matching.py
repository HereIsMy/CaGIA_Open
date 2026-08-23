from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from Utils.gradients import trainable_named_parameters


def _as_logits(outputs):
    return outputs.logits if hasattr(outputs, "logits") else outputs


def _decode(tokenizer, ids: Sequence[Sequence[int]]) -> List[str]:
    return [tokenizer.decode(list(seq), skip_special_tokens=True) for seq in ids]


def _nearest_tokens(embeddings: torch.Tensor, embedding_weight: torch.Tensor) -> List[List[int]]:
    norm_guess = F.normalize(embeddings, dim=-1)
    norm_vocab = F.normalize(embedding_weight, dim=-1)
    return torch.matmul(norm_guess, norm_vocab.T).argmax(dim=-1).detach().cpu().tolist()


def _length_mask(batch_size: int, seq_len: int, lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([[1 if pos < lengths[row] else 0 for pos in range(seq_len)] for row in range(batch_size)], device=device)


def _selected_named_parameters(model: torch.nn.Module, uploaded_gradients: Dict[str, torch.Tensor], cfg: Dict) -> List[Tuple[str, torch.nn.Parameter]]:
    names = [name for name, _p in trainable_named_parameters(model)]
    keyword = cfg.get("gradient_keyword") or cfg.get("target_keyword")
    selected_names = [name for name in names if name in uploaded_gradients]

    common_names = set(names) & set(uploaded_gradients.keys())

    if keyword:
        matching_names = [name for name in selected_names if str(keyword) in name]
        selected_names = matching_names
    layer_indices = cfg.get("layer_indices")
    if layer_indices is not None:
        markers = [f".{int(index)}." for index in layer_indices]
        selected_names = [name for name in selected_names if any(marker in name for marker in markers)]
    params = dict(model.named_parameters())
    return [(name, params[name]) for name in selected_names if name in params]


def _gradient_distance(grads: Iterable[torch.Tensor | None], targets: Iterable[torch.Tensor], norm: str, l1_weight: float) -> torch.Tensor:
    loss = None
    for grad, target in zip(grads, targets):
        if grad is None or grad.shape != target.shape:
            continue
        diff = grad - target
        if norm == "cosine":
            term = 1 - F.cosine_similarity(grad.flatten(), target.flatten(), dim=0)
        else:
            term = torch.norm(diff, p=2)
        if l1_weight:
            term = term + l1_weight * torch.norm(diff, p=1)
        loss = term if loss is None else loss + term
    if loss is None:
        raise RuntimeError("No uploaded gradients matched trainable model parameters.")
    return loss


def _lm_perplexity(texts: Sequence[str], lm_model, lm_tokenizer, device: torch.device) -> List[float]:
    if lm_model is None or lm_tokenizer is None:
        return [0.0 for _ in texts]
    scores = []
    lm_model.eval()
    for text in texts:
        encoded = lm_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = lm_model(**encoded, labels=encoded["input_ids"], use_cache=False)
        scores.append(float(torch.exp(out.loss.detach()).item()))
    return scores


def _candidate_variants(ids: List[List[int]], pad_id: int, max_candidates: int) -> List[List[List[int]]]:
    candidates = [[list(seq) for seq in ids]]
    for row, seq in enumerate(ids):
        valid = [token for token in seq if token != pad_id]
        if len(valid) < 2:
            continue
        for pos in range(len(valid) - 1):
            swapped = [list(item) for item in ids]
            new_seq = list(valid)
            new_seq[pos], new_seq[pos + 1] = new_seq[pos + 1], new_seq[pos]
            swapped[row] = (new_seq + [pad_id] * len(seq))[: len(seq)]
            candidates.append(swapped)
            if len(candidates) >= max_candidates:
                return candidates
        moved = [list(item) for item in ids]
        new_seq = valid[1:] + valid[:1]
        moved[row] = (new_seq + [pad_id] * len(seq))[: len(seq)]
        candidates.append(moved)
        if len(candidates) >= max_candidates:
            return candidates
    return candidates


def _score_discrete_ids(model, token_ids, labels, params, targets, tokenizer, cfg: Dict, device: torch.device) -> float:
    ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    mask = torch.tensor([[0 if token == pad_id else 1 for token in seq] for seq in token_ids], device=device)
    logits = _as_logits(model(input_ids=ids, attention_mask=mask, use_cache=False))
    task_loss = torch.nn.CrossEntropyLoss()(logits, labels)
    grads = torch.autograd.grad(task_loss, params, create_graph=False, allow_unused=True)
    loss = _gradient_distance(grads, targets, str(cfg.get("gradient_norm", "l2")), float(cfg.get("l1_weight", 0.0)))
    return float(loss.detach().item())


def gradient_matching_attack(global_model, uploaded_gradients, peft_config, tokenizer, cfg: Dict, method_name: str) -> Dict:
    device = cfg.pop("device")
    batch_size = int(cfg.get("batch_size", 1))
    seq_len = int(cfg.get("seq_len", cfg.get("max_length", 16)))
    iterations = int(cfg.get("iterations", 300))
    restarts = int(cfg.get("restarts", cfg.get("outer_iterations", 3)))
    lr = float(cfg.get("lr", 0.05))
    num_labels = int(cfg.get("num_labels", 2))
    label_candidates = cfg.get("labels") or list(range(num_labels))
    embedding_reg = float(cfg.get("embedding_regularization", 0.0))
    discrete_steps = int(cfg.get("discrete_steps", 0))
    discrete_candidates = int(cfg.get("discrete_candidates", 16))
    lm_weight = float(cfg.get("lm_weight", 0.0))
    individual_lengths = cfg.get("individual_lengths") or [seq_len] * batch_size
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    params_with_names = _selected_named_parameters(global_model, uploaded_gradients, cfg)
    if not params_with_names:
        return {"method": method_name, "reconstructed_text": [], "token_ids": [], "status": "no_matching_gradients"}
    params = [param for _name, param in params_with_names]
    targets = [uploaded_gradients[name].detach().to(device) for name, _param in params_with_names]
    embedding_weight = global_model.get_input_embeddings().weight.detach().to(device)
    avg_embedding_norm = embedding_weight.norm(p=2, dim=1).mean()
    attention_mask = _length_mask(batch_size, seq_len, individual_lengths, device)

    lm_model = cfg.get("lm_model")
    lm_tokenizer = cfg.get("lm_tokenizer")
    if cfg.get("lm_name_or_path") and lm_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        lm_tokenizer = AutoTokenizer.from_pretrained(cfg["lm_name_or_path"])
        lm_model = AutoModelForCausalLM.from_pretrained(cfg["lm_name_or_path"]).to(device)

    global_model.eval()
    best = {"score": float("inf"), "ids": [], "continuous_ids": [], "label": None}
    for label_value in label_candidates:
        labels = torch.full((batch_size,), int(label_value), dtype=torch.long, device=device)
        for _restart in range(restarts):
            dummy = torch.randn(batch_size, seq_len, embedding_weight.shape[1], device=device)
            dummy = dummy / dummy.norm(dim=-1, keepdim=True).clamp_min(1e-12) * avg_embedding_norm
            dummy.requires_grad_(True)
            optimizer = torch.optim.Adam([dummy], lr=lr)
            for _step in range(iterations):
                optimizer.zero_grad()
                logits = _as_logits(global_model(inputs_embeds=dummy, attention_mask=attention_mask, use_cache=False))
                task_loss = torch.nn.CrossEntropyLoss()(logits, labels)
                grads = torch.autograd.grad(task_loss, params, create_graph=True, allow_unused=True)
                objective = _gradient_distance(grads, targets, str(cfg.get("gradient_norm", "l2")), float(cfg.get("l1_weight", 0.0)))
                if embedding_reg:
                    objective = objective + embedding_reg * (dummy.norm(p=2, dim=-1).mean() - avg_embedding_norm).square()
                objective.backward()
                optimizer.step()
                with torch.no_grad():
                    for row, length in enumerate(individual_lengths):
                        if length < seq_len:
                            dummy[row, length:, :] = embedding_weight[pad_id]

            ids = _nearest_tokens(dummy.detach(), embedding_weight)
            score = _score_discrete_ids(global_model, ids, labels, params, targets, tokenizer, cfg, device)
            for _ in range(discrete_steps):
                variants = _candidate_variants(ids, pad_id, discrete_candidates)
                texts = _decode(tokenizer, [variant[0] for variant in variants]) if batch_size == 1 else [" ".join(_decode(tokenizer, variant)) for variant in variants]
                perplexities = _lm_perplexity(texts, lm_model, lm_tokenizer, device)
                scored = []
                for variant, ppl in zip(variants, perplexities):
                    grad_score = _score_discrete_ids(global_model, variant, labels, params, targets, tokenizer, cfg, device)
                    scored.append((grad_score + lm_weight * ppl, grad_score, variant))
                scored.sort(key=lambda item: item[0])
                if scored and scored[0][1] <= score:
                    score, ids = scored[0][1], scored[0][2]
                else:
                    break
            if score < best["score"]:
                best = {"score": score, "ids": ids, "continuous_ids": _nearest_tokens(dummy.detach(), embedding_weight), "label": int(label_value)}

    return {
        "method": method_name,
        "reconstructed_text": _decode(tokenizer, best["ids"]),
        "token_ids": best["ids"],
        "continuous_token_ids": best["continuous_ids"],
        "inferred_label": best["label"],
        "objective": best["score"],
        "optimization": "gradient_matching",
        "status": "ok" if best["ids"] else "empty_solution",
    }
