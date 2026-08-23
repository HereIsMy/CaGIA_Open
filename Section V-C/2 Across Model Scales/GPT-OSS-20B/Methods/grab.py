from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from Utils.gradients import trainable_named_parameters


def _as_logits(outputs):
    return outputs.logits if hasattr(outputs, "logits") else outputs


def _gradient_objective(logits, labels, model, params, target_grads, alpha: float):
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    objective = torch.zeros((), device=logits.device)
    for grad, target in zip(grads, target_grads):
        if grad is None or grad.shape != target.shape:
            continue
        diff = grad - target
        objective = objective + torch.norm(diff, p=2) + alpha * torch.norm(diff, p=1)
    return objective


def _attention_mask(ids: Sequence[Sequence[int]], pad_token_id: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([[0 if token == pad_token_id else 1 for token in seq] for seq in ids], device=device)


def _length_attention_mask(batch_size: int, seq_len: int, lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([[1 if pos < lengths[row] else 0 for pos in range(seq_len)] for row in range(batch_size)], device=device)


def _decode(tokenizer, ids: Sequence[Sequence[int]]) -> List[str]:
    return [tokenizer.decode(list(seq), skip_special_tokens=True) for seq in ids]


def _tokenize_guess(embedding: torch.Tensor, token_embedding: torch.Tensor) -> List[List[int]]:
    norm_guess = F.normalize(embedding, dim=-1)
    norm_vocab = F.normalize(token_embedding, dim=-1)
    scores = torch.matmul(norm_guess, norm_vocab.T)
    return scores.argmax(dim=-1).detach().cpu().tolist()


def _evaluate_discrete_solution(model, token_ids, labels, params, target_grads, tokenizer, alpha: float, device: torch.device) -> float:
    ids = torch.tensor(token_ids, device=device, dtype=torch.long)
    mask = _attention_mask(token_ids, getattr(tokenizer, "pad_token_id", 0) or 0, device)
    logits = _as_logits(model(input_ids=ids, attention_mask=mask, use_cache=False))
    return -float(_gradient_objective(logits, labels, model, params, target_grads, alpha).detach().item())


def _beam_refine(
    model,
    initial_ids: List[List[int]],
    labels,
    params,
    target_grads,
    tokenizer,
    separate_tokens: List[List[int]],
    individual_lengths: Sequence[int],
    alpha: float,
    beam: int,
    num_iters: int,
    num_perms: int,
    device: torch.device,
) -> Tuple[List[List[int]], float]:
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    batch_size = len(initial_ids)
    seq_len = len(initial_ids[0]) if initial_ids else 0
    population = [initial_ids]
    scores = [_evaluate_discrete_solution(model, initial_ids, labels, params, target_grads, tokenizer, alpha, device)]

    for _ in range(num_perms):
        permuted = []
        for row, seq in enumerate(initial_ids):
            valid = list(seq[: individual_lengths[row]])
            if len(valid) > 1:
                order = torch.randperm(len(valid)).tolist()
                valid = [valid[index] for index in order]
            permuted.append((valid + [pad_id] * seq_len)[:seq_len])
        population.append(permuted)
        scores.append(_evaluate_discrete_solution(model, permuted, labels, params, target_grads, tokenizer, alpha, device))

    top = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[: max(1, beam)]
    beams = [population[index] for index in top]
    beam_scores = [scores[index] for index in top]

    for _ in range(num_iters):
        for sample_idx in range(batch_size):
            length = int(individual_lengths[sample_idx])
            for pos in range(1, max(1, length - 1)):
                candidates: List[List[List[int]]] = []
                candidate_scores: List[float] = []
                for solution in beams:
                    for token in [tk for tk in separate_tokens[sample_idx] if tk != pad_id]:
                        candidate = [list(seq) for seq in solution]
                        candidate[sample_idx][pos] = token
                        candidates.append(candidate)
                        candidate_scores.append(_evaluate_discrete_solution(model, candidate, labels, params, target_grads, tokenizer, alpha, device))
                if candidate_scores:
                    top = sorted(range(len(candidate_scores)), key=lambda index: candidate_scores[index], reverse=True)[: max(1, beam)]
                    beams = [candidates[index] for index in top]
                    beam_scores = [candidate_scores[index] for index in top]
    best = max(range(len(beam_scores)), key=lambda index: beam_scores[index])
    return beams[best], beam_scores[best]


def run_attack(global_model, uploaded_gradients, peft_config, tokenizer, **kwargs) -> Dict:
    device = kwargs.pop("device")
    iterations = int(kwargs.get("iterations", 300))
    restarts = int(kwargs.get("outer_iterations", kwargs.get("restarts", 5)))
    seq_len = int(kwargs.get("seq_len", kwargs.get("max_length", 16)))
    batch_size = int(kwargs.get("batch_size", 1))
    lr = float(kwargs.get("lr", 0.05))
    alpha = float(kwargs.get("alpha", 0.01))
    num_labels = int(kwargs.get("num_labels", 2))
    beta = float(kwargs.get("embedding_regularization", 0.0))
    simreg = float(kwargs.get("similarity_regularization", 100.0))
    beam = int(kwargs.get("beam", 4))
    discrete_iters = int(kwargs.get("discrete_iterations", 5))
    num_perms = int(kwargs.get("num_permutations", 2000))
    init_size = int(kwargs.get("init_size", 2000))
    individual_lengths = kwargs.get("individual_lengths") or [seq_len] * batch_size
    token_set = kwargs.get("token_set") or list(range(len(tokenizer)))
    params_with_names = trainable_named_parameters(global_model)
    params = [p for _name, p in params_with_names]
    target_grads = [uploaded_gradients[name].detach().to(device) for name, _p in params_with_names if name in uploaded_gradients]
    params = [p for name, p in params_with_names if name in uploaded_gradients]
    token_embedding = global_model.get_input_embeddings().weight.detach().to(device)
    attention_mask = _length_attention_mask(batch_size, seq_len, individual_lengths, device)
    avg_embedding_norm = token_embedding.norm(p=2, dim=1).mean()
    best_score = -float("inf")
    best_ids: List[List[int]] = []
    best_continuous_ids: List[List[int]] = []

    global_model.eval()
    discrete_solution = kwargs.get("discrete_solution")
    previous_discrete_solution = None
    accumulated_separate_tokens: List[List[int]] = [[] for _ in range(batch_size)]
    for _ in range(restarts):
        if discrete_solution is None:
            candidates = []
            scores = []
            for _init in range(init_size):
                valid_len = max(individual_lengths)
                emb = torch.randn(batch_size, valid_len, token_embedding.shape[1])
                pad_len = seq_len - valid_len
                if pad_len > 0:
                    pad = token_embedding[getattr(tokenizer, "pad_token_id", 0) or 0].detach().cpu().repeat(batch_size, pad_len, 1)
                    emb = torch.cat((emb, pad), dim=1)
                emb = emb / torch.norm(emb, dim=2, keepdim=True).clamp_min(1e-12) * avg_embedding_norm.detach().cpu()
                candidates.append(emb)
            label_logits = torch.randn(batch_size, num_labels, device=device)
            for emb in candidates:
                score = _evaluate_discrete_solution(global_model, _tokenize_guess(emb.to(device), token_embedding), label_logits, params, target_grads, tokenizer, alpha, device)
                scores.append(score)
            dummy_embedding = candidates[max(range(len(scores)), key=lambda index: scores[index])].to(device).requires_grad_(True)
        else:
            ids = torch.tensor(discrete_solution, device=device, dtype=torch.long)
            dummy_embedding = global_model.get_input_embeddings()(ids).detach().clone().requires_grad_(True)
            label_logits = best_labels.detach().clone() if "best_labels" in locals() else torch.randn(batch_size, num_labels, device=device)
        label_logits = label_logits.detach().clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([dummy_embedding, label_logits], lr=lr)
        for _step in range(iterations):
            optimizer.zero_grad()
            logits = _as_logits(global_model(inputs_embeds=dummy_embedding, attention_mask=attention_mask, use_cache=False))
            objective = _gradient_objective(logits, label_logits, global_model, params, target_grads, alpha)
            objective = objective + beta * (dummy_embedding.norm(p=2, dim=2).mean() - avg_embedding_norm).square()
            if simreg and batch_size > 1:
                means = F.normalize(dummy_embedding.mean(dim=1), dim=-1)
                objective = objective + simreg * torch.triu(torch.matmul(means, means.T), diagonal=1).mean()
            objective.backward()
            optimizer.step()
            label_logits.data.clamp_(0, 1)
            with torch.no_grad():
                pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
                for row, length in enumerate(individual_lengths):
                    if length < seq_len:
                        dummy_embedding[row, length:, :] = token_embedding[pad_id]
        with torch.no_grad():
            continuous_ids = _tokenize_guess(dummy_embedding, token_embedding)
            hard_labels = label_logits.detach().clone()
        for row, ids in enumerate(continuous_ids):
            accumulated_separate_tokens[row] = list(set(accumulated_separate_tokens[row] + list(set(ids[: individual_lengths[row]]))))
            if not accumulated_separate_tokens[row]:
                accumulated_separate_tokens[row] = token_set[: individual_lengths[row]]
        continuous_score = _evaluate_discrete_solution(global_model, continuous_ids, hard_labels, params, target_grads, tokenizer, alpha, device)
        discrete_ids, discrete_score = _beam_refine(
            global_model, continuous_ids, hard_labels, params, target_grads, tokenizer, accumulated_separate_tokens, individual_lengths, alpha,
            beam, discrete_iters, num_perms, device
        )
        if max(continuous_score, discrete_score) > best_score:
            best_score = max(continuous_score, discrete_score)
            best_continuous_ids = continuous_ids
            best_ids = discrete_ids if discrete_score > continuous_score else continuous_ids
            best_labels = hard_labels
        if discrete_ids == continuous_ids or discrete_ids == previous_discrete_solution:
            break
        previous_discrete_solution = discrete_ids
        discrete_solution = discrete_ids

    return {
        "method": "GRAB",
        "reconstructed_text": _decode(tokenizer, best_ids),
        "token_ids": best_ids,
        "continuous_token_ids": best_continuous_ids,
        "objective": best_score,
        "optimization": "continuous_label_optimization_plus_discrete_beam_search",
        "status": "ok" if best_ids else "empty_solution",
    }
