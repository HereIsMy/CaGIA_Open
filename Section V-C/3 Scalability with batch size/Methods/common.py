from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from Utils.gradients import TensorDict
from Utils.hooks import register_attention_input_hooks, remove_hooks, iter_transformer_blocks, get_module_by_path
from Utils.linear_algebra import batch_tensor, compute_all_residuals_preprocessed, find_expressible_vectors, find_expressible_vectors_preprocessed, generate_combinations, preprocess_matrix, reduce_full_rank_columns, select_by_error_ratio, top_k_by_error
from Utils.memory_manager import get_gpu_memory_info, clear_gpu_cache
from Utils.peft_config import get_peft_method_config


def _is_linear_module(model: torch.nn.Module, module_path: str) -> bool:
    try:
        for block in iter_transformer_blocks(model):
            module = get_module_by_path(block, module_path)
            return isinstance(module, torch.nn.Linear)
    except Exception:
        pass
    return False


@dataclass
class AttackContext:
    global_model: torch.nn.Module
    uploaded_gradients: TensorDict
    peft_config: Dict
    tokenizer: object
    device: torch.device


def _slice_gradient_matrix(grad: torch.Tensor, projection: str | None) -> torch.Tensor:
    if grad.ndim != 2 or not projection:
        return grad
    if grad.shape[0] <= 0 or grad.shape[1] != 3 * grad.shape[0]:
        return grad
    hidden = grad.shape[0]
    if projection == "q":
        return grad[:, :hidden]
    if projection == "k":
        return grad[:, hidden : 2 * hidden]
    if projection == "v":
        return grad[:, 2 * hidden : 3 * hidden]
    return grad


def _prepare_gradient(grad: torch.Tensor, projection: str | None, transpose: bool) -> torch.Tensor:
    if grad.ndim != 2:
        return grad
    g = grad.detach()
    if transpose:
        g = g.T
    return _slice_gradient_matrix(g, projection)


def gradient_matrices(uploaded_gradients: TensorDict, keyword: str | None = None, projection: str | None = None, transpose: bool = False) -> List[torch.Tensor]:
    matrices = []
    for name, grad in uploaded_gradients.items():
        if keyword and keyword not in name:
            continue
        if grad.ndim >= 2:
            matrices.append(_prepare_gradient(grad, projection, transpose))
    return matrices


def gradient_matrices_by_keywords(uploaded_gradients: TensorDict, keywords: Sequence[str], projection: str | None = None, transpose: bool = False) -> List[torch.Tensor]:
    for keyword in keywords:
        matrices = gradient_matrices(uploaded_gradients, keyword, projection, transpose)
        if matrices:
            return matrices
    return []


def _reduce_full_rank_matrices(matrices: List[torch.Tensor], peft_method: str, cfg: Dict) -> List[torch.Tensor]:
    if peft_method.lower() != "partial":
        return matrices
    max_removal = int(cfg.get("reduce_full_rank_neurons", 1))
    if max_removal <= 0:
        return matrices
    reduced_matrices = []
    total_removed = 0
    for matrix in matrices:
        reduced, removed = reduce_full_rank_columns(matrix, max_removal)
        total_removed += removed
        reduced_matrices.append(reduced)
    if total_removed > 0:
        pass
    return reduced_matrices


def default_linear_gradient_keyword(peft_method: str, gradient_keyword: str | None) -> str | None:
    if gradient_keyword is not None:
        return gradient_keyword
    method = peft_method.lower()
    if method in {"adapter", "lora"}:
        return "adapter_down.weight"
    return None


def resolve_linear_gradient_keyword(peft_method: str, gradient_keyword: str | None) -> str | None:
    method = peft_method.lower()
    if method == "adapter" and gradient_keyword in {None, "adapter"}:
        return "adapter_down.weight"
    if method == "lora" and gradient_keyword in {None, "lora_layer"}:
        return "adapter_down.weight"
    return default_linear_gradient_keyword(peft_method, gradient_keyword)


def linear_gradient_keywords(peft_method: str, gradient_keyword: str | None, hook_module: str | None) -> List[str]:
    resolved = resolve_linear_gradient_keyword(peft_method, gradient_keyword)
    candidates = [resolved, gradient_keyword, hook_module]
    if peft_method.lower() == "partial":
        candidates.extend(["attn.c_attn", "c_attn"])
    return [str(item) for item in dict.fromkeys(item for item in candidates if item)]


def _matrix_for_candidate_dim(matrix: torch.Tensor, candidate_dim: int) -> torch.Tensor | None:
    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    if matrix.shape[0] == candidate_dim:
        return matrix
    if matrix.ndim == 2 and matrix.shape[1] == candidate_dim:
        return matrix.T
    return None


def _tensor_cache_key(tensor: torch.Tensor) -> tuple:
    base = tensor._base if getattr(tensor, "_base", None) is not None else tensor
    return (int(base.data_ptr()), tuple(tensor.shape), str(tensor.device), str(tensor.dtype), tuple(tensor.stride()))


def _estimate_sequence_batch_size(
    seq_len: int,
    hidden_dim: int,
    layer_count: int,
    preferred_batch_size: int,
    max_batch_size: int,
    min_batch_size: int,
    target_memory_usage: float,
    safety_margin: float,
    min_free_memory_gb: float,
    memory_estimate_multiplier: float,
    verbose: bool = False,
) -> int:
    if not torch.cuda.is_available() or hidden_dim <= 0 or layer_count <= 0:
        return preferred_batch_size
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    total_memory = total_bytes / (1024 ** 3)
    free_memory = free_bytes / (1024 ** 3)
    if total_memory <= 0 or free_memory <= 0:
        return preferred_batch_size
    target_free_gb = max(total_memory * (1.0 - target_memory_usage), total_memory * safety_margin, min_free_memory_gb)
    usable_gb = max(0.0, free_memory - target_free_gb)
    bytes_per_sample = max(1, seq_len) * hidden_dim * max(1, layer_count) * 4 * memory_estimate_multiplier
    estimated = int((usable_gb * (1024 ** 3)) / bytes_per_sample) if bytes_per_sample else preferred_batch_size
    batch_size = max(min_batch_size, min(max_batch_size, estimated if estimated > 0 else preferred_batch_size))
    if verbose:
        pass
    return batch_size


def _dynamic_scan_batches(
    tensor: torch.Tensor,
    seq_len: int,
    hidden_dim: int,
    layer_count: int,
    preferred_batch_size: int,
    max_batch_size: int,
    min_batch_size: int,
    enable_dynamic_batch: bool,
    target_memory_usage: float,
    safety_margin: float,
    min_free_memory_gb: float = 2.0,
    memory_estimate_multiplier: float = 12.0,
    verbose: bool = False,
):
    start = 0
    current = preferred_batch_size
    total = tensor.shape[0]
    while start < total:
        if enable_dynamic_batch:
            current = _estimate_sequence_batch_size(
                seq_len,
                hidden_dim,
                layer_count,
                current,
                max_batch_size,
                min_batch_size,
                target_memory_usage,
                safety_margin,
                min_free_memory_gb,
                memory_estimate_multiplier,
                verbose,
            )
        end = min(start + current, total)
        yield tensor[start:end]
        start = end


def effective_single_tokens(model: torch.nn.Module, tokenizer, device: torch.device, batch_size: int = 1024, epsilon: float = 1e-20) -> List[List[int]]:
    vocab = torch.arange(len(tokenizer), device=device)
    effective: List[List[int]] = []
    for chunk in batch_tensor(vocab, batch_size):
        ids = chunk.unsqueeze(1)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            try:
                embeds = model.get_input_embeddings()(ids)
            except AttributeError:
                embeds = model.transformer.wte(ids)
        keep = torch.abs(embeds).sum(dim=-1).squeeze(1) > epsilon
        effective.extend([[int(token)] for token in chunk[keep].tolist()])
    return effective


def decode_token_sequences(tokenizer, sequences: Sequence[Sequence[int]]) -> List[str]:
    return [tokenizer.decode(list(seq), clean_up_tokenization_spaces=True, skip_special_tokens=True) for seq in sequences]


def _has_consecutive_duplicate_tail(seq: Sequence[int]) -> bool:
    return len(seq) >= 2 and seq[-1] == seq[-2]


def _collect_candidates(model, token_batches, store, module_path: str, device: torch.device, token_index):
    was_training = model.training
    model.eval()

    def is_oom(error: RuntimeError) -> bool:
        return "out of memory" in str(error).lower()

    def forward_batch(ids: torch.Tensor):
        try:
            store.clear()
            store.token_index = token_index
            attention_mask = torch.ones_like(ids, device=device)
            with torch.no_grad():
                model(input_ids=ids.to(device), attention_mask=attention_mask)
            if not store.values:
                return
            selected = list(store.values)
            activations = torch.stack(selected, dim=0)
            if activations.ndim == 3:
                activations = activations.unsqueeze(2)
            yield ids, activations
        except RuntimeError as error:
            if not is_oom(error) or ids.shape[0] <= 1:
                raise
            clear_gpu_cache(False)
            mid = max(1, ids.shape[0] // 2)
            yield from forward_batch(ids[:mid])
            yield from forward_batch(ids[mid:])

    try:
        for ids in token_batches:
            yield from forward_batch(ids)
    finally:
        if was_training:
            model.train()


def linear_reconstruct(
    ctx: AttackContext,
    cfg: Dict,
    optimized: bool = False,
    position_candidates: List[List[int]] | None = None,
) -> Dict:
    peft_method = ctx.peft_config.get("method", "partial")

    hook_module = cfg.get("hook_module")
    gradient_keyword = cfg.get("gradient_keyword")

    if hook_module is None or gradient_keyword is None:
        try:
            peft_method_config = get_peft_method_config(peft_method)
            default_config = peft_method_config["default_config"]

            if hook_module is None:
                hook_module = default_config.get("target_module", "attn.c_attn")

            if gradient_keyword is None:
                gradient_keyword = resolve_linear_gradient_keyword(peft_method, None)
                if gradient_keyword is None:
                    train_keywords = default_config.get("train_keywords", [])
                    gradient_keyword = train_keywords[0] if train_keywords else "attn.c_attn"
        except Exception:
            if hook_module is None:
                hook_module = "attn.c_attn"
            if gradient_keyword is None:
                gradient_keyword = resolve_linear_gradient_keyword(peft_method, None) or "attn.c_attn"
    gradient_keywords = linear_gradient_keywords(peft_method, gradient_keyword, hook_module)

    projection = cfg.get("gradient_projection")
    transpose_gradient = _is_linear_module(ctx.global_model, hook_module)
    matrices = gradient_matrices_by_keywords(ctx.uploaded_gradients, gradient_keywords, projection, transpose=transpose_gradient)
    layer_indices = cfg.get("layer_indices")
    if layer_indices is not None:
        selected = [int(index) for index in layer_indices]
        matrices = [matrices[index] for index in selected if 0 <= index < len(matrices)]
    if not matrices:
        return {"reconstructed_text": [], "token_ids": [], "status": "no_gradient_matrix"}

    matrices = _reduce_full_rank_matrices(matrices, peft_method, cfg)

    max_len = int(cfg.get("max_length", 16))
    combo_batch_size = int(cfg.get("combination_batch_size", 4096))
    first_batch_size = int(cfg.get("first_batch_size", 8192))
    max_scan_batch_size = int(cfg.get("max_scan_batch_size", max(first_batch_size, combo_batch_size, 65536)))
    min_scan_batch_size = int(cfg.get("min_scan_batch_size", 512))
    chunk_size = int(cfg.get("chunk_size", 2048))
    tol = float(cfg.get("tolerance", 1e-2))
    top_k = int(cfg.get("top_k", 200))
    threshold = float(cfg.get("error_ratio_threshold", 5.0))
    eos_id = int(cfg.get("eos_token_id", getattr(ctx.tokenizer, "eos_token_id", 0) or 0))
    device = ctx.device

    enable_dynamic_batch = cfg.get("enable_dynamic_batch", True)
    target_memory_usage = cfg.get("target_memory_usage", 0.85)
    safety_margin = cfg.get("safety_margin", 0.1)
    min_free_memory_gb = float(cfg.get("min_free_memory_gb", 2.0))
    memory_estimate_multiplier = float(cfg.get("memory_estimate_multiplier", 12.0))
    verbose_memory = cfg.get("verbose_memory", False)

    clear_gpu_cache(verbose_memory)

    if verbose_memory:
        total_memory, used_memory, free_memory = get_gpu_memory_info()

    store, handles = register_attention_input_hooks(ctx.global_model, hook_module)
    try:
        token_pool = position_candidates or effective_single_tokens(ctx.global_model, ctx.tokenizer, device, int(cfg.get("vocab_batch_size", 1024)), float(cfg.get("embedding_epsilon", 1e-20)))
        preprocessed = {}

        def get_preprocessed(layer_idx: int, matrix: torch.Tensor) -> Dict:
            cache_key = (layer_idx, _tensor_cache_key(matrix))
            if cache_key not in preprocessed:
                preprocessed[cache_key] = preprocess_matrix(matrix)
            return preprocessed[cache_key]

        def match_candidates(layer_idx: int, matrix: torch.Tensor, candidates: torch.Tensor):
            return find_expressible_vectors_preprocessed(get_preprocessed(layer_idx, matrix), candidates, tol, chunk_size)

        hidden_hint = max((max(matrix.shape) for matrix in matrices if matrix.ndim >= 2), default=1)
        layer_hint = max(1, len(matrices))
        first = generate_combinations(token_pool, [[eos_id]], device)
        layer_hits: List[List] = [[] for _ in matrices]
        all_token_residuals: Dict[int, float] = {} if not optimized else None

        first_batches = _dynamic_scan_batches(
            first,
            seq_len=2,
            hidden_dim=hidden_hint,
            layer_count=layer_hint,
            preferred_batch_size=first_batch_size,
            max_batch_size=max_scan_batch_size,
            min_batch_size=max(min_scan_batch_size, 1024),
            enable_dynamic_batch=enable_dynamic_batch,
            target_memory_usage=target_memory_usage,
            safety_margin=safety_margin,
            min_free_memory_gb=min_free_memory_gb,
            memory_estimate_multiplier=memory_estimate_multiplier,
            verbose=verbose_memory,
        )
        for ids, acts in _collect_candidates(ctx.global_model, first_batches, store, hook_module, device, 0):
            for layer_idx, matrix in enumerate(matrices[: acts.shape[0]]):
                aligned_matrix = _matrix_for_candidate_dim(matrix, acts[layer_idx].shape[-1])
                if aligned_matrix is None:
                    continue
                hits = match_candidates(layer_idx, aligned_matrix, acts[layer_idx])
                for row, _col, error in hits:
                    layer_hits[layer_idx].append((int(ids[row, 0].item()), error))
                if optimized:
                    layer_hits[layer_idx] = top_k_by_error(layer_hits[layer_idx], top_k)
                if all_token_residuals is not None:
                    residuals = compute_all_residuals_preprocessed(get_preprocessed(layer_idx, aligned_matrix), acts[layer_idx], chunk_size)
                    if residuals is not None:
                        for i in range(residuals.shape[0]):
                            token_id = int(ids[i, 0].item())
                            err = float(residuals[i].item())
                            prev = all_token_residuals.get(token_id)
                            if prev is None or err < prev:
                                all_token_residuals[token_id] = err
        if all_token_residuals is not None and all_token_residuals:
            sorted_residuals = sorted(all_token_residuals.items(), key=lambda x: x[1])
            top_n = sorted_residuals[:100]
        if optimized:
            first_tokens = select_by_error_ratio([top_k_by_error(h, top_k) for h in layer_hits], threshold)
        else:
            first_tokens = [token for hits in layer_hits for token, _error in hits]
        possible = [[int(t)] for t in dict.fromkeys(first_tokens)]
        delayed: List[List[int]] = []
        completed: List[List[int]] = []

        prefix_group_size = max(1, int(cfg.get("prefix_group_size", 8)))

        def extend_prefixes(prefixes: List[List[int]]) -> Dict[tuple, List[List[int]]]:
            active_prefixes = [prefix for prefix in prefixes if len(prefix) < max_len]
            if not active_prefixes:
                return {}
            combos = generate_combinations(active_prefixes, token_pool, device)
            batches = _dynamic_scan_batches(
                combos,
                seq_len=len(active_prefixes[0]) + 1,
                hidden_dim=hidden_hint,
                layer_count=layer_hint,
                preferred_batch_size=combo_batch_size,
                max_batch_size=max_scan_batch_size,
                min_batch_size=min_scan_batch_size,
                enable_dynamic_batch=enable_dynamic_batch,
                target_memory_usage=target_memory_usage,
                safety_margin=safety_margin,
                min_free_memory_gb=min_free_memory_gb,
                memory_estimate_multiplier=memory_estimate_multiplier,
                verbose=verbose_memory,
            )
            next_hits_by_prefix: Dict[tuple, List[List]] = {tuple(prefix): [[] for _ in matrices] for prefix in active_prefixes}
            for ids, acts in _collect_candidates(ctx.global_model, batches, store, hook_module, device, -1):
                for layer_idx, matrix in enumerate(matrices[: acts.shape[0]]):
                    if layer_idx == 0 and len(matrices) > 1:
                        continue
                    aligned_matrix = _matrix_for_candidate_dim(matrix, acts[layer_idx].shape[-1])
                    if aligned_matrix is None:
                        continue
                    hits = match_candidates(layer_idx, aligned_matrix, acts[layer_idx])
                    for row, _col, error in hits:
                        seq = ids[row].tolist()
                        prefix_key = tuple(seq[:-1])
                        if prefix_key in next_hits_by_prefix:
                            next_hits_by_prefix[prefix_key][layer_idx].append((seq, error))
                    if optimized:
                        for prefix_key in next_hits_by_prefix:
                            next_hits_by_prefix[prefix_key][layer_idx] = top_k_by_error(
                                next_hits_by_prefix[prefix_key][layer_idx], top_k
                            )
            if enable_dynamic_batch:
                clear_gpu_cache(verbose_memory)
            extended: Dict[tuple, List[List[int]]] = {}
            for prefix_key, layer_hits_for_prefix in next_hits_by_prefix.items():
                if optimized:
                    next_tokens = select_by_error_ratio([top_k_by_error(h, top_k) for h in layer_hits_for_prefix], threshold)
                else:
                    next_tokens = [seq for hits in layer_hits_for_prefix for seq, _error in hits]
                extended[prefix_key] = [list(seq) for seq in dict.fromkeys(tuple(x) for x in next_tokens)]
            return extended

        while possible or delayed:
            current = possible[:]
            current_delayed = delayed[:]
            possible = []
            delayed = []

            pending = current_delayed + current
            for prefix in pending:
                if len(prefix) >= max_len:
                    completed.append(prefix)
            extendable = [prefix for prefix in pending if len(prefix) < max_len]

            for group_start in range(0, len(extendable), prefix_group_size):
                group = extendable[group_start : group_start + prefix_group_size]
                group_next = extend_prefixes(group)
                for prefix in group:
                    unique_next = group_next.get(tuple(prefix), [])
                    if not unique_next:
                        completed.append(prefix)
                    elif optimized or len(unique_next) == 1:
                        possible.extend(unique_next)
                    else:
                        delayed.extend(unique_next)
            possible = [list(seq) for seq in dict.fromkeys(tuple(x) for x in possible)]
            delayed = [list(seq) for seq in dict.fromkeys(tuple(x) for x in delayed)]
            if possible or delayed:
                candidates = possible + delayed
                longest = max(candidates, key=len)
                extra = len(candidates) - 1
                suffix = f" (另有 {extra} 条候选)" if extra > 0 else ""
                print(f"[CaGIA {len(longest)}/{max_len}]{suffix} {longest}", flush=True)

        return {
            "reconstructed_text": decode_token_sequences(ctx.tokenizer, completed),
            "token_ids": completed,
            "gradient_projection": projection or "full",
            "status": "ok",
        }
    finally:
        remove_hooks(handles)


def dager_reconstruct(ctx: AttackContext, cfg: Dict) -> Dict:
    peft_method = ctx.peft_config.get("method", "partial")
    hook_module = cfg.get("hook_module") or ctx.peft_config.get("target_module", "attn.c_attn")
    gradient_keyword = cfg.get("gradient_keyword")
    if gradient_keyword is None:
        train_keywords = ctx.peft_config.get("train_keywords")
        gradient_keyword = train_keywords[0] if train_keywords else default_linear_gradient_keyword(peft_method, None) or hook_module
    gradient_keywords = linear_gradient_keywords(peft_method, gradient_keyword, hook_module)

    projection = cfg.get("gradient_projection")
    transpose_gradient = _is_linear_module(ctx.global_model, hook_module)
    matrices = gradient_matrices_by_keywords(ctx.uploaded_gradients, gradient_keywords, projection, transpose=transpose_gradient)
    if not matrices:
        return {"reconstructed_text": [], "token_ids": [], "status": "no_gradient_matrix"}
    if len(matrices) < 2:
        return {"reconstructed_text": [], "token_ids": [], "status": "insufficient_gradient_layers"}

    matrices = _reduce_full_rank_matrices(matrices, peft_method, cfg)

    device = ctx.device
    max_len = int(cfg.get("max_length", 16))
    guess_len = int(cfg.get("guess_max_length", cfg.get("guess_max_tok_len", 32)))
    vocab_batch_size = int(cfg.get("vocab_batch_size", 512))
    combination_batch_size = int(cfg.get("combination_batch_size", 1024))
    chunk_size = int(cfg.get("chunk_size", 2048))
    tol = float(cfg.get("tolerance", 1e-2))
    embedding_epsilon = float(cfg.get("embedding_epsilon", 1e-20))

    store, handles = register_attention_input_hooks(ctx.global_model, hook_module)
    try:
        effective_tokens = effective_single_tokens(ctx.global_model, ctx.tokenizer, device, vocab_batch_size, embedding_epsilon)
        vocab = torch.tensor([tok[0] for tok in effective_tokens], device=device)
        position_candidates: List[List[int]] = [[] for _ in range(guess_len)]

        for chunk in batch_tensor(vocab, vocab_batch_size):
            ids = chunk.unsqueeze(1).repeat(1, guess_len)
            for batch_ids, acts in _collect_candidates(ctx.global_model, [ids], store, hook_module, device, slice(None)):
                matrix = _matrix_for_candidate_dim(matrices[0], acts[0].shape[-1])
                if matrix is None:
                    continue
                for row, col, _error in find_expressible_vectors(matrix, acts[0], tol, chunk_size):
                    if col < guess_len:
                        position_candidates[col].append(int(batch_ids[row, col].item()))

        position_candidates = [list(dict.fromkeys(tokens)) for tokens in position_candidates if tokens]
        if not position_candidates:
            return {"reconstructed_text": [], "token_ids": [], "position_candidates": [], "status": "no_position_candidates"}

        unified_token_set: List[int] = []
        for tokens in position_candidates:
            for token in tokens:
                if token not in unified_token_set:
                    unified_token_set.append(token)

        possible: List[List[int]] = [[token] for token in unified_token_set]
        delayed: List[List[int]] = []
        completed: List[List[int]] = []
        link_layer = 1

        for pos in range(max_len - 1):
            next_tokens = [[token] for token in unified_token_set]
            current = possible[:]
            current_delayed = delayed[:]
            possible = []
            delayed = []

            for prefix in current_delayed + current:
                combos = generate_combinations([prefix], next_tokens, device)
                hits_for_prefix: List[List[int]] = []
                for ids, acts in _collect_candidates(ctx.global_model, batch_tensor(combos, combination_batch_size), store, hook_module, device, -1):
                    matrix = _matrix_for_candidate_dim(matrices[link_layer], acts[link_layer].shape[-1])
                    if matrix is None:
                        continue
                    for row, _col, _error in find_expressible_vectors(matrix, acts[link_layer], tol, chunk_size):
                        hits_for_prefix.append(ids[row].tolist())
                hits_for_prefix = [list(seq) for seq in dict.fromkeys(tuple(x) for x in hits_for_prefix)]
                valid_hits = [seq for seq in hits_for_prefix if not _has_consecutive_duplicate_tail(seq)]
                if not valid_hits:
                    completed.append(prefix)
                elif len(valid_hits) == 1:
                    possible.extend(valid_hits)
                else:
                    delayed.extend(valid_hits)
                if pos == max_len - 2 and valid_hits:
                    completed.extend(valid_hits)

            possible = [list(seq) for seq in dict.fromkeys(tuple(x) for x in possible)]
            delayed = [list(seq) for seq in dict.fromkeys(tuple(x) for x in delayed)]
            completed = [list(seq) for seq in dict.fromkeys(tuple(x) for x in completed)]

        completed.extend(possible)
        completed = [seq[:max_len] for seq in dict.fromkeys(tuple(x) for x in completed)]
        return {
            "reconstructed_text": decode_token_sequences(ctx.tokenizer, completed),
            "token_ids": completed,
            "position_candidates": position_candidates,
            "unified_token_set": unified_token_set,
            "gradient_projection": projection or "full",
            "status": "ok",
        }
    finally:
        remove_hooks(handles)
