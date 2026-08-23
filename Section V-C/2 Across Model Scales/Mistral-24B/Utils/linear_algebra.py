from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def preprocess_matrix(matrix: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor | int]:
    a = matrix.detach().float()
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    original_device = a.device
    a_cpu = a.cpu()
    q, r = torch.linalg.qr(a_cpu, mode="reduced")
    singular_values = torch.linalg.svdvals(r.double())
    max_sv = singular_values.max().clamp_min(torch.finfo(singular_values.dtype).tiny)
    eps_machine = torch.finfo(a_cpu.dtype).eps
    rank_threshold = min(r.shape[0], r.shape[1]) * eps_machine * max_sv
    rank = int((singular_values > rank_threshold).sum().item())
    if rank <= 0:
        return {"rank": 0}
    return {
        "rank": rank,
        "q": q[:, :rank].to(original_device),
        "r": (r[:rank, :rank] + eps * torch.eye(rank)).to(original_device),
        "pinv": torch.linalg.pinv(a_cpu).to(original_device),
        "a": a,
    }


def find_expressible_vectors(matrix: torch.Tensor, candidates: torch.Tensor, tol: float, chunk_size: int = 2048) -> List[Tuple[int, int, float]]:
    return find_expressible_vectors_preprocessed(preprocess_matrix(matrix), candidates, tol, chunk_size)


def find_expressible_vectors_preprocessed(pre: Dict[str, torch.Tensor | int], candidates: torch.Tensor, tol: float, chunk_size: int = 2048) -> List[Tuple[int, int, float]]:
    if int(pre.get("rank", 0)) <= 0:
        return []
    b = candidates.detach().float()
    n1, n2, dim = b.shape
    flat = b.reshape(-1, dim)
    q = pre["q"]
    r = pre["r"]
    pinv = pre["pinv"]
    a = pre["a"]
    results: List[Tuple[int, int, float]] = []
    chunk = chunk_size
    for start in range(0, flat.shape[0], chunk):
        batch = flat[start : start + chunk]
        qtb = q.T @ batch.T
        qr_solution = torch.linalg.solve_triangular(r, qtb, upper=True)
        qr_recon = (q @ qr_solution).T
        pinv_solution = pinv @ batch.T
        pinv_recon = (a @ pinv_solution).T
        errors = torch.minimum(torch.linalg.norm(batch - qr_recon, dim=1), torch.linalg.norm(batch - pinv_recon, dim=1))
        for offset in torch.where(errors < tol)[0].tolist():
            idx = start + offset
            results.append((idx // n2, idx % n2, float(errors[offset].item())))
    return results


def compute_all_residuals_preprocessed(pre: Dict[str, torch.Tensor | int], candidates: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    if int(pre.get("rank", 0)) <= 0:
        return None
    b = candidates.detach().float()
    n1, n2, dim = b.shape
    flat = b.reshape(-1, dim)
    q = pre["q"]
    r = pre["r"]
    pinv = pre["pinv"]
    a = pre["a"]
    all_errors: List[torch.Tensor] = []
    chunk = chunk_size
    for start in range(0, flat.shape[0], chunk):
        batch = flat[start : start + chunk]
        qtb = q.T @ batch.T
        qr_solution = torch.linalg.solve_triangular(r, qtb, upper=True)
        qr_recon = (q @ qr_solution).T
        pinv_solution = pinv @ batch.T
        pinv_recon = (a @ pinv_solution).T
        errors = torch.minimum(torch.linalg.norm(batch - qr_recon, dim=1), torch.linalg.norm(batch - pinv_recon, dim=1))
        all_errors.append(errors)
    return torch.cat(all_errors, dim=0) if all_errors else None


def generate_combinations(prefixes: Sequence[Sequence[int]], suffixes: Sequence[Sequence[int]], device: torch.device) -> torch.Tensor:
    left = torch.tensor(prefixes, dtype=torch.long, device=device)
    right = torch.tensor(suffixes, dtype=torch.long, device=device)
    left = left.unsqueeze(1).repeat(1, right.shape[0], 1)
    right = right.unsqueeze(0).repeat(left.shape[0], 1, 1)
    return torch.cat((left, right), dim=2).reshape(-1, left.shape[-1] + right.shape[-1])


def batch_tensor(tensor: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    return [tensor[i : i + batch_size] for i in range(0, tensor.shape[0], batch_size)]


def top_k_by_error(items: List[Tuple[Sequence[int] | int, float]], k: int) -> List[Tuple[Sequence[int] | int, float]]:
    return sorted(items, key=lambda x: x[1])[:k]


def select_by_error_ratio(layer_results: List[List[Tuple[Sequence[int] | int, float]]], threshold: float) -> List[Sequence[int] | int]:
    selected = []
    for errors in layer_results:
        if len(errors) < 2:
            continue
        values = [err for _, err in errors]
        ratios = [values[i] / values[i - 1] if values[i - 1] != 0 else float("inf") for i in range(1, len(values))]
        if ratios and max(ratios) >= threshold:
            selected.extend(token for token, _ in errors[: ratios.index(max(ratios)) + 1])
    return selected


def reduce_full_rank_columns(matrix: torch.Tensor, max_removal: int = 1, eps: float | None = None) -> Tuple[torch.Tensor, int]:
    if matrix.ndim != 2 or max_removal <= 0:
        return matrix, 0
    rows, cols = matrix.shape
    orig_dtype = matrix.detach().dtype if matrix.detach().is_floating_point() else torch.float32
    singular_values = torch.linalg.svdvals(matrix.detach().double())
    max_sv = singular_values.max().clamp_min(torch.finfo(singular_values.dtype).tiny)
    if eps is None:
        eps_machine = torch.finfo(orig_dtype).eps
        threshold = min(rows, cols) * eps_machine * max_sv
    else:
        threshold = eps * max_sv
    rank = int((singular_values > threshold).sum().item())
    min_dim = min(rows, cols)
    if rank < min_dim:
        return matrix, 0

    if rows == cols:
        removal = min(max_removal, cols - 1)
        if removal <= 0:
            return matrix, 0
        col_norms = torch.norm(matrix.detach().double(), dim=0)
        _, sorted_indices = torch.sort(col_norms)
        remove_set = set(sorted_indices[:removal].tolist())
        keep_indices = torch.tensor([i for i in range(cols) if i not in remove_set], device=matrix.device, dtype=torch.long)
        return matrix[:, keep_indices], removal

    if rows < cols:
        target_rank = max(1, rows - max_removal)
        if target_rank >= rows:
            return matrix, 0
        u, s, vt = torch.linalg.svd(matrix.detach().double(), full_matrices=False)
        truncated = (u[:, :target_rank] * s[:target_rank]) @ vt[:target_rank, :]
        return truncated.to(orig_dtype).to(matrix.device), rows - target_rank

    return matrix, 0
