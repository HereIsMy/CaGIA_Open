from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def preprocess_matrix(matrix: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor | int]:
    a = matrix.detach().float()
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    q, r = torch.linalg.qr(a, mode="reduced")
    # 使用 float64 计算奇异值以减少 SVD 计算噪声
    singular_values = torch.linalg.svdvals(r.double())
    max_sv = singular_values.max().clamp_min(torch.finfo(singular_values.dtype).tiny)
    # size-aware 数值秩阈值：max(m, n) * eps_float32 * max_sv
    # 仅保留显著非零的主要部分（真实列空间），丢弃噪声级别的小奇异值，
    # 避免 R 的近零对角元在 solve_triangular 中产生 1/λ 放大效应导致假阳 token
    eps_machine = torch.finfo(a.dtype).eps
    rank_threshold = max(r.shape[0], r.shape[1]) * eps_machine * max_sv
    rank = int((singular_values > rank_threshold).sum().item())
    if rank <= 0:
        return {"rank": 0}
    return {
        "rank": rank,
        "q": q[:, :rank],
        "r": r[:rank, :rank] + eps * torch.eye(rank, device=a.device),
        "pinv": torch.linalg.pinv(a),
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
    """计算所有候选向量的重建残差（不受 tol 过滤），返回形状为 (n1*n2,) 的残差张量。"""
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
    """对满秩方阵移除最多 max_removal 个列（神经元梯度），使其变为低秩。

    当 PEFT 方法为 partial 且梯度输入维度等于输出维度（方阵）时，若梯度满秩，
    其列空间为整个 R^n，线性重建攻击无法缩小候选范围。移除少量列可使列空间
    变为真子空间，从而恢复攻击的区分能力。优先移除范数最小的列。

    Args:
        matrix: 梯度矩阵 (in, out)
        max_removal: 最多移除的列数
        eps: 奇异值阈值系数。若为 None，则使用 size-aware 的数值秩阈值
            max(rows, cols) * eps_machine * max_sv，避免大矩阵数值噪声
            导致真实低秩矩阵被误判为满秩

    Returns:
        (缩减后的矩阵, 实际移除的列数)
    """
    if matrix.ndim != 2 or max_removal <= 0:
        return matrix, 0
    rows, cols = matrix.shape
    # 仅处理方阵（输入维度 == 输出维度）
    if rows != cols:
        return matrix, 0
    # 保留原始数据 dtype，用于确定数值秩阈值
    orig_dtype = matrix.detach().dtype if matrix.detach().is_floating_point() else torch.float32
    # 使用 float64 计算奇异值以减少 SVD 计算噪声
    singular_values = torch.linalg.svdvals(matrix.detach().double())
    max_sv = singular_values.max().clamp_min(torch.finfo(singular_values.dtype).tiny)
    # size-aware 数值秩阈值：max(m, n) * eps_orig * max_sv
    # 关键：必须使用原始数据 dtype 的机器精度（float32 ≈ 1.2e-7），
    # 而非 SVD 计算 dtype（float64 ≈ 2.2e-16）。因为即使 SVD 用 float64 计算，
    # 输入数据本身是 float32，其零空间奇异值的噪声地板约为
    # max(rows, cols) * eps_float32 * max_sv ≈ 5e-4 * max_sv（对 4096x4096）。
    # 若误用 float64 的 eps，阈值变为 9e-13 * max_sv，远低于实际噪声，
    # 会导致真实低秩矩阵被误判为满秩，进而错误移除列、引入大量假阳 token。
    if eps is None:
        eps_machine = torch.finfo(orig_dtype).eps
        threshold = max(rows, cols) * eps_machine * max_sv
    else:
        threshold = eps * max_sv
    rank = int((singular_values > threshold).sum().item())
    # 已经是低秩，无需处理
    if rank < rows:
        return matrix, 0
    # 移除范数最小的 removal 个列
    removal = min(max_removal, cols - 1)
    if removal <= 0:
        return matrix, 0
    col_norms = torch.norm(matrix.detach().double(), dim=0)
    _, sorted_indices = torch.sort(col_norms)
    remove_set = set(sorted_indices[:removal].tolist())
    keep_indices = torch.tensor([i for i in range(cols) if i not in remove_set], device=matrix.device, dtype=torch.long)
    return matrix[:, keep_indices], removal
