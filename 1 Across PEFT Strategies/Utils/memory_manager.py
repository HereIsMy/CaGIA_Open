import torch
from typing import Tuple


def get_gpu_memory_info() -> Tuple[float, float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0

    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    used_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
    free_memory = total_memory - used_memory

    return total_memory, used_memory, free_memory


def calculate_dynamic_batch_size(
    current_batch_size: int,
    max_batch_size: int,
    min_batch_size: int = 128,
    target_memory_usage: float = 0.85,
    safety_margin: float = 0.1,
    verbose: bool = False
) -> int:
    if not torch.cuda.is_available():
        return current_batch_size

    total_memory, used_memory, free_memory = get_gpu_memory_info()

    if total_memory == 0:
        return current_batch_size

    memory_usage_ratio = used_memory / total_memory

    if verbose:
        print(f"显存使用情况: {used_memory:.2f}GB / {total_memory:.2f}GB ({memory_usage_ratio*100:.1f}%)")
        print(f"当前batch size: {current_batch_size}")

    if memory_usage_ratio > target_memory_usage:
        reduction_factor = (target_memory_usage + safety_margin) / memory_usage_ratio
        new_batch_size = int(current_batch_size * reduction_factor)
        new_batch_size = max(min_batch_size, new_batch_size)

        if verbose:
            print(f"显存使用率过高，减少batch size: {current_batch_size} -> {new_batch_size}")

        return new_batch_size

    elif memory_usage_ratio < target_memory_usage - safety_margin:
        increase_factor = target_memory_usage / memory_usage_ratio
        new_batch_size = int(current_batch_size * increase_factor)
        new_batch_size = min(max_batch_size, new_batch_size)

        if verbose:
            print(f"显存使用率较低，增加batch size: {current_batch_size} -> {new_batch_size}")

        return new_batch_size

    return current_batch_size


def safe_batch_execution(
    func,
    batch_size: int,
    max_retries: int = 3,
    min_batch_size: int = 128,
    verbose: bool = False
):
    current_batch_size = batch_size

    for attempt in range(max_retries):
        try:
            if verbose:
                print(f"尝试使用batch size: {current_batch_size}")

            result = func(batch_size=current_batch_size)

            if verbose:
                print(f"成功执行，batch size: {current_batch_size}")

            return result, current_batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                current_batch_size = max(current_batch_size // 2, min_batch_size)

                if verbose:
                    print(f"显存溢出，减少batch size到: {current_batch_size}")

                if current_batch_size == min_batch_size:
                    raise RuntimeError(f"即使使用最小batch size {min_batch_size} 仍然显存溢出")
            else:
                raise e

    raise RuntimeError(f"在 {max_retries} 次尝试后仍然无法执行")


def estimate_optimal_batch_size(
    tensor_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    target_memory_usage: float = 0.8,
    safety_margin: float = 0.2,
    verbose: bool = False
) -> int:
    if not torch.cuda.is_available():
        return 4096

    total_memory, used_memory, free_memory = get_gpu_memory_info()

    if total_memory == 0:
        return 4096

    element_size = torch.tensor([], dtype=dtype).element_size()
    elements_per_sample = 1
    for dim in tensor_shape:
        elements_per_sample *= dim

    bytes_per_sample = elements_per_sample * element_size

    available_memory = free_memory * (target_memory_usage - safety_margin)
    available_bytes = available_memory * (1024 ** 3)

    estimated_batch_size = int(available_bytes / bytes_per_sample)

    estimated_batch_size = max(128, min(estimated_batch_size, 8192))

    if verbose:
        print(f"估计最优batch size: {estimated_batch_size}")
        print(f"单样本显存占用: {bytes_per_sample / (1024**2):.2f}MB")
        print(f"可用显存: {available_memory:.2f}GB")

    return estimated_batch_size


def clear_gpu_cache(verbose: bool = False):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        if verbose:
            total_memory, used_memory, free_memory = get_gpu_memory_info()
            print(f"清理后显存: {used_memory:.2f}GB / {total_memory:.2f}GB ({used_memory/total_memory*100:.1f}%)")
