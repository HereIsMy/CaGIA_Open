import torch
import random
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter
import numpy as np


def set_random_seed(seed: int = 1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def can_be_expressed_two(vectors, target, device, atol=2e-5):
    A = vectors.t().float().to(device)
    target_temp = target.float().to(device)
    n1, n2, m = target_temp.shape
    target_flattened = target_temp.view(n1 * n2, m)
    q, r = torch.linalg.qr(A, mode='reduced')
    q_t_target = q.t() @ target_flattened.t()
    solution = torch.linalg.solve_triangular(r, q_t_target, upper=True)
    reconstructed_vector_in = A @ solution
    isclose_tensor = torch.isclose(reconstructed_vector_in.t(), target_flattened, atol=atol, rtol=atol)
    is_close_in = torch.all(isclose_tensor, dim=1)
    is_close_in_first = is_close_in.view(n1, n2)

    A = vectors.float().to(device)
    target_temp = target.float().to(device)
    n1, n2, m = target_temp.shape
    target_flattened = target_temp.view(n1 * n2, m)
    coefficients = torch.linalg.pinv(A.t()) @ target_flattened.t()
    reconstructed_vector_flattened = A.t() @ coefficients
    isclose_tensor = torch.isclose(reconstructed_vector_flattened.t(), target_flattened, atol=atol, rtol=atol)
    is_close_in = torch.all(isclose_tensor, dim=1)
    is_close_in_second = is_close_in.view(n1, n2)

    is_close_in = torch.logical_or(is_close_in_first, is_close_in_second)
    is_close_in_rows, is_close_in_cols = torch.where(is_close_in)
    coefficients_in = solution.t().view(n1, n2, -1)
    reconstructed_vector_in = reconstructed_vector_in.t().view(n1, n2, -1)
    coefficients_in = coefficients_in[is_close_in_rows, is_close_in_cols, :]
    reconstructed_vector_in = reconstructed_vector_in[is_close_in_rows, is_close_in_cols, :]
    return is_close_in_rows, is_close_in_cols, coefficients_in, reconstructed_vector_in


def can_be_expressed_two_gpu2(vectors, target, device, atol=2e-5):
    A = vectors.to(device).t().float()
    target_temp = target.to(device).float()
    n1, n2, m = target_temp.shape
    target_flattened = target_temp.view(n1 * n2, m)
    coefficients_pinv = torch.linalg.pinv(A) @ target_flattened.t()
    reconstructed_vector_pinv = (A @ coefficients_pinv).t()
    isclose_tensor_pinv = torch.isclose(reconstructed_vector_pinv, target_flattened, atol=atol, rtol=atol)
    is_close_in_pinv = torch.all(isclose_tensor_pinv, dim=1)
    is_close_in = is_close_in_pinv.view(n1, n2)
    is_close_in_rows, is_close_in_cols = torch.where(is_close_in)
    coefficients_in_pinv = coefficients_pinv.t().view(n1, n2, -1)
    reconstructed_vector_in_pinv = reconstructed_vector_pinv.view(n1, n2, -1)
    coefficients_in = coefficients_in_pinv[is_close_in_rows, is_close_in_cols, :]
    reconstructed_vector_in = reconstructed_vector_in_pinv[is_close_in_rows, is_close_in_cols, :]
    return is_close_in_rows, is_close_in_cols, coefficients_in, reconstructed_vector_in


class CustomDataset(Dataset):
    def __init__(self, data):
        self.labels = torch.tensor(data['label'].tolist(), dtype=torch.long)
        self.sentences = data['sentence'].tolist()

    def __getitem__(self, idx):
        item = {
            'labels': self.labels[idx],
            'sentence': self.sentences[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)


def preprocess_tsv(file_path):
    data = pd.read_csv(file_path, delimiter='\t', header=None, names=['label', 'sentence_id', 'sentence'])
    return data


def get_random_batch(train_dataloader):
    num_batch = len(train_dataloader)
    batch_num_target = random.randrange(0, num_batch)
    random_batch_target = {}
    for i, batch in enumerate(train_dataloader):
        if i == batch_num_target:
            random_batch_target = batch
            break
    return random_batch_target


def can_be_expressed_two_gpu(vectors, target, device, atol=1e-5):
    A = vectors.to(device).t().float()
    target_temp = target.to(device).float()
    n1, n2, m = target_temp.shape
    target_flattened = target_temp.view(n1 * n2, m)
    q, r = torch.linalg.qr(A, mode='reduced')
    q_t_target = q.t() @ target_flattened.t()
    solution_qr = torch.linalg.solve_triangular(r, q_t_target, upper=True)
    reconstructed_vector_qr = (A @ solution_qr).t()
    coefficients_pinv = torch.linalg.pinv(A) @ target_flattened.t()
    reconstructed_vector_pinv = (A @ coefficients_pinv).t()
    isclose_tensor_qr = torch.isclose(reconstructed_vector_qr, target_flattened, atol=atol, rtol=atol)
    is_close_in_qr = torch.all(isclose_tensor_qr, dim=1)
    isclose_tensor_pinv = torch.isclose(reconstructed_vector_pinv, target_flattened, atol=atol, rtol=atol)
    is_close_in_pinv = torch.all(isclose_tensor_pinv, dim=1)
    is_close_in = torch.logical_or(is_close_in_qr, is_close_in_pinv)
    is_close_in = is_close_in.view(n1, n2)
    is_close_in_rows, is_close_in_cols = torch.where(is_close_in)
    coefficients_in_qr = solution_qr.t().view(n1, n2, -1)
    coefficients_in_pinv = coefficients_pinv.t().view(n1, n2, -1)
    reconstructed_vector_in_qr = reconstructed_vector_qr.view(n1, n2, -1)
    reconstructed_vector_in_pinv = reconstructed_vector_pinv.view(n1, n2, -1)
    coefficients_in = torch.where(is_close_in.unsqueeze(-1), coefficients_in_qr, coefficients_in_pinv)
    reconstructed_vector_in = torch.where(is_close_in.unsqueeze(-1), reconstructed_vector_in_qr,
                                          reconstructed_vector_in_pinv)
    coefficients_in = coefficients_in[is_close_in_rows, is_close_in_cols, :]
    reconstructed_vector_in = reconstructed_vector_in[is_close_in_rows, is_close_in_cols, :]
    return is_close_in_rows, is_close_in_cols, coefficients_in, reconstructed_vector_in


def remove_zero_vectors(tensor):
    row_sums = tensor.abs().sum(dim=1)
    non_zero_mask = row_sums > 0
    non_zero_tensor = tensor[non_zero_mask]
    return non_zero_tensor


def remove_duplicates(tensor_list):
    tensor_tuples = [tuple(t.cpu().numpy()) for t in tensor_list]
    seen = set()
    unique_tensor_list = [
        t for t, tpl in zip(tensor_list, tensor_tuples)
        if tpl not in seen and not seen.add(tpl)
    ]
    return unique_tensor_list


def filter_identical_elements(tensor_list):
    filtered_tensor_list = [t for t in tensor_list if not torch.equal(t, torch.full_like(t, t[0]))]
    return filtered_tensor_list


def remove_duplicates_if_any(tensor_list):
    tensor_tuples = [tuple(t) for t in tensor_list]
    has_duplicates = len(tensor_tuples) != len(set(tensor_tuples))
    if has_duplicates:
        seen = set()
        unique_tensor_list = [
            t for t, tpl in zip(tensor_list, tensor_tuples)
            if tpl not in seen and not seen.add(tpl)
        ]
        return unique_tensor_list
    else:
        return tensor_list


def normalize_and_flatten(tensor_list):
    flattened = []
    for item in tensor_list:
        if isinstance(item, list):
            flattened.extend(normalize_and_flatten(item))
        elif isinstance(item, torch.Tensor):
            flattened.append(item)
        else:
            raise TypeError(f"Unsupported type: {type(item)}")
    return flattened


def functionTokenize(text):
    return text.split()


def rouge_1(candidate, references):
    candidate_tokens = functionTokenize(candidate)
    candidate_counter = Counter(candidate_tokens)
    max_recall = 0
    for reference in references:
        reference_tokens = functionTokenize(reference)
        reference_counter = Counter(reference_tokens)
        common_words = candidate_counter & reference_counter
        recall = sum(common_words.values()) / max(1, sum(reference_counter.values()))
        max_recall = max(max_recall, recall)
    return max_recall


def rouge_2(candidate, references):
    candidate_tokens = functionTokenize(candidate)
    candidate_bigrams = Counter(zip(candidate_tokens, candidate_tokens[1:]))
    max_recall = 0
    for reference in references:
        reference_tokens = functionTokenize(reference)
        reference_bigrams = Counter(zip(reference_tokens, reference_tokens[1:]))
        common_bigrams = candidate_bigrams & reference_bigrams
        recall = sum(common_bigrams.values()) / max(1, sum(reference_bigrams.values()))
        max_recall = max(max_recall, recall)
    return max_recall


def lcs_length(candidate_tokens, reference_tokens):
    m = len(candidate_tokens)
    n = len(reference_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if candidate_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def rouge_l(candidate, references):
    candidate_tokens = functionTokenize(candidate)
    max_recall = 0

    for reference in references:
        reference_tokens = functionTokenize(reference)
        lcs_len = lcs_length(candidate_tokens, reference_tokens)
        recall = lcs_len / max(1, len(reference_tokens))
        max_recall = max(max_recall, recall)
    return max_recall


def read_csv_in_batches(file_path, batch_size=1024):
    chunk_iter = pd.read_csv(file_path, chunksize=batch_size)
    for chunk in chunk_iter:
        batch_data = chunk.to_numpy()
        yield batch_data
