import torch
from torch.utils.data import Dataset
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import json


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


class CustomDataset_list(Dataset):
    def __init__(self, data):
        self.labels = torch.tensor(data['label'], dtype=torch.long)
        self.sentences = data['sentence']

    def __getitem__(self, idx):
        item = {
            'labels': self.labels[idx],
            'sentence': self.sentences[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)


def convert_hf_dataset_to_custom(hf_dataset):
    data_dict = {
        'label': hf_dataset['label'],
        'sentence': hf_dataset['sentence']
    }
    return data_dict


def preprocess_tsv(file_path):
    data = pd.read_csv(file_path, delimiter='\t', header=None, names=['label', 'sentence_id', 'sentence'])
    return data


def preprocess_csv(file_path):
    data = pd.read_csv(file_path, skiprows=1, header=None, names=['sentence', 'label'])
    return data


def preprocess_IMDB(file_path, data_class='train'):
    sentences = []
    labels = []
    for label in ['pos', 'neg']:
        subdir_path = os.path.join(file_path, 'aclImdb', data_class, label)
        if not os.path.exists(subdir_path):
            raise ValueError(f"路径不存在: {subdir_path}")
        for filename in os.listdir(subdir_path):
            filepath = os.path.join(subdir_path, filename)
            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    sentence = f.read().strip()
                    sentences.append(sentence)
                    labels.append(0 if label == 'neg' else 1)
    data = pd.DataFrame({
        'label': labels,
        'sentence': sentences
    })
    return data


def preprocess_yelp(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            sentences.append(json_obj['text'])
            labels.append(int(json_obj['compliment_count']))
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
        sentences, labels, test_size=0.1, random_state=42
    )
    train_data = pd.DataFrame({
        'sentence': train_sentences,
        'label': train_labels
    })
    test_data = pd.DataFrame({
        'sentence': test_sentences,
        'label': test_labels
    })
    return train_data, test_data


def can_be_expressed(vectors, target):
    A = vectors.t().float()
    target = target.float().unsqueeze(1)
    q, r = torch.linalg.qr(A, mode='reduced')
    solution = torch.linalg.solve_triangular(r, q.t() @ target, upper=True)
    reconstructed_vector = A @ solution
    coefficients = solution.squeeze(1)
    is_close = torch.allclose(reconstructed_vector.squeeze(1), target.squeeze(1), atol=5e-1, rtol=5e-1)
    res_atol = abs(reconstructed_vector.squeeze(1)-target.squeeze(1))-5e-1*(abs(target.squeeze(1)))
    res_atol_final = max(res_atol)
    return is_close, coefficients, reconstructed_vector.squeeze(1), res_atol_final


def can_be_expressed_Bert(vectors, target):
    D, N = vectors.shape
    K = target.shape[0]
    A = vectors.t().float()
    target = target.float()
    q, r = torch.linalg.qr(A, mode='reduced')
    Qt_target = q.t() @ target.t()
    try:
        solution = torch.linalg.solve_triangular(r, Qt_target, upper=True)
    except RuntimeError as e:
        print(f"QR/Solve error: {e}")
        fake_coef = torch.zeros(K, D, device=vectors.device)
        return torch.zeros(K, dtype=torch.bool), fake_coef, fake_coef, torch.full((K,), float('inf'))
    coefficients = solution.t()
    reconstructed = coefficients @ A
    atol = 5e-1
    rtol = 5e-1
    is_close = torch.allclose(reconstructed, target, atol=atol, rtol=rtol, equal_nan=True)
    element_wise_close = torch.isclose(reconstructed, target, atol=atol, rtol=rtol)
    is_close_per_sample = torch.all(element_wise_close, dim=1)
    residual = torch.abs(reconstructed - target)
    tolerance = atol + rtol * torch.abs(target)
    exceeded_residual = residual - tolerance
    max_residuals = torch.max(exceeded_residual, dim=1).values
    return is_close_per_sample, coefficients, reconstructed, max_residuals


def load_trainable_params(model, params_cpu_list):
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) != len(params_cpu_list):
        raise ValueError(f"参数数量不匹配：模型有 {len(trainable_params)} 个可训练参数，"
                         f"但传入了 {len(params_cpu_list)} 个参数。")
    with torch.no_grad():
        for param, saved_param in zip(trainable_params, params_cpu_list):
            param.copy_(saved_param.to(param.device))


def prune_smallest_percent1(attn_grade, per=0.01):
    all_abs_values = []
    for t in attn_grade:
        all_abs_values.append(t.abs().view(-1))
    all_abs_values = torch.cat(all_abs_values)
    k = int(len(all_abs_values) * per)
    if k == 0:
        k = 1
    threshold_value = torch.kthvalue(all_abs_values, k).values.item()
    pruned_attn_grade = []
    for t in attn_grade:
        mask = t.abs() >= threshold_value
        pruned_t = t * mask
        pruned_attn_grade.append(pruned_t)
    return pruned_attn_grade


def prune_smallest_percent(attn_grade, per=0.01):
    if isinstance(attn_grade, list):
        if not attn_grade:
            return attn_grade

        all_abs_values = []
        for t in attn_grade:
            if not isinstance(t, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor in list, got {type(t)}")
            all_abs_values.append(t.abs().view(-1))

        all_abs_values = torch.cat(all_abs_values)

        total_num = len(all_abs_values)
        k = max(1, int(total_num * per))
        threshold_value = torch.kthvalue(all_abs_values, k).values.item()

        pruned_attn_grade = []
        for t in attn_grade:
            mask = t.abs() >= threshold_value
            pruned_t = t * mask
            pruned_attn_grade.append(pruned_t)

        return pruned_attn_grade

    else:
        if not isinstance(attn_grade, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor or list of torch.Tensor, got {type(attn_grade)}")

        t = attn_grade
        num_elements = t.numel()
        k = max(1, int(num_elements * per))

        abs_vals = t.abs().view(-1)
        threshold_value = torch.kthvalue(abs_vals, k).values.item()

        mask = t.abs() >= threshold_value
        pruned_t = t * mask

        return pruned_t


def add_gaussian_noise(tensor, mean=0.0, std=0.0001):
    if isinstance(tensor, list):
        noisy_tensors = []
        for t in tensor:
            if not isinstance(t, torch.Tensor):
                raise ValueError(f"Expected tensor, but got {type(t)}")
            noise = torch.randn_like(t, dtype=t.dtype, device=t.device) * std + mean
            noisy_t = t + noise
            noisy_tensors.append(noisy_t)
        return noisy_tensors
    else:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected tensor, but got {type(tensor)}")
        noise = torch.randn_like(tensor, dtype=tensor.dtype, device=tensor.device) * std + mean
        return tensor + noise
