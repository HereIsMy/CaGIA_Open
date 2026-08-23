# -*- coding: utf-8 -*-
import os
import json
from collections import Counter
from itertools import combinations, permutations
from typing import List, Dict, Any, Optional, Tuple

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def lcs_length(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def normalize_text(text: str) -> str:
    text = ' '.join(text.split())
    text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
    text = text.replace(' ;', ';').replace(' :', ':').replace(' \'', '\'').replace(' "', '"')
    text = text.replace('\'s', '\'s').replace('s\'', 's\'')
    punctuation = [',', '.', '!', '?', ';', ':', "'", '"']
    for p in punctuation:
        text = text.replace(p, ' ' + p)
    text = ' '.join(text.split())
    return text


def normalized_words(text: str) -> List[str]:
    return normalize_text(text).strip().split() if text else []


def normalized_word_count(text: str) -> int:
    return len(normalized_words(text))


def _empty_sentence_detail(original_text: str) -> Dict[str, Any]:
    return {
        "original": original_text,
        "reconstructed": "",
        "rouge_l": 0.0,
        "rouge_1": 0.0,
        "rouge_2": 0.0,
        "token_accuracy": 0.0,
        "correct_tokens": 0,
        "f1_score": 0.0,
        "total_tokens": normalized_word_count(original_text),
        "recon_token_count": 0,
    }


def _pair_metrics(original_text: str, recon_text: str) -> Dict[str, Any]:
    rouge_l = calculate_rouge_l(original_text, recon_text)
    rouge_1 = calculate_rouge_1(original_text, recon_text)
    rouge_2 = calculate_rouge_2(original_text, recon_text)
    token_accuracy, correct_tokens = calculate_token_accuracy(original_text, recon_text)
    f1_score = calculate_f1_score(correct_tokens, normalized_word_count(original_text), normalized_word_count(recon_text))
    return {
        "reconstructed": recon_text,
        "rouge_l": rouge_l,
        "rouge_1": rouge_1,
        "rouge_2": rouge_2,
        "token_accuracy": token_accuracy,
        "correct_tokens": correct_tokens,
        "f1_score": f1_score,
        "total_tokens": normalized_word_count(original_text),
        "match_score": token_accuracy * 0.7 + rouge_l * 0.3,
    }


def _best_text_assignment(ref_sentences: List[str], recon_texts: List[str]) -> Dict[int, Dict[str, Any]]:
    if not recon_texts:
        return {}

    pair_metrics = {
        (ref_index, recon_index): _pair_metrics(original_text, recon_text)
        for ref_index, original_text in enumerate(ref_sentences)
        if original_text
        for recon_index, recon_text in enumerate(recon_texts)
    }
    ref_indices = [index for index, text in enumerate(ref_sentences) if text]
    assignment_size = min(len(ref_indices), len(recon_texts))

    if assignment_size <= 6:
        best_score = -1.0
        best_assignment: Dict[int, Dict[str, Any]] = {}
        selected_ref_groups = combinations(ref_indices, assignment_size)
        for selected_ref_indices in selected_ref_groups:
            for selected_recon_indices in permutations(range(len(recon_texts)), assignment_size):
                current_assignment = {}
                current_score = 0.0
                for ref_index, recon_index in zip(selected_ref_indices, selected_recon_indices):
                    metrics = pair_metrics[(ref_index, recon_index)]
                    current_score += metrics["match_score"]
                    current_assignment[ref_index] = metrics
                if current_score > best_score:
                    best_score = current_score
                    best_assignment = current_assignment
        return best_assignment

    match_scores = []
    for (ref_index, recon_index), metrics in pair_metrics.items():
        match_scores.append((metrics["match_score"], ref_index, recon_index, metrics))
    match_scores.sort(reverse=True)

    used_ref_indices = set()
    used_recon_indices = set()
    assignment = {}
    for _score, ref_index, recon_index, metrics in match_scores:
        if ref_index not in used_ref_indices and recon_index not in used_recon_indices:
            assignment[ref_index] = metrics
            used_ref_indices.add(ref_index)
            used_recon_indices.add(recon_index)
    return assignment


def calculate_rouge_1(reference: str, candidate: str) -> float:
    if not reference:
        return 0.0

    reference = normalize_text(reference)
    candidate = normalize_text(candidate)

    ref_words = reference.strip().split()
    cand_words = candidate.strip().split()

    if not ref_words:
        return 0.0

    ref_counts = Counter(ref_words)
    cand_counts = Counter(cand_words)
    overlap = sum((ref_counts & cand_counts).values())

    return overlap / len(ref_words)


def calculate_rouge_2(reference: str, candidate: str) -> float:
    if not reference:
        return 0.0

    reference = normalize_text(reference)
    candidate = normalize_text(candidate)

    ref_words = reference.strip().split()
    cand_words = candidate.strip().split()

    if len(ref_words) < 2:
        return 0.0

    def get_bigrams(words):
        return [(words[i], words[i + 1]) for i in range(len(words) - 1)]

    ref_bigrams = Counter(get_bigrams(ref_words))
    cand_bigrams = Counter(get_bigrams(cand_words))

    if not ref_bigrams:
        return 0.0

    overlap = sum((ref_bigrams & cand_bigrams).values())
    return overlap / (len(ref_words) - 1)


def calculate_rouge_l(reference: str, candidate: str) -> float:
    if not reference:
        return 0.0

    reference = normalize_text(reference)
    candidate = normalize_text(candidate)

    ref_words = reference.strip().split()
    cand_words = candidate.strip().split()

    if not ref_words:
        return 0.0

    lcs = lcs_length(ref_words, cand_words)
    return lcs / len(ref_words)


def calculate_token_accuracy(reference: str, candidate: str) -> Tuple[float, int]:
    if not reference:
        return 0.0, 0

    reference = normalize_text(reference)
    candidate = normalize_text(candidate)

    ref_words = reference.strip().split()
    cand_words = candidate.strip().split()

    total_words = len(ref_words)
    if total_words == 0:
        return 0.0, 0

    min_len = min(total_words, len(cand_words))
    correct = 0

    for i in range(min_len):
        if ref_words[i] == cand_words[i]:
            correct += 1

    accuracy = correct / total_words
    return accuracy, correct


def calculate_f1_score(correct_tokens: int, total_tokens: int, recon_token_count: int) -> float:
    denominator = total_tokens + recon_token_count
    if denominator == 0:
        return 0.0
    return 2.0 * correct_tokens / denominator


def calculate_single_sentence_metrics(original_text: str, reconstructed_candidates: List[str]) -> Dict[str, Any]:
    if not reconstructed_candidates:
        return {
            "original": original_text,
            "reconstructed": "",
            "rouge_l": 0.0,
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "token_accuracy": 0.0,
            "correct_tokens": 0,
            "total_tokens": normalized_word_count(original_text),
            "f1_score": 0.0
        }

    best_metrics = None
    best_reconstructed = ""

    for recon_text in reconstructed_candidates:
        if not recon_text:
            continue

        rouge_l = calculate_rouge_l(original_text, recon_text)
        rouge_1 = calculate_rouge_1(original_text, recon_text)
        rouge_2 = calculate_rouge_2(original_text, recon_text)
        token_accuracy, correct_tokens = calculate_token_accuracy(original_text, recon_text)
        f1_score = calculate_f1_score(correct_tokens, normalized_word_count(original_text), normalized_word_count(recon_text))

        if best_metrics is None or token_accuracy > best_metrics["token_accuracy"]:
            best_metrics = {
                "original": original_text,
                "reconstructed": recon_text,
                "rouge_l": rouge_l,
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "token_accuracy": token_accuracy,
                "correct_tokens": correct_tokens,
                "total_tokens": normalized_word_count(original_text),
                "f1_score": f1_score
            }
            best_reconstructed = recon_text

    if best_metrics is None:
        return {
            "original": original_text,
            "reconstructed": "",
            "rouge_l": 0.0,
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "token_accuracy": 0.0,
            "correct_tokens": 0,
            "total_tokens": normalized_word_count(original_text),
            "f1_score": 0.0
        }

    return best_metrics


def calculate_batch_metrics(ref_sentences: List[str], recon_sentences: List[Any]) -> Dict[str, Any]:
    if not ref_sentences:
        return {
            "mean_rouge_l": 0.0,
            "mean_rouge_1": 0.0,
            "mean_rouge_2": 0.0,
            "mean_token_accuracy": 0.0,
            "mean_correct_tokens": 0.0,
            "mean_f1_score": 0.0,
            "sentence_count": 0,
            "total_correct_tokens": 0,
            "sentence_details": []
        }

    all_recon_texts = []
    for recon_item in recon_sentences:
        if isinstance(recon_item, str):
            if recon_item.strip():
                all_recon_texts.append(recon_item)
        elif isinstance(recon_item, list):
            for s in recon_item:
                if isinstance(s, str) and s.strip():
                    all_recon_texts.append(s)

    ref_matches = _best_text_assignment(ref_sentences, all_recon_texts)

    total_rouge_l = 0.0
    total_rouge_1 = 0.0
    total_rouge_2 = 0.0
    total_token_accuracy = 0.0
    total_correct_tokens = 0
    total_f1_score = 0.0
    sentence_count = len(ref_sentences)
    sentence_details = []

    for i, original_text in enumerate(ref_sentences):
        if not original_text:
            sentence_details.append(_empty_sentence_detail(original_text))
            continue

        match = ref_matches.get(i)
        if match:
            total_rouge_l += match["rouge_l"]
            total_rouge_1 += match["rouge_1"]
            total_rouge_2 += match["rouge_2"]
            total_token_accuracy += match["token_accuracy"]
            total_correct_tokens += match["correct_tokens"]
            total_f1_score += match["f1_score"]

            sentence_details.append({
                "original": original_text,
                "reconstructed": match["reconstructed"],
                "rouge_l": match["rouge_l"],
                "rouge_1": match["rouge_1"],
                "rouge_2": match["rouge_2"],
                "token_accuracy": match["token_accuracy"],
                "correct_tokens": match["correct_tokens"],
                "f1_score": match["f1_score"],
                "total_tokens": match["total_tokens"],
                "recon_token_count": normalized_word_count(match["reconstructed"])
            })
        else:
            detail = _empty_sentence_detail(original_text)
            detail["recon_token_count"] = 0
            sentence_details.append(detail)

    return {
        "mean_rouge_l": total_rouge_l / sentence_count,
        "mean_rouge_1": total_rouge_1 / sentence_count,
        "mean_rouge_2": total_rouge_2 / sentence_count,
        "mean_token_accuracy": total_token_accuracy / sentence_count,
        "mean_correct_tokens": total_correct_tokens / sentence_count,
        "mean_f1_score": total_f1_score / sentence_count,
        "sentence_count": sentence_count,
        "total_correct_tokens": total_correct_tokens,
        "sentence_details": sentence_details
    }


def calculate_method_metrics(ref_sentences: Dict[str, List[List[str]]], recon_dict: Dict[str, List[List[str]]]) -> Dict[str, Any]:
    results = {
        "batch_metrics": {},
        "list_metrics": {},
        "mean_metrics": {
            "rouge_l": 0.0,
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "token_accuracy": 0.0,
            "correct_tokens": 0.0,
            "f1_score": 0.0
        },
        "details": []
    }

    all_rouge_l = []
    all_rouge_1 = []
    all_rouge_2 = []
    all_token_accuracy = []
    all_correct_tokens = []
    all_f1_score = []

    for batch_size in sorted(ref_sentences.keys(), key=lambda x: int(x)):
        if batch_size not in ref_sentences or batch_size not in recon_dict:
            continue

        ref_data = ref_sentences[batch_size]
        recon_data = recon_dict[batch_size]

        batch_list_metrics = []
        batch_details = []

        for list_idx in range(5):
            if list_idx >= len(ref_data) or list_idx >= len(recon_data):
                continue

            ref_sublist = ref_data[list_idx]
            recon_sublist = recon_data[list_idx]

            if not ref_sublist:
                continue

            group_metrics = calculate_batch_metrics(ref_sublist, recon_sublist)
            batch_list_metrics.append(group_metrics)

            batch_details.append({
                "list_index": list_idx,
                "rouge_l": group_metrics["mean_rouge_l"],
                "rouge_1": group_metrics["mean_rouge_1"],
                "rouge_2": group_metrics["mean_rouge_2"],
                "token_accuracy": group_metrics["mean_token_accuracy"],
                "correct_tokens": group_metrics["total_correct_tokens"],
                "f1_score": group_metrics["mean_f1_score"],
                "sentence_count": group_metrics["sentence_count"],
                "sentence_details": group_metrics.get("sentence_details", [])
            })

            all_rouge_l.append(group_metrics["mean_rouge_l"])
            all_rouge_1.append(group_metrics["mean_rouge_1"])
            all_rouge_2.append(group_metrics["mean_rouge_2"])
            all_token_accuracy.append(group_metrics["mean_token_accuracy"])
            all_correct_tokens.append(group_metrics["total_correct_tokens"])
            all_f1_score.append(group_metrics["mean_f1_score"])

        if batch_list_metrics:
            batch_mean_rouge_l = sum(m["mean_rouge_l"] for m in batch_list_metrics) / len(batch_list_metrics)
            batch_mean_rouge_1 = sum(m["mean_rouge_1"] for m in batch_list_metrics) / len(batch_list_metrics)
            batch_mean_rouge_2 = sum(m["mean_rouge_2"] for m in batch_list_metrics) / len(batch_list_metrics)
            batch_mean_token_accuracy = sum(m["mean_token_accuracy"] for m in batch_list_metrics) / len(batch_list_metrics)
            batch_total_correct_tokens = sum(m["total_correct_tokens"] for m in batch_list_metrics)
            batch_mean_correct_tokens = batch_total_correct_tokens / len(batch_list_metrics)
            batch_mean_f1_score = sum(m["mean_f1_score"] for m in batch_list_metrics) / len(batch_list_metrics)
        else:
            batch_mean_rouge_l = 0.0
            batch_mean_rouge_1 = 0.0
            batch_mean_rouge_2 = 0.0
            batch_mean_token_accuracy = 0.0
            batch_mean_correct_tokens = 0.0
            batch_mean_f1_score = 0.0

        results["batch_metrics"][batch_size] = {
            "rouge_l": batch_mean_rouge_l,
            "rouge_1": batch_mean_rouge_1,
            "rouge_2": batch_mean_rouge_2,
            "token_accuracy": batch_mean_token_accuracy,
            "correct_tokens": batch_mean_correct_tokens,
            "f1_score": batch_mean_f1_score
        }
        results["list_metrics"][batch_size] = batch_list_metrics
        results["details"].append({
            "batch_size": batch_size,
            "mean_rouge_l": batch_mean_rouge_l,
            "mean_rouge_1": batch_mean_rouge_1,
            "mean_rouge_2": batch_mean_rouge_2,
            "mean_token_accuracy": batch_mean_token_accuracy,
            "mean_correct_tokens": batch_mean_correct_tokens,
            "mean_f1_score": batch_mean_f1_score,
            "list_details": batch_details
        })

    if all_rouge_l:
        results["mean_metrics"]["rouge_l"] = sum(all_rouge_l) / len(all_rouge_l)
        results["mean_metrics"]["rouge_1"] = sum(all_rouge_1) / len(all_rouge_1)
        results["mean_metrics"]["rouge_2"] = sum(all_rouge_2) / len(all_rouge_2)
        results["mean_metrics"]["token_accuracy"] = sum(all_token_accuracy) / len(all_token_accuracy)
        results["mean_metrics"]["correct_tokens"] = sum(all_correct_tokens) / len(all_correct_tokens)
        results["mean_metrics"]["f1_score"] = sum(all_f1_score) / len(all_f1_score)

    return results


def calculate_multiple_methods(ref_sentences: Dict[str, List[List[str]]], recon_methods: Dict[str, Dict[str, List[List[str]]]]) -> Dict[str, Dict[str, Any]]:
    all_results = {}

    for method_name, recon_dict in recon_methods.items():
        method_results = calculate_method_metrics(ref_sentences, recon_dict)
        all_results[method_name] = method_results

    return all_results


class MetricsResult:
    def __init__(self, results: Dict[str, Dict[str, Any]]):
        self._results = results

    def get_method_names(self) -> List[str]:
        return list(self._results.keys())

    def get_batch_sizes(self, method_name: str) -> List[str]:
        if method_name not in self._results:
            return []

        if "batch_metrics" in self._results[method_name]:
            return sorted(self._results[method_name]["batch_metrics"].keys(), key=lambda x: int(x))
        return []

    def get_method_mean_metrics(self, method_name: str) -> Optional[Dict[str, float]]:
        if method_name in self._results:
            return self._results[method_name].get("mean_metrics", {})
        return None

    def get_batch_metrics(self, method_name: str, batch_size: str) -> Optional[Dict[str, float]]:
        if method_name in self._results and batch_size in self._results[method_name]["batch_metrics"]:
            return self._results[method_name]["batch_metrics"][batch_size]
        return None

    def print_summary(self):
        print("\n" + "=" * 120)
        print("文本重建指标汇总 (每个方法的指标是5个列表的均值)")
        print("=" * 120)
        print(f"{'方法':<20} {'总均值F1':<12} {'总均值R-L':<12} {'总均值N_c':<12} {'Batch Size':<12} {'Batch F1':<12} {'Batch R-L':<12} {'Batch N_c':<12}")
        print("-" * 120)

        for method_name in self.get_method_names():
            mean_metrics = self.get_method_mean_metrics(method_name)
            batch_sizes = self.get_batch_sizes(method_name)

            if batch_sizes:
                first_batch = batch_sizes[0]
                batch_metrics = self.get_batch_metrics(method_name, first_batch)

                print(f"{method_name:<20} {mean_metrics.get('f1_score', 0.0):<12.4f} {mean_metrics['rouge_l']:<12.4f} {mean_metrics['correct_tokens']:<12.4f} {first_batch:<12} {batch_metrics.get('f1_score', 0.0):<12.4f} {batch_metrics['rouge_l']:<12.4f} {batch_metrics['correct_tokens']:<12.4f}")

                for batch_size in batch_sizes[1:]:
                    batch_metrics = self.get_batch_metrics(method_name, batch_size)
                    print(f"{'':<20} {'':<12} {'':<12} {'':<12} {batch_size:<12} {batch_metrics.get('f1_score', 0.0):<12.4f} {batch_metrics['rouge_l']:<12.4f} {batch_metrics['correct_tokens']:<12.4f}")
            else:
                print(f"{method_name:<20} {mean_metrics.get('f1_score', 0.0):<12.4f} {mean_metrics['rouge_l']:<12.4f} {mean_metrics['correct_tokens']:<12.4f} {'无数据':<12} {'-':<12} {'-':<12} {'-':<12}")

        print("=" * 120)

    def save_to_json(self, output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self._results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")


def load_results_data() -> Tuple[Dict[str, List[List[str]]], Dict[str, Dict[str, List[List[str]]]]]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_dirs = [
        os.path.join(base_dir, "Naive", "DP", "results"),
        os.path.join(base_dir, "Opt", "DP", "results"),
        os.path.join(base_dir, "Naive", "GP", "results"),
        os.path.join(base_dir, "Opt", "GP", "results"),
    ]

    ref_sentences: Dict[str, List[List[str]]] = {}
    recon_methods: Dict[str, Dict[str, List[List[str]]]] = {}

    for results_dir in script_dirs:
        if not os.path.isdir(results_dir):
            continue
        for filename in sorted(os.listdir(results_dir)):
            if not filename.endswith(".json"):
                continue
            method_name = filename[:-len(".json")].replace("NAive", "Naive").replace("_", "-")
            file_path = os.path.join(results_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    records = json.load(f)
                except json.JSONDecodeError:
                    continue

            method_refs: Dict[str, List[List[str]]] = {}
            method_recons: Dict[str, List[List[str]]] = {}
            for record in records:
                batch_key = str(record.get("batch_size", 1))
                method_refs.setdefault(batch_key, []).append([str(s) for s in record.get("reference_text", [])])
                method_recons.setdefault(batch_key, []).append([str(s) for s in record.get("reconstruct_text", [])])

            if not method_refs:
                continue
            recon_methods[method_name] = method_recons
            if not ref_sentences:
                ref_sentences.update(method_refs)

    return ref_sentences, recon_methods

def main():
    print("计算指标...")
    ref_sentences, recon_methods = load_results_data()

    if not ref_sentences or not recon_methods:
        print("未找到结果文件，请先运行攻击脚本生成 results/*.json")
        return

    results = calculate_multiple_methods(ref_sentences, recon_methods)

    metrics_result = MetricsResult(results)

    metrics_result.print_summary()

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_results.json")
    metrics_result.save_to_json(output_file)

if __name__ == "__main__":
    main()
