"""
评估指标计算工具函数
参照开源代码中的方法，计算重建文本的评估指标
"""
from collections import Counter
from itertools import combinations, permutations
from typing import List, Dict, Any, Tuple


def lcs_length(x: List[str], y: List[str]) -> int:
    """
    计算两个序列的最长公共子序列（LCS）长度
    """
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
    """
    标准化文本，统一处理标点符号和空格
    """
    # 替换多个连续空格为单个空格
    text = ' '.join(text.split())
    # 统一标点符号格式：将所有标点符号与前面的单词分开
    text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
    text = text.replace(' ;', ';').replace(' :', ':').replace(' \'', '\'').replace(' "', '"')
    text = text.replace('\'s', '\'s').replace('s\'', 's\'')
    # 确保标点符号与前面的单词分开
    punctuation = [',', '.', '!', '?', ';', ':', "'", '"']
    for p in punctuation:
        text = text.replace(p, ' ' + p)
    # 再次替换多个连续空格为单个空格
    text = ' '.join(text.split())
    return text


def normalized_words(text: str) -> List[str]:
    """Return normalized word sequence used by position-sensitive metrics."""
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
    """
    计算ROUGE-1指标，计算unigram的重叠度（不去重，按 multiset 计数）

    Args:
        reference: 参考句子（ground truth）
        candidate: 候选句子（reconstructed）

    Returns:
        ROUGE-1值
    """
    if not reference:
        return 0.0

    # 标准化文本
    reference = normalize_text(reference)
    candidate = normalize_text(candidate)

    ref_words = reference.strip().split()
    cand_words = candidate.strip().split()

    if not ref_words:
        return 0.0

    # 不去重：使用 Counter 统计每个 unigram 的出现次数，重叠取最小计数
    ref_counts = Counter(ref_words)
    cand_counts = Counter(cand_words)
    overlap = sum((ref_counts & cand_counts).values())

    return overlap / len(ref_words)


def calculate_rouge_2(reference: str, candidate: str) -> float:
    """
    计算ROUGE-2指标，计算bigram的重叠度（不去重，按 multiset 计数）

    Args:
        reference: 参考句子（ground truth）
        candidate: 候选句子（reconstructed）

    Returns:
        ROUGE-2值
    """
    if not reference:
        return 0.0

    # 标准化文本
    reference = normalize_text(reference)
    candidate = normalize_text(candidate)

    ref_words = reference.strip().split()
    cand_words = candidate.strip().split()

    if len(ref_words) < 2:
        return 0.0

    # 计算bigram
    def get_bigrams(words):
        return [(words[i], words[i + 1]) for i in range(len(words) - 1)]

    # 不去重：使用 Counter 统计每个 bigram 的出现次数，重叠取最小计数
    ref_bigrams = Counter(get_bigrams(ref_words))
    cand_bigrams = Counter(get_bigrams(cand_words))

    if not ref_bigrams:
        return 0.0

    overlap = sum((ref_bigrams & cand_bigrams).values())
    # 分母使用 bigram 总数（不去重），即 len(ref_words) - 1
    return overlap / (len(ref_words) - 1)


def calculate_rouge_l(reference: str, candidate: str) -> float:
    """
    计算ROUGE-L指标，按照论文中的公式：R-L = LCS(T_gt, T_rec) / L_gt
    
    Args:
        reference: 参考句子（ground truth）
        candidate: 候选句子（reconstructed）
    
    Returns:
        ROUGE-L值
    """
    if not reference:
        return 0.0
    
    # 标准化文本
    reference = normalize_text(reference)
    candidate = normalize_text(candidate)
    
    ref_words = reference.strip().split()
    cand_words = candidate.strip().split()
    
    if not ref_words:
        return 0.0
    
    lcs = lcs_length(ref_words, cand_words)
    return lcs / len(ref_words)


def calculate_token_accuracy(reference: str, candidate: str) -> Tuple[float, int]:
    """
    计算Token准确率和正确Token数，按照论文中的公式：
    A_t = sum(1(t^g_i = t^r_i)) / L_gt
    N_c = sum(1(t^g_i = t^r_i))
    
    Args:
        reference: 参考句子（ground truth）
        candidate: 候选句子（reconstructed）
    
    Returns:
        (A_t, N_c) 元组，分别是Token准确率和正确Token数
    """
    if not reference:
        return 0.0, 0
    
    # 标准化文本
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
    """
    计算F1分数，按照论文公式(33)：
    F1 = 2 * Σ 1(t_i^g = t_i^r) / (L_gt + L_rec)

    Args:
        correct_tokens: 位置精确匹配的token数 Nt
        total_tokens: 原文token数 L_gt
        recon_token_count: 重建文本token数 L_rec

    Returns:
        F1分数
    """
    denominator = total_tokens + recon_token_count
    if denominator == 0:
        return 0.0
    return 2.0 * correct_tokens / denominator


def calculate_single_sentence_metrics(original_text: str, reconstructed_candidates: List[str]) -> Dict[str, Any]:
    """
    计算单个句子的所有指标，选择最佳的重建结果
    
    Args:
        original_text: 原始文本（ground truth）
        reconstructed_candidates: 重建文本候选列表
    
    Returns:
        包含所有指标的字典
    """
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
        
        # 选择准确率最高的重建结果
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
    """
    计算批次内的平均指标
    
    Args:
        ref_sentences: 参考句子列表
        recon_sentences: 重建句子列表，每个元素可以是字符串或字符串列表
    
    Returns:
        批次的平均指标和详细信息
    """
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
    
    # 收集所有重建文本（包括单个单词）
    all_recon_texts = []
    for recon_item in recon_sentences:
        if isinstance(recon_item, str):
            # 允许单个单词作为重建结果
            if recon_item.strip():
                all_recon_texts.append(recon_item)
        elif isinstance(recon_item, list):
            for s in recon_item:
                if isinstance(s, str) and s.strip():
                    all_recon_texts.append(s)
    
    # batch size > 1 时，先在批次内确定原文-重建文本对应关系，再统计指标。
    ref_matches = _best_text_assignment(ref_sentences, all_recon_texts)
    
    # 处理所有原始文本
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
            # 没有找到对应的重建文本
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


def print_metrics_summary(metrics: Dict[str, Any], method_name: str = ""):
    """
    打印评估指标摘要
    
    Args:
        metrics: 评估指标字典
        method_name: 方法名称
    """
    if method_name:
        print(f"\n{'=' * 60}")
        print(f"{method_name} 评估指标摘要")
        print(f"{'=' * 60}")
    
    print(f"ROUGE-1: {metrics['mean_rouge_1']:.4f}")
    print(f"ROUGE-2: {metrics['mean_rouge_2']:.4f}")
    print(f"ROUGE-L: {metrics['mean_rouge_l']:.4f}")
    print(f"Token准确率: {metrics['mean_token_accuracy']:.4f}")
    print(f"平均正确Token数: {metrics['mean_correct_tokens']:.2f}")
    print(f"F1分数: {metrics['mean_f1_score']:.4f}")
    print(f"句子数量: {metrics['sentence_count']}")
    print(f"总正确Token数: {metrics['total_correct_tokens']}")
    
    # 打印详细句子信息
    if metrics.get("sentence_details"):
        print(f"\n{'=' * 60}")
        print("句子详细评估结果")
        print(f"{'=' * 60}")
        for i, detail in enumerate(metrics["sentence_details"], 1):
            print(f"\n句子 {i}:")
            print(f"  原始: {detail['original']}")
            print(f"  重建: {detail['reconstructed']}")
            print(f"  ROUGE-1: {detail['rouge_1']:.4f}, ROUGE-2: {detail['rouge_2']:.4f}, ROUGE-L: {detail['rouge_l']:.4f}")
            print(f"  Token准确率: {detail['token_accuracy']:.4f}, 正确Token数: {detail['correct_tokens']}/{detail['total_tokens']}")
            print(f"  F1分数: {detail['f1_score']:.4f}")


def recompute_metrics_from_results(results_path: str, verbose: bool = True) -> int:
    """遍历 results.json，仅重新计算每条记录的评估指标（不重跑实验）。

    用于指标计算逻辑变更后（如 ROUGE-1/2 去重策略调整），
    在不重新执行攻击的前提下更新已有结果的 metrics 字段。

    Args:
        results_path: results.json 文件路径
        verbose: 是否打印进度信息

    Returns:
        重新计算指标的记录数
    """
    import json
    from pathlib import Path

    path = Path(results_path)
    if not path.exists() or path.stat().st_size == 0:
        if verbose:
            print(f"[Recompute] 结果文件不存在或为空: {results_path}")
        return 0

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        if verbose:
            print(f"[Recompute] 结果文件格式非列表，跳过: {results_path}")
        return 0

    recomputed = 0
    for idx, record in enumerate(records):
        ref_sentences = record.get("reference_text") or []
        attack = record.get("attack") or {}
        recon_sentences = attack.get("reconstructed_text") or []

        # 缺少参考文本或重建文本时无法重算，跳过
        if not ref_sentences:
            continue

        metrics = calculate_batch_metrics(ref_sentences, recon_sentences)
        record["metrics"] = metrics
        recomputed += 1

        if verbose:
            method = record.get("method_name", "?")
            client = record.get("client", "?")
            batch = record.get("batch_idx", "?")
            print(f"[Recompute] #{idx + 1} method={method} client={client} batch={batch} "
                  f"R-1={metrics['mean_rouge_1']:.4f} R-2={metrics['mean_rouge_2']:.4f} "
                  f"R-L={metrics['mean_rouge_l']:.4f}")

    if recomputed > 0:
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        if verbose:
            print(f"[Recompute] 已重新计算 {recomputed}/{len(records)} 条记录的指标，写回 {results_path}")
    elif verbose:
        print(f"[Recompute] 没有可重算的记录（共 {len(records)} 条）")

    return recomputed