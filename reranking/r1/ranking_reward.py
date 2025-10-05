# rerank_reward_rel_ndcg_kendall_top1.py
import re
import math
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple

# -----------------------------
# Logging utilities
# -----------------------------

def _save_format_error_log(reward_input: Dict[str, Any]) -> None:
    """포맷 에러 시 reward_input을 로그 파일로 저장 (최근 100개만 유지)"""
    log_dir = "format_error_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "format_errors.jsonl")
    
    # 현재 시간과 함께 로그 엔트리 생성
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "reward_input": reward_input
    }
    
    # 기존 로그 읽기
    existing_logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        existing_logs.append(json.loads(line.strip()))
        except (json.JSONDecodeError, FileNotFoundError):
            existing_logs = []
    
    # 새 로그 추가
    existing_logs.append(log_entry)
    
    # 최근 100개만 유지
    if len(existing_logs) > 100:
        existing_logs = existing_logs[-100:]
    
    # 파일에 저장
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            for log_entry in existing_logs:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Warning: Failed to save format error log: {e}")

# -----------------------------
# Parsing & format validation
# -----------------------------

# ANSWER_RE = re.compile(
#     r"<think>\s*<content>.*?</content>\s*<contrast>.*?</contrast>\s*<summary>.*?</summary>\s*</think>\s*"
#     r"<answer>\s*\[(\d)\]\s*>\s*\[(\d)\]\s*>\s*\[(\d)\]\s*>\s*\[(\d)\]\s*>\s*\[(\d)\]\s*</answer>\s*$",
#     re.DOTALL
# )

ANSWER_RE = re.compile(
    r"<think>.*?</think>\s*"
    r"<answer>\s*\[([1-5])\]\s*>\s*\[([1-5])\]\s*>\s*\[([1-5])\]\s*>\s*\[([1-5])\]\s*>\s*\[([1-5])\]\s*</answer>\s*$",
    re.DOTALL
)

def parse_rank_from_response(response: str) -> Tuple[bool, List[int]]:
    """
    Returns:
      (format_ok, pi) where pi is a 0-based permutation list of length 5
      e.g., [0,2,3,4,1] means 1st=idx0, 2nd=idx2, ...
    """
    m = ANSWER_RE.fullmatch(response or "")
    if not m:
        return False, []
    order = [int(m.group(i)) for i in range(1, 6)]
    # ensure it's a permutation of {1..5}
    if sorted(order) != [1,2,3,4,5]:
        return False, []
    # convert to 0-based
    return True, [x-1 for x in order]

# -----------------------------
# Utilities
# -----------------------------

def gains_from_a(a: List[float], eps: float = 1e-12) -> List[float]:
    """Min–Max → [0,1], order preserved, non-negative gains."""
    amin, amax = min(a), max(a)
    denom = (amax - amin) + eps
    return [(x - amin) / denom for x in a]

def ndcg_at5(pi: List[int], g: List[float]) -> float:
    """nDCG@5 with DCG discount 1/log2(i+1)."""
    disc = [1.0/math.log2(i+2) for i in range(5)]  # i=0..4 -> positions 1..5
    dcg  = sum(g[pi[i]] * disc[i] for i in range(5))
    ideal = sorted(g, reverse=True)
    idcg = sum(ideal[i] * disc[i] for i in range(5))
    return dcg / (idcg + 1e-12)

def relative_ndcg_reward(pi: List[int], a: List[float]) -> float:
    """
    Relative improvement over IV2 initial ranking.
    R in [0,1], 0 = as good as baseline (or worse), 1 = as good as ideal.
    (남겨두되, compute_reward에서는 절대 nDCG를 사용)
    """
    g = gains_from_a(a)
    pi0 = sorted(range(5), key=lambda i: -a[i])        # baseline by IV2
    nd   = ndcg_at5(pi, g)
    nd0  = ndcg_at5(pi0, g)
    # ideal nDCG with gains is 1.0 by definition
    if nd <= nd0:
        return 0.0
    return (nd - nd0) / (1.0 - nd0 + 1e-12)

def weighted_kendall_tau(pi: List[int], a: List[float], gamma: float = 1.0) -> float:
    """
    Weighted Kendall’s tau in [0,1] after linear mapping.
    pair weight w_ij = |a_i - a_j|^gamma
    sign uses baseline order induced by a (larger a should appear earlier)
    """
    # ranks in produced permutation
    rank = [0]*5
    for pos, idx in enumerate(pi):
        rank[idx] = pos + 1  # 1..5

    num, den = 0.0, 0.0
    for i in range(5):
        for j in range(i+1, 5):
            w = abs(a[i] - a[j]) ** gamma
            den += w
            # desired sign: a_i > a_j  => i should be BEFORE j -> rank[i] < rank[j]
            s1 = 1 if (a[i] > a[j]) else (-1 if a[i] < a[j] else 0)
            s2 = 1 if (rank[j] - rank[i]) > 0 else (-1 if (rank[j] - rank[i]) < 0 else 0)
            num += w * (1 if s1 == s2 else (-1 if s1 * s2 != 0 else 0))
    if den <= 0:
        return 0.5  # neutral
    tau = num / den  # in [-1,1]
    return 0.5 * (tau + 1.0)  # map to [0,1]

def top1_acc(pi: List[int], a: List[float]) -> float:
    """1.0 if model's top-1 equals IV2's top-1; else 0.0"""
    top_iv2 = max(range(5), key=lambda i: a[i])
    if top_iv2 == pi[0]:
        return 1.0
    elif top_iv2 in pi[2:]:
        return -1.0
    else:
        return 0.0

# -----------------------------
# Main reward API (single / batch)
# -----------------------------

def compute_reward(
    reward_input: Dict[str, Any],
    weights: Dict[str, float] = None,
    kendall_gamma: float = 1.0,
) -> Dict[str, float]:
    """
    Inputs (example):
      reward_input = {
        "response": "<think>...<answer> [1] > [3] > [4] > [5] > [2] </answer>",
        "ground_truth": {
          "order_sim_desc": [2,4,3,5,1],                # (optional) GT 순서 (1-based cand_id)
          "correct_cand_id": 2,                         # (optional) GT Top-1 cand_id
          "cand_sim_scores": {1: 0.82, 2: 0.10, 3: 0.77, 4: 0.40, 5: 0.31}
        }
      }
    Returns: dict with overall and components in [0,1]
    """
    # if weights is None:
    #     # keep it simple & stable: nDCG dominates (키 이름은 호환을 위해 rel_ndcg 유지)
    #    # weights = {"rel_ndcg": 1.0, "kendall": 0.2, "top1": 0.2, "format": 0.02}
    #     weights = {"top1": 1.0, "format": 1.0}

    resp = reward_input.get("response", "")
    ground_truth = reward_input.get("ground_truth", {})
    cand_sim_scores = ground_truth.get("cand_sim_scores", {})
    correct_cand_id = ground_truth.get("correct_cand_id", None)
    gt_order_1based = ground_truth.get("order_sim_desc", None)

    # cand_sim_scores를 리스트로 변환 (candidate_id 1~5 순서)
    a = [cand_sim_scores.get(i, 0.0) for i in range(1, 6)]
    if len(a) != 5:
        raise ValueError("cand_sim_scores must contain scores for candidates 1-5.")

    format_ok, pi = parse_rank_from_response(resp)
    format_score = 1.0 if format_ok else 0.0
    
    # 포맷 실패 시 로그 저장
    if not format_ok:
        _save_format_error_log(reward_input)

    # 포맷 실패 시: 0점 처리
    if not format_ok:
        # format 실패 시 모든 점수를 0으로 설정
        return {
            "overall": 0.0,
            #"rel_ndcg": 0.0,
            #"kendall": 0.0,
            "top1": 0.0,
            "format": 0.0
        }

    # --- 보상 구성 ---
    # 1) 절대 nDCG (gains: min-max 정규화)
    # g = gains_from_a(a)
    #r_rel = ndcg_at5(pi, g)  # 키 이름 호환을 위해 rel_ndcg로 계속 사용

    # 2) Kendall (sim 가중 버전 그대로 사용)
    r_knd = weighted_kendall_tau(pi, list(map(float, a)), gamma=kendall_gamma)

    # 3) Top-1 (GT 제공 시 우선, 없으면 sim 최대값 기준)
    if isinstance(correct_cand_id, int) and 1 <= correct_cand_id <= 5:
        if correct_cand_id-1 == pi[0]:
            r_t1 = 1.0
        elif correct_cand_id-1 in  pi[2:]:
            r_t1 = -1.0
        else:
            r_t1 = 0.0
    else:
        r_t1  = top1_acc(pi, a)
    # print(correct_cand_id, pi, r_t1)

    # Weighted blend (then clamp to [0,1])
    num = (#weights["rel_ndcg"] * r_rel +
           weights["kendall"]  * r_knd +
           weights["top1"]     * r_t1  +
           weights["format"]   * format_score)
    
    overall = num

    return {
        "overall": overall,
        #"rel_ndcg": r_rel,      # 이제 '절대' nDCG 점수
        "kendall": r_knd,
        "top1": r_t1,
        "format": format_score,
    }

def compute_rewards_batch(
    batch_inputs: List[Dict[str, Any]],
    weights: Dict[str, float] = None,
    kendall_gamma: float = 1.0,
) -> List[Dict[str, float]]:
    """
    Batch version for GRPO group sampling.
    If zscore_normalize=True, z-normalize overall within the batch and squish to [0,1].
    """

    weights = {"top1": 1.0, "format": 0.0, "kendall": 0.0}

    outs = [compute_reward(x, weights=weights, kendall_gamma=kendall_gamma) for x in batch_inputs]
    # if zscore_normalize and len(outs) >= 2:
    #     vals = [o["overall"] for o in outs]
    #     mu = sum(vals)/len(vals)
    #     var = sum((v-mu)**2 for v in vals)/(len(vals)-1)
    #     sd = math.sqrt(var) if var > 0 else 1.0
    #     # z -> [0,1] via clip to [-2,2]
    #     for o, v in zip(outs, vals):
    #         z = (v - mu) / sd
    #         z = max(-2.0, min(2.0, z))
    #         o["overall"] = 0.5 + 0.25 * z
    return outs
