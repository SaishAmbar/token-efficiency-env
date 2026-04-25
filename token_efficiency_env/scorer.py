"""
Token Efficiency Scorer — 6-Component Reward Function (Phase 3)

Components and weights:

    1. Correctness        (55%) — LLM judge via huggingface_hub.InferenceClient
                                  (default: meta-llama/Llama-3.1-8B-Instruct).
                                  Falls back to keyword scoring on failure.
    2. Efficiency         (15%) — tokens used vs the IDEAL count for the
                                  question's complexity (NOT vs the model's
                                  self-allocated budget — that was the source
                                  of the Phase 2 contradiction with #3).
    3. Self-Assessment    (15%) — asymmetric: mild reward for tight, honest
                                  upper-bound predictions; steep penalty for
                                  overshooting the budget the model itself set.
    4. Redundancy Penalty  (5%) — penalises repetitive answers
    5. Keyword Verification(5%) — word-boundary keyword match (sanity check)
    6. Format Quality      (5%) — clean, well-structured response

Total: 100%

Phase 3 changes vs Phase 2:
    * DROPPED ``budget_reasonableness`` (10% → 0%). It only depends on
      (allocated_budget, complexity), so the model could max it out by
      memorising the prior — no learning signal.
    * ``efficiency`` rebased: now compares tokens_used to a complexity-based
      IDEAL count, not to allocated_budget. Previously efficiency rewarded
      "use less than your budget" while self_assessment rewarded "use exactly
      your budget" — the two fought each other. They are now orthogonal.
    * ``self_assessment`` switched from symmetric to asymmetric:
        - Under-budget: small slack penalty (caps at -0.3 of full credit).
        - Over-budget: steep linear penalty (slope -2.0) floored at -0.5.
      This rewards the model for predicting a TIGHT BUT HONEST upper bound
      on its token usage — exactly the meta-cognition skill we want.
    * Reweighted toward correctness (40% → 55%) so noise from minor components
      can't dominate the gradient.

Reward range from this scorer: roughly [-0.5, ~0.97].
The env layer adds harder ``-1.0`` cliffs for malformed responses; those
short-circuit before this function is called.
"""

from __future__ import annotations

import re
from collections import Counter

# Import the judge factory. Use a try/except so this module works both when
# loaded as ``token_efficiency_env.scorer`` (installed package) and when
# loaded as a top-level ``scorer`` module (uvicorn from /app/env).
try:
    from .judge import get_judge
except ImportError:  # pragma: no cover — exercised only at container start
    from judge import get_judge  # type: ignore[no-redef]


# Maps complexity labels to the IDEAL token count for a good answer.
# Numbers chosen to be ~1.5× the typical concise correct answer length —
# enough room to be informative, not so much that verbose answers get a free
# pass. Tune in Phase 4 if smoke testing shows a bias.
COMPLEXITY_IDEAL_TOKENS = {
    "easy": 15,     # "Paris.", "100 degrees Celsius."
    "medium": 60,   # 2-3 sentence explanations
    "hard": 130,    # Multi-step reasoning
}


def _self_assessment_asymmetric(allocated_budget: int, tokens_used: int) -> float:
    """Asymmetric reward for budget-vs-actual prediction.

    Behaviour by zone:
      * tokens_used == budget          → 1.00  (perfect prediction)
      * tokens_used <  budget (slack)  → 1.00 - 0.30 * slack_ratio
                                         (max -0.30, capped at 0.70 floor)
      * tokens_used >  budget (over)   → 1.00 - 2.00 * overshoot_ratio
                                         (slope -2.0, floored at -0.50)

    Calibration examples (budget=20):
        used=20  → 1.00   used=18 → 0.97   used=2  → 0.73
        used=22  → 0.80   used=30 → 0.00   used=60 → -0.50 (floor)

    Rationale: the asymmetry penalises the failure mode that hurts users
    (lying about an upper bound) much harder than the failure mode that
    only wastes a bit of capacity (slight under-prediction). Combined with
    ``efficiency`` (which separately rewards low absolute token counts),
    the two components together push the model toward "predict the smallest
    honest upper bound on your token usage".
    """
    if allocated_budget <= 0 or tokens_used < 0:
        return 0.0
    if tokens_used <= allocated_budget:
        slack_ratio = (allocated_budget - tokens_used) / allocated_budget
        return max(0.70, 1.00 - 0.30 * slack_ratio)
    overshoot_ratio = (tokens_used - allocated_budget) / allocated_budget
    return max(-0.50, 1.00 - 2.00 * overshoot_ratio)


def _efficiency_absolute(tokens_used: int, complexity: str) -> float:
    """Reward absolute brevity vs the complexity-ideal token count.

    Behaviour:
      * tokens_used <= ideal           → 1.00
      * tokens_used >  ideal           → 1.00 - 0.50 * overshoot_ratio
                                         (slope -0.5, floored at -0.50)

    No reward bonus for going under the ideal — we don't want to incentivise
    "0 tokens" hacks (the env's near-empty-answer cliff handles those, but
    keeping efficiency monotonic-non-decreasing-in-brevity here lets the
    correctness component dominate the trade-off naturally).
    """
    ideal = COMPLEXITY_IDEAL_TOKENS.get(complexity, 60)
    if tokens_used <= ideal:
        return 1.00
    overshoot_ratio = (tokens_used - ideal) / ideal
    return max(-0.50, 1.00 - 0.50 * overshoot_ratio)


def score_answer(
    prompt: str,
    response: str,
    allocated_budget: int,
    tokens_used: int,
    complexity: str = "medium",
    expected_keywords: list[str] | None = None,
) -> dict:
    """Compute the 6-component reward for a model response.

    Args:
        prompt:            The question that was asked.
        response:          The model's answer (text inside <answer> tags).
        allocated_budget:  Budget the model self-allocated (from <budget>).
        tokens_used:       Actual token count of the response.
        complexity:        Question complexity ("easy", "medium", "hard").
        expected_keywords: List of keyword stems used by the keyword component
                           AND as a fallback if the LLM judge call fails.

    Returns:
        ``{"reward": float, "details": {component_name: score, ...}}``.
        Reward is roughly in ``[-0.5, ~0.97]``. The env layer applies harder
        ``-1.0`` cliffs for malformed responses BEFORE calling this function.
    """
    if expected_keywords is None:
        expected_keywords = []

    details: dict[str, float] = {}

    # ─── 1. Correctness via LLM judge (55%) ─────────────────────────
    correctness = get_judge().score_correctness(
        prompt=prompt,
        answer=response,
        expected_keywords=expected_keywords,
    )
    correctness = max(0.0, min(1.0, float(correctness)))
    details["correctness"] = round(correctness, 4)

    # ─── 2. Efficiency (15%) — vs complexity ideal, NOT vs budget ──
    efficiency = _efficiency_absolute(tokens_used, complexity)
    details["efficiency"] = round(efficiency, 4)

    # ─── 3. Self-Assessment (15%) — asymmetric ─────────────────────
    self_assessment = _self_assessment_asymmetric(allocated_budget, tokens_used)
    details["self_assessment"] = round(self_assessment, 4)

    # ─── 4. Redundancy (5%) ────────────────────────────────────────
    words = response.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if len(words) >= 2:
            bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
            most_common_count = Counter(bigrams).most_common(1)[0][1]
            bigram_penalty = min(most_common_count / max(len(bigrams), 1), 1.0)
        else:
            bigram_penalty = 0.0
        redundancy_score = unique_ratio * 0.7 + (1.0 - bigram_penalty) * 0.3
    else:
        redundancy_score = 0.0
    details["redundancy"] = round(redundancy_score, 4)

    # ─── 5. Keyword Verification (5%) ──────────────────────────────
    if expected_keywords:
        answer_lower = response.lower()
        matches = sum(
            1 for kw in expected_keywords
            if re.search(rf"\b{re.escape(kw.lower())}\b", answer_lower)
        )
        keyword_score = matches / len(expected_keywords)
    else:
        keyword_score = 1.0
    details["keyword_verification"] = round(keyword_score, 4)

    # ─── 6. Format Quality (5%) ────────────────────────────────────
    format_score = 1.0
    if "  " in response or response != response.strip():
        format_score -= 0.2
    if complexity in ("medium", "hard") and len(words) < 5:
        format_score -= 0.3
    if complexity == "easy" and len(words) > 30:
        format_score -= 0.3
    prompt_words = set(prompt.lower().split())
    answer_words = set(response.lower().split())
    if len(answer_words) > 0:
        overlap = len(prompt_words & answer_words) / len(answer_words)
        if overlap > 0.7:
            format_score -= 0.3
    format_score = max(0.0, min(1.0, format_score))
    details["format_quality"] = round(format_score, 4)

    # ─── Final Weighted Reward ──────────────────────────────────────
    reward = (
        0.55 * correctness +
        0.15 * efficiency +
        0.15 * self_assessment +
        0.05 * redundancy_score +
        0.05 * keyword_score +
        0.05 * format_score
    )
    reward = round(float(reward), 4)
    details["final_reward"] = reward

    return {"reward": reward, "details": details}
