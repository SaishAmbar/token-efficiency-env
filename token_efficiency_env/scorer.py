"""
Token Efficiency Scorer — 7-Component Reward Function

Components and weights:
    1. Correctness         (40%) — LLM judge via Claude API
    2. Efficiency          (20%) — tokens used vs allocated budget
    3. Budget Reasonableness (10%) — budget matches question complexity
    4. Redundancy Penalty  (10%) — penalises repetitive answers
    5. Self-Assessment     (10%) — how accurately budget predicts actual usage
    6. Keyword Verification (5%) — pattern match for known answer fragments
    7. Format Quality       (5%) — clean, well-structured response

Total: 100%

Design rationale (aligned with Hackathon Self-Serve Guide):
    - Multiple independent reward functions (Guide §7, §8)
    - Keyword verification as secondary signal, not sole reliance on LLM judge (Guide §9)
    - Anti-hacking through redundancy detection (Guide §8)
"""

import anthropic
import os
import re
from collections import Counter

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Maps complexity labels to expected budget ratios (fraction of 200)
COMPLEXITY_BUDGET = {
    "easy": 0.15,    # ~30 tokens ideal
    "medium": 0.45,  # ~90 tokens ideal
    "hard": 0.80,    # ~160 tokens ideal
}


def score_answer(prompt, response, allocated_budget, tokens_used,
                 complexity="medium", expected_keywords=None):
    """
    Compute a multi-component reward for the model's response.

    Args:
        prompt:           The question that was asked
        response:         The model's answer (text inside <answer> tags)
        allocated_budget: The budget the model self-allocated (from <budget> tag)
        tokens_used:      Actual token count of the response
        complexity:       Question complexity level ("easy", "medium", "hard")
        expected_keywords: List of keyword stems to verify in the answer

    Returns:
        dict with keys:
            "reward"  — final weighted reward (float, roughly 0.0 to 1.0)
            "details" — dict of individual component scores for monitoring
    """
    if expected_keywords is None:
        expected_keywords = []

    details = {}

    # ─── Part 1: Correctness via Claude API (40%) ───────────────────
    try:
        judge = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": (
                    f"Question: {prompt}\n"
                    f"Answer: {response}\n"
                    "Rate this answer's correctness from 0.0 to 1.0.\n"
                    "Reply with only a number. Example: 0.7"
                )
            }]
        )
        correctness = float(judge.content[0].text.strip())
        correctness = max(0.0, min(1.0, correctness))
    except Exception:
        correctness = 0.0

    details["correctness"] = round(correctness, 4)

    # ─── Part 2: Efficiency (20%) ───────────────────────────────────
    if allocated_budget <= 0:
        efficiency = 0.0
    elif tokens_used <= allocated_budget:
        # used less than budget — good; closer to zero usage = better
        efficiency = 1.0 - (tokens_used / allocated_budget) * 0.3
    else:
        # went over own budget — penalise
        overshoot = (tokens_used - allocated_budget) / allocated_budget
        efficiency = max(-0.5, -0.5 * overshoot)

    details["efficiency"] = round(efficiency, 4)

    # ─── Part 3: Budget Reasonableness (10%) ────────────────────────
    # Uses the complexity label from PROMPT_BANK for more accurate matching
    expected_ratio = COMPLEXITY_BUDGET.get(complexity, 0.45)
    allocated_ratio = min(allocated_budget / 200, 1.0)
    budget_reasonableness = 1.0 - abs(allocated_ratio - expected_ratio)
    budget_reasonableness = max(0.0, budget_reasonableness)

    details["budget_reasonableness"] = round(budget_reasonableness, 4)

    # ─── Part 4: Redundancy Penalty (10%) ───────────────────────────
    # Measures unique information density — repeated content is wasteful
    words = response.lower().split()
    if len(words) > 0:
        unique_words = set(words)
        unique_ratio = len(unique_words) / len(words)
        # Also check for repeated phrases (bigrams)
        if len(words) >= 2:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            bigram_counts = Counter(bigrams)
            most_common_count = bigram_counts.most_common(1)[0][1]
            bigram_penalty = min(most_common_count / max(len(bigrams), 1), 1.0)
        else:
            bigram_penalty = 0.0
        # High unique_ratio = good, high bigram_penalty = bad
        redundancy_score = unique_ratio * 0.7 + (1.0 - bigram_penalty) * 0.3
    else:
        redundancy_score = 0.0  # empty answer

    details["redundancy"] = round(redundancy_score, 4)

    # ─── Part 5: Self-Assessment Accuracy (10%) ─────────────────────
    # How well did the model predict its own token usage?
    if allocated_budget > 0 and tokens_used > 0:
        prediction_error = abs(allocated_budget - tokens_used) / allocated_budget
        self_assessment = max(0.0, 1.0 - prediction_error)
    else:
        self_assessment = 0.0

    details["self_assessment"] = round(self_assessment, 4)

    # ─── Part 6: Keyword Verification (5%) ──────────────────────────
    # Secondary correctness check — does NOT rely on LLM judge
    if expected_keywords:
        answer_lower = response.lower()
        matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        keyword_score = matches / len(expected_keywords)
    else:
        keyword_score = 1.0  # no keywords defined → full marks

    details["keyword_verification"] = round(keyword_score, 4)

    # ─── Part 7: Format Quality (5%) ───────────────────────────────
    # Rewards clean, well-formed responses
    format_score = 1.0

    # Penalise excessive whitespace
    if "  " in response or response != response.strip():
        format_score -= 0.2

    # Penalise very short answers for non-easy questions
    if complexity in ("medium", "hard") and len(words) < 5:
        format_score -= 0.3

    # Penalise overly long answers for easy questions
    if complexity == "easy" and len(words) > 30:
        format_score -= 0.3

    # Penalise if answer just restates the question
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
        0.40 * correctness +
        0.20 * efficiency +
        0.10 * budget_reasonableness +
        0.10 * redundancy_score +
        0.10 * self_assessment +
        0.05 * keyword_score +
        0.05 * format_score
    )

    reward = round(float(reward), 4)
    details["final_reward"] = reward

    return {"reward": reward, "details": details}
