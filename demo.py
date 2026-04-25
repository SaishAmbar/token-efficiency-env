"""
TokenEfficiencyEnv — Interactive Demo
======================================
Demonstrates the full env lifecycle WITHOUT needing:
  - OpenEnv dependency
  - Anthropic API key
  - GPU or model weights

Simulates what happens during RL training with 3 example scenarios.
"""

import re
import sys
import os
import io
from collections import Counter

# Fix Windows console encoding for Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── Add project to path ────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "token_efficiency_env"))
from prompts import PROMPT_BANK

# ─── Constants (mirrored from env_server.py) ────────────────────────
MAX_TOKEN_LIMIT = 200
MIN_BUDGET = 1
MAX_BUDGET = MAX_TOKEN_LIMIT
MAX_ANSWER_TOKENS = 500
REPETITION_THRESHOLD = 0.6

COMPLEXITY_BUDGET = {
    "easy": 0.15,    # ~30 tokens ideal
    "medium": 0.45,  # ~90 tokens ideal
    "hard": 0.80,    # ~160 tokens ideal
}


# ─── Helpers ────────────────────────────────────────────────────────
def count_tokens_simple(text):
    """Rough token count (words × 1.3) — avoids needing the Qwen tokenizer."""
    return max(1, int(len(text.split()) * 1.3))


def mock_correctness(prompt, response, expected_keywords):
    """Stand-in for Claude API correctness judge — uses keyword overlap."""
    if not expected_keywords:
        return 0.5
    answer_lower = response.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(matches / len(expected_keywords), 2)


def score_demo(prompt, response, allocated_budget, tokens_used,
               complexity, expected_keywords):
    """Simplified 7-component scorer (no API calls)."""
    details = {}

    # 1. Correctness (40%) — mock via keywords
    correctness = mock_correctness(prompt, response, expected_keywords)
    details["correctness"] = correctness

    # 2. Efficiency (20%)
    if allocated_budget <= 0:
        efficiency = 0.0
    elif tokens_used <= allocated_budget:
        efficiency = 1.0 - (tokens_used / allocated_budget) * 0.3
    else:
        overshoot = (tokens_used - allocated_budget) / allocated_budget
        efficiency = max(-0.5, -0.5 * overshoot)
    details["efficiency"] = round(efficiency, 4)

    # 3. Budget Reasonableness (10%)
    expected_ratio = COMPLEXITY_BUDGET.get(complexity, 0.45)
    allocated_ratio = min(allocated_budget / 200, 1.0)
    budget_reasonableness = max(0.0, 1.0 - abs(allocated_ratio - expected_ratio))
    details["budget_reasonableness"] = round(budget_reasonableness, 4)

    # 4. Redundancy (10%)
    words = response.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if len(words) >= 2:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            bigram_counts = Counter(bigrams)
            bigram_penalty = min(bigram_counts.most_common(1)[0][1] / max(len(bigrams), 1), 1.0)
        else:
            bigram_penalty = 0.0
        redundancy_score = unique_ratio * 0.7 + (1.0 - bigram_penalty) * 0.3
    else:
        redundancy_score = 0.0
    details["redundancy"] = round(redundancy_score, 4)

    # 5. Self-Assessment (10%)
    if allocated_budget > 0 and tokens_used > 0:
        prediction_error = abs(allocated_budget - tokens_used) / allocated_budget
        self_assessment = max(0.0, 1.0 - prediction_error)
    else:
        self_assessment = 0.0
    details["self_assessment"] = round(self_assessment, 4)

    # 6. Keyword Verification (5%)
    if expected_keywords:
        answer_lower = response.lower()
        matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        keyword_score = matches / len(expected_keywords)
    else:
        keyword_score = 1.0
    details["keyword_verification"] = round(keyword_score, 4)

    # 7. Format Quality (5%)
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

    # Final weighted reward
    reward = (
        0.40 * correctness +
        0.20 * efficiency +
        0.10 * budget_reasonableness +
        0.10 * redundancy_score +
        0.10 * self_assessment +
        0.05 * keyword_score +
        0.05 * format_score
    )
    details["final_reward"] = round(reward, 4)

    return {"reward": round(reward, 4), "details": details}


# ─── Pretty Printing ────────────────────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

def bar(value, width=20):
    """Visual bar chart for a 0.0–1.0 value."""
    filled = int(value * width)
    empty = width - filled
    if value >= 0.7:
        color = GREEN
    elif value >= 0.4:
        color = YELLOW
    else:
        color = RED
    return f"{color}{'█' * filled}{'░' * empty}{RESET} {value:.2f}"


def print_header(text):
    print(f"\n{'═' * 60}")
    print(f" {BOLD}{CYAN}{text}{RESET}")
    print(f"{'═' * 60}")


def run_scenario(scenario_num, task, raw_response, description):
    """Run one complete env lifecycle scenario."""
    print_header(f"SCENARIO {scenario_num}: {description}")

    # ── Step 1: reset() ─────────────────────────────────────────
    print(f"\n{BOLD}1. env.reset(){RESET}")
    print(f"   {DIM}Picks a question from the prompt bank{RESET}")
    print(f"   {MAGENTA}Complexity:{RESET} {task['complexity']}")
    print(f"   {MAGENTA}Prompt:{RESET}     \"{task['prompt']}\"")
    print(f"   {MAGENTA}Keywords:{RESET}   {task.get('expected_keywords', [])}")
    print(f"   {MAGENTA}Token limit:{RESET} {MAX_TOKEN_LIMIT}")

    # ── Step 2: model generates ─────────────────────────────────
    print(f"\n{BOLD}2. Model generates response{RESET}")
    print(f"   {CYAN}{raw_response}{RESET}")

    # ── Step 3: env.step() — parse ──────────────────────────────
    print(f"\n{BOLD}3. env.step() — Parse & Validate{RESET}")

    budget_match = re.search(r"<budget>(\d+)</budget>", raw_response)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw_response, re.DOTALL)

    if not budget_match or not answer_match:
        print(f"   {RED}✗ BAD FORMAT — missing <budget> or <answer> tags{RESET}")
        print(f"   {RED}→ reward = -1.0 (episode ends immediately){RESET}")
        return

    allocated_budget = int(budget_match.group(1))
    answer = answer_match.group(1).strip()

    print(f"   ✓ Parsed budget: {allocated_budget}")
    print(f"   ✓ Parsed answer: \"{answer}\"")

    # Budget clamping
    clamped = max(MIN_BUDGET, min(MAX_BUDGET, allocated_budget))
    if clamped != allocated_budget:
        print(f"   {YELLOW}⚠ Budget clamped: {allocated_budget} → {clamped}{RESET}")
        allocated_budget = clamped
    else:
        print(f"   ✓ Budget in range [1, 200]")

    # Empty check
    if not answer or answer.isspace():
        print(f"   {RED}✗ EMPTY ANSWER → reward = -1.0{RESET}")
        return
    print(f"   ✓ Answer is not empty")

    # Repetition check
    words = answer.lower().split()
    if len(words) > 3:
        word_counts = Counter(words)
        most_common_word, most_common_count = word_counts.most_common(1)[0]
        ratio = most_common_count / len(words)
        if ratio > REPETITION_THRESHOLD:
            print(f"   {RED}✗ REPETITION: '{most_common_word}' is {ratio:.0%} of words → reward = -0.5{RESET}")
            return
    print(f"   ✓ No repetition detected")

    # Token count
    tokens_used = count_tokens_simple(answer)
    print(f"   ✓ Tokens used: ~{tokens_used}")

    if tokens_used > MAX_ANSWER_TOKENS:
        print(f"   {RED}✗ TOO LONG: {tokens_used} > {MAX_ANSWER_TOKENS} → reward = -0.5{RESET}")
        return
    print(f"   ✓ Under {MAX_ANSWER_TOKENS} token cap")

    # ── Step 4: 7-component scoring ─────────────────────────────
    print(f"\n{BOLD}4. 7-Component Scoring{RESET}")

    result = score_demo(
        prompt=task["prompt"],
        response=answer,
        allocated_budget=allocated_budget,
        tokens_used=tokens_used,
        complexity=task["complexity"],
        expected_keywords=task.get("expected_keywords", []),
    )

    details = result["details"]

    print(f"   ┌─────────────────────────────────────────────────┐")
    print(f"   │ Component              Weight   Score           │")
    print(f"   ├─────────────────────────────────────────────────┤")
    print(f"   │ Correctness            40%   {bar(details['correctness'])}  │")
    print(f"   │ Efficiency             20%   {bar(details['efficiency'])}  │")
    print(f"   │ Budget Reasonableness  10%   {bar(details['budget_reasonableness'])}  │")
    print(f"   │ Redundancy             10%   {bar(details['redundancy'])}  │")
    print(f"   │ Self-Assessment        10%   {bar(details['self_assessment'])}  │")
    print(f"   │ Keyword Verification    5%   {bar(details['keyword_verification'])}  │")
    print(f"   │ Format Quality          5%   {bar(details['format_quality'])}  │")
    print(f"   ├─────────────────────────────────────────────────┤")
    print(f"   │ {BOLD}FINAL REWARD          100%   {bar(result['reward'])}{RESET}  │")
    print(f"   └─────────────────────────────────────────────────┘")

    # ── Explain the result ──────────────────────────────────────
    reward = result["reward"]
    if reward >= 0.7:
        verdict = f"{GREEN}EXCELLENT — model is concise AND correct{RESET}"
    elif reward >= 0.5:
        verdict = f"{YELLOW}DECENT — room for improvement{RESET}"
    elif reward >= 0.3:
        verdict = f"{YELLOW}MEDIOCRE — model needs more training{RESET}"
    else:
        verdict = f"{RED}POOR — model is struggling{RESET}"

    print(f"\n   {BOLD}Verdict:{RESET} {verdict}")


# ═══════════════════════════════════════════════════════════════════
#                           DEMO SCENARIOS
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"""
{BOLD}{CYAN}
╔══════════════════════════════════════════════════════════════╗
║           TokenEfficiencyEnv — Interactive Demo             ║
║                                                              ║
║  Simulates 4 scenarios showing the full env lifecycle:      ║
║    1. Perfect concise answer (easy question)                ║
║    2. Verbose wasteful answer (easy question)               ║
║    3. Good medium-complexity answer                         ║
║    4. Bad format (missing tags)                             ║
╚══════════════════════════════════════════════════════════════╝
{RESET}""")

    # ── Scenario 1: Perfect concise answer ──────────────────────
    run_scenario(
        scenario_num=1,
        task=PROMPT_BANK[1],  # "What is the capital of France?" — easy
        raw_response="<budget>20</budget><answer>Paris.</answer>",
        description="Perfect concise answer (easy)",
    )

    # ── Scenario 2: Verbose answer to easy question ─────────────
    run_scenario(
        scenario_num=2,
        task=PROMPT_BANK[1],  # same question
        raw_response=(
            "<budget>150</budget><answer>The capital of France is Paris. "
            "Paris is a beautiful city located in the north-central part of "
            "France. It is known for its iconic landmarks such as the Eiffel "
            "Tower, the Louvre Museum, and the Arc de Triomphe. Paris has been "
            "the capital of France since the 10th century and is one of the "
            "most visited cities in the world.</answer>"
        ),
        description="Verbose wasteful answer (easy)",
    )

    # ── Scenario 3: Good medium answer ──────────────────────────
    run_scenario(
        scenario_num=3,
        task=PROMPT_BANK[8],  # "Explain what gravity is." — medium
        raw_response=(
            "<budget>80</budget><answer>Gravity is a fundamental force of "
            "attraction between objects with mass. The greater the mass, the "
            "stronger the gravitational pull. It keeps planets in orbit around "
            "stars and holds us on Earth's surface.</answer>"
        ),
        description="Good medium-complexity answer",
    )

    # ── Scenario 4: Bad format ──────────────────────────────────
    run_scenario(
        scenario_num=4,
        task=PROMPT_BANK[0],  # "What is 15% of 200?" — easy
        raw_response="The answer is 30.",
        description="Bad format (no tags → instant penalty)",
    )

    # ── Summary ─────────────────────────────────────────────────
    print_header("WHAT HAPPENS DURING TRAINING")
    print(f"""
   The RL trainer (TRL with GRPO algorithm) runs thousands of these
   episodes. Over time, the model learns:

   {GREEN}✓{RESET} Use the <budget>N</budget><answer>text</answer> format
   {GREEN}✓{RESET} Set budget proportional to question complexity
   {GREEN}✓{RESET} Answer correctly but concisely
   {GREEN}✓{RESET} Avoid repetition and filler text
   {GREEN}✓{RESET} Predict its own token usage accurately

   The curriculum system ensures it masters easy questions first,
   then gradually faces harder ones as its average reward improves.
""")
