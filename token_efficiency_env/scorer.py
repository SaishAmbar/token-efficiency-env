import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

COMPLEXITY_BUDGET = {
    "easy": 0.2,
    "medium": 0.5,
    "hard": 0.9
}

def score_answer(prompt, response, allocated_budget, tokens_used):

    # --- Part 1: correctness via Claude API (50% of reward) ---
    try:
        judge = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": f"""Question: {prompt}
Answer: {response}
Rate this answer's correctness from 0.0 to 1.0.
Reply with only a number. Example: 0.7"""
            }]
        )
        correctness = float(judge.content[0].text.strip())
        correctness = max(0.0, min(1.0, correctness))
    except:
        correctness = 0.0

    # --- Part 2: efficiency score (30% of reward) ---
    if tokens_used <= allocated_budget:
        # used less than budget — great
        efficiency = 1.0 - (tokens_used / max(allocated_budget, 1)) * 0.3
    else:
        # went over own budget — penalise
        efficiency = -0.5

    # --- Part 3: budget reasonableness (20% of reward) ---
    # model should allocate more budget for harder questions
    # we use prompt length as a proxy for complexity
    prompt_length = len(prompt.split())
    expected_ratio = min(prompt_length / 15, 1.0)
    allocated_ratio = min(allocated_budget / 200, 1.0)
    budget_reasonableness = 1.0 - abs(allocated_ratio - expected_ratio)

    # --- Final reward ---
    reward = (
        0.5 * correctness +
        0.3 * efficiency +
        0.2 * budget_reasonableness
    )

    return round(float(reward), 4)
