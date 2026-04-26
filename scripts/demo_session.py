"""Tier-2 demo: drive TokenEfficiencyEnv from Python with four canonical answers.

Why this script exists:

* Familiarisation. Running this once gives you a printable, copy-pasteable
  transcript of what the env looks like from a trainer's point of view —
  no browser, no typing, nothing to forget.
* Demo video B-roll. Running this on-camera shows "good answer → high
  reward", "wasteful answer → efficiency tanks", "bad format → cliff",
  "parrot → cliff" in ~15 seconds.
* Regression reassurance. If the env ever starts returning weird rewards,
  re-running this is the fastest way to see it before opening a
  debugger.

Usage::

    # Start the env server first (separate terminal):
    #   $env:ENABLE_WEB_INTERFACE = 'true'
    #   $env:JUDGE_BACKEND        = 'keyword'
    #   python -m uvicorn token_efficiency_env.server.app:app --port 8000
    #
    # Then:
    python scripts/demo_session.py
    python scripts/demo_session.py --base-url http://localhost:8000
    python scripts/demo_session.py --base-url https://<you>-tokeneff.hf.space
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from token_efficiency_env.client import TokenEfficiencyEnv  # noqa: E402
from token_efficiency_env.models import TokenEfficiencyAction  # noqa: E402


@dataclass
class Trial:
    """One question/answer trial we want to exercise."""

    label: str
    raw_response: str
    expectation: str


# Lookup table: for each of the 24 STARTER_PROMPTS, a tight, correct
# answer. We match by prompt-substring so we're robust to whitespace /
# trailing punctuation drift. Used by the "smart" 5th trial below —
# whichever prompt .reset() hands us, we try to answer it correctly so
# the user sees correctness=1.0 at least once in the session.
CANNED_ANSWERS: List[tuple] = [
    # (substring the prompt must contain, tight good answer text)
    ("15% of 200",            "30"),
    ("capital of France",     "Paris"),
    ("CPU stand",             "Central Processing Unit"),
    ("2 to the power of 8",   "256"),
    ("red and blue",          "Purple"),
    ("sides does a hexagon",  "Six"),
    ("boiling point of water","100"),
    ("Romeo and Juliet",      "Shakespeare"),
    # Medium
    ("gravity is",            "Gravity is the force that attracts objects with mass toward one another."),
    ("RAM and ROM",           "RAM is volatile read-write memory; ROM is non-volatile read-only memory."),
    ("vaccine work",          "A vaccine teaches the immune system to produce antibodies against a pathogen."),
    ("causes seasons",        "Earth's axis tilt changes how directly sunlight hits each hemisphere as it orbits the sun."),
    ("inflation means",       "Inflation is a general rise in prices that reduces the purchasing power of money."),
    ("speed and velocity",    "Speed is a scalar (magnitude only); velocity is a vector (magnitude and direction)."),
    ("internet work",         "Devices exchange data through interconnected networks of routers and servers using shared protocols."),
    ("photosynthesis",        "Plants use sunlight to convert water and carbon dioxide into glucose, releasing oxygen."),
    # Hard
    ("transformers work",     "Transformers use self-attention across tokens in parallel layers to model long-range dependencies."),
    ("climate change",        "Greenhouse gases like carbon dioxide trap heat, raising global temperatures and disrupting weather patterns."),
    ("supervised and unsupervised", "Supervised learning trains on labeled examples; unsupervised learning finds structure (e.g. clusters) in unlabeled data."),
    ("immune system fight",   "Immune cells recognise pathogen antigens and produce antibodies that neutralise the virus."),
    ("quantum entanglement",  "Two particles share a joint state so measuring one instantly determines the other's state."),
    ("Turing Test",           "It proposes that a machine showing human-indistinguishable conversational intelligence demonstrates intelligence."),
    ("neural networks learn", "They adjust layer weights via gradient descent using backpropagation to reduce prediction error."),
    ("theory of relativity",  "Einstein showed space and time are linked; motion and gravity distort both relative to the observer."),
]


def _canned_answer(prompt: str) -> Optional[str]:
    """Find a tight correct answer for a starter-bank prompt, or None."""
    for needle, answer in CANNED_ANSWERS:
        if needle.lower() in prompt.lower():
            return answer
    return None


# Four canonical trials chosen to exercise the full reward surface.
# They use fixed `raw_response` strings that make sense for MOST easy
# prompts ("capital", "color", etc.) — if the env happens to draw a
# prompt where these are nonsensical, that's still fine: the important
# thing is seeing the reward *shape*, not the absolute numbers.
TRIALS: List[Trial] = [
    Trial(
        label="GOOD — tight, honest budget",
        raw_response="<budget>5</budget><answer>Paris.</answer>",
        expectation="reward ~0.6–0.9 (all 6 components healthy)",
    ),
    Trial(
        label="WASTEFUL — correct but over-explains",
        raw_response=(
            "<budget>200</budget><answer>Paris is the capital city of France, "
            "which is a country in Western Europe bordered by Belgium, Germany, "
            "Switzerland, Italy, Spain, and the Atlantic Ocean. The city has "
            "been the French capital since the 10th century.</answer>"
        ),
        expectation="reward ~0.2–0.5 (efficiency + self_assessment tanks)",
    ),
    Trial(
        label="BAD FORMAT — missing tags",
        raw_response="Paris",
        expectation="reward = -1.0, error='bad_format' (cliff override)",
    ),
    Trial(
        label="REPETITION — single word spammed",
        raw_response="<budget>5</budget><answer>Paris Paris Paris Paris Paris</answer>",
        expectation="reward = -0.5, error='repetition' (cliff override)",
    ),
]


def _fmt_components(c: dict) -> str:
    """Short one-line rendering of the 6 reward components."""
    order = [
        "correctness", "efficiency", "self_assessment",
        "redundancy", "keyword_verification", "format_quality",
    ]
    return "  ".join(f"{k[:4]}={c.get(k, 0.0):+.2f}" for k in order)


def _run_smart_trial(env, idx: int, total: int) -> None:
    """Trial that reads the prompt from reset() and answers it correctly.

    This is what a *good* policy looks like: tight budget, correct answer
    matching the expected keywords, no repetition, proper format. Should
    yield reward in the 0.7–0.95 range on easy tier."""
    print(f"\n┌── Trial {idx + 1}/{total}: SMART — read prompt, answer correctly")
    reset_result = env.reset()
    o0 = getattr(reset_result, "observation", reset_result)
    q = getattr(o0, "prompt", "") or "(no prompt)"
    complexity = getattr(o0, "complexity", "?")
    print(f"│  Q           : {q}")
    print(f"│  tier        : {complexity}")

    answer = _canned_answer(q)
    if answer is None:
        print("│  (no canned answer for this prompt — falling back to Paris)")
        answer = "Paris"

    # Pick a budget that matches the complexity tier: tight on easy,
    # roomier on medium/hard. The env's ideal tokens are 15/60/130.
    budget_by_tier = {"easy": 10, "medium": 50, "hard": 110}
    budget = budget_by_tier.get(complexity, 20)
    raw = f"<budget>{budget}</budget><answer>{answer}</answer>"
    print(f"│  sending     : {raw[:80]}{'…' if len(raw) > 80 else ''}")

    result = env.step(TokenEfficiencyAction(raw_response=raw))
    o = result.observation
    components = getattr(o, "reward_components", {}) or {}
    cliff_err = getattr(o, "error", "") or "—"

    print(f"│")
    print(f"│  reward      : {result.reward:+.3f}")
    print(f"│  tokens_used : {getattr(o, 'tokens_used', '?')}   "
          f"budget pred: {getattr(o, 'allocated_budget', '?')}")
    print(f"│  cliff/error : {cliff_err}")
    if components:
        print(f"│  components  : {_fmt_components(components)}")
    print(f"└──")


def _run_trial(env, trial: Trial, idx: int, total: int) -> None:
    """Run one reset→step cycle and pretty-print the result."""
    print(f"\n┌── Trial {idx + 1}/{total}: {trial.label}")

    reset_result = env.reset()
    # ``reset()`` returns a StepResult-like container; the actual
    # observation lives under ``.observation`` (same shape as ``step()``).
    o0 = getattr(reset_result, "observation", reset_result)
    q = getattr(o0, "prompt", "") or "(no prompt)"
    limit = getattr(o0, "episode_token_limit", None)
    print(f"│  Q           : {q}")
    print(f"│  budget cap  : {limit}")
    print(f"│  expect      : {trial.expectation}")
    print(f"│  sending     : {trial.raw_response[:80]}"
          f"{'…' if len(trial.raw_response) > 80 else ''}")

    result = env.step(TokenEfficiencyAction(raw_response=trial.raw_response))
    o = result.observation

    tokens_used = getattr(o, "tokens_used", "?")
    budget_pred = getattr(o, "allocated_budget", "?")
    cliff_err = getattr(o, "error", "") or "—"
    complexity = getattr(o, "complexity", "?")
    components = getattr(o, "reward_components", {}) or {}

    print(f"│")
    print(f"│  reward      : {result.reward:+.3f}")
    print(f"│  tokens_used : {tokens_used}   budget pred: {budget_pred}"
          f"   tier: {complexity}")
    print(f"│  cliff/error : {cliff_err}")
    if components:
        print(f"│  components  : {_fmt_components(components)}")
    print(f"└──")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive TokenEfficiencyEnv through 4 canonical trials."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Env server URL (default: local uvicorn on port 8000).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print("=" * 70)
    print(" TokenEfficiencyEnv — Tier-2 Python client demo")
    print("=" * 70)
    print(f" base_url   : {args.base_url}")
    print(f" trials     : {len(TRIALS) + 1}  (4 fixed + 1 smart)")

    try:
        # OpenEnv clients are async by default; .sync() wraps them for
        # plain `with ... as env`.
        total = len(TRIALS) + 1  # +1 for the smart trial at the end
        with TokenEfficiencyEnv(base_url=args.base_url).sync() as env:
            for i, t in enumerate(TRIALS):
                _run_trial(env, t, i, total)
            _run_smart_trial(env, len(TRIALS), total)
    except ConnectionError as e:
        print(f"\n[ERROR] Could not reach the env server at {args.base_url}.")
        print("         Is `uvicorn token_efficiency_env.server.app:app` running?")
        print(f"         Details: {e}")
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"\n[ERROR] Unexpected failure: {type(e).__name__}: {e}")
        return 3

    print()
    print("─" * 70)
    print("How to read this:")
    print("  • corr=0 on Trials 1–2 is expected — we send 'Paris' to non-Paris")
    print("    questions, so the keyword judge correctly rates correctness=0.")
    print("  • Trial 1 shows FORMAT + EFFICIENCY working even when wrong.")
    print("  • Trial 2 shows SELF_ASSESSMENT staying positive (budget=200,")
    print("    actual=60 → honest upper-bound undershoot).")
    print("  • Trials 3–4 show the CLIFFS (bad_format, repetition) overriding")
    print("    the formula and returning fixed negative rewards.")
    print()
    print("To see correctness=1, either (a) set JUDGE_BACKEND=huggingface +")
    print("HF_TOKEN and answer sensibly, or (b) hand-craft an answer to match")
    print("whatever prompt .reset() happens to hand you.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
