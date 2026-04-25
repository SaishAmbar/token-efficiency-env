"""Deterministic train/holdout split of ``PROMPT_BANK``.

We carve off 6 questions for evaluation (2 easy + 2 medium + 2 hard) and
train on the remaining 18. The split is hard-coded by index rather than
random-seeded so:

    * The held-out set never silently rotates if ``PROMPT_BANK`` is
      reordered or extended (a random split would, breaking all comparisons
      across runs).
    * Reviewers can read the indices and immediately know what the model
      was *not* trained on.

If ``PROMPT_BANK`` ever shrinks or the chosen indices stop matching their
intended complexity tier, ``_assert_split_invariants`` (called at import
time) will raise a clear error rather than letting the trainer leak the
held-out set into training.
"""

from __future__ import annotations

from typing import Dict, List

from token_efficiency_env.prompts import PROMPT_BANK

# ─── The split ─────────────────────────────────────────────────────────
# Layout in prompts.py (24 questions, 8 per tier):
#   indices  0-7  : easy
#   indices  8-15 : medium
#   indices 16-23 : hard
#
# We pull the *last 2* of each tier into the held-out set. That keeps the
# first 6 of each tier — the "canonical examples" — in training, where
# they're most useful for shaping curriculum behaviour.
HOLDOUT_INDICES: List[int] = [
    6, 7,    # easy   : "boiling point of water" + "Romeo and Juliet"
    14, 15,  # medium : "how does the internet work" + "what is photosynthesis"
    22, 23,  # hard   : "how do neural networks learn" + "theory of relativity"
]
TRAIN_INDICES: List[int] = [i for i in range(len(PROMPT_BANK)) if i not in HOLDOUT_INDICES]


def train_prompts() -> List[Dict]:
    """Return the training subset of PROMPT_BANK (deep-copied, won't mutate the bank)."""
    return [dict(PROMPT_BANK[i]) for i in TRAIN_INDICES]


def holdout_prompts() -> List[Dict]:
    """Return the held-out subset (deep-copied)."""
    return [dict(PROMPT_BANK[i]) for i in HOLDOUT_INDICES]


def describe_split() -> str:
    """One-liner per held-out prompt, for the notebook to print up top."""
    lines = ["Held-out prompts (NOT seen during training):"]
    for i in HOLDOUT_INDICES:
        p = PROMPT_BANK[i]
        lines.append(f"  [{i:2d}] {p['complexity']:<6} | {p['prompt']}")
    lines.append(f"\nTraining set: {len(TRAIN_INDICES)} prompts.")
    return "\n".join(lines)


# ─── Sanity checks (run at import time so split bugs blow up early) ────
def _assert_split_invariants() -> None:
    n = len(PROMPT_BANK)
    if n < 24:
        raise RuntimeError(
            f"PROMPT_BANK has {n} entries; the Phase 7 split assumes >=24. "
            "Either extend the bank or update HOLDOUT_INDICES."
        )

    # No overlap between train and holdout (paranoia, the comprehension
    # above already enforces this).
    if set(TRAIN_INDICES) & set(HOLDOUT_INDICES):
        raise RuntimeError("train/holdout overlap; split is broken.")

    # Each holdout index must point at a real prompt with the complexity
    # tier we expect (so the bank can't silently re-tier under us).
    expected_tiers = {
        6: "easy", 7: "easy",
        14: "medium", 15: "medium",
        22: "hard", 23: "hard",
    }
    for idx, expected in expected_tiers.items():
        actual = PROMPT_BANK[idx].get("complexity")
        if actual != expected:
            raise RuntimeError(
                f"PROMPT_BANK[{idx}] has complexity={actual!r}, "
                f"expected {expected!r}. Split file or prompts.py is out "
                "of sync — fix one of them before training."
            )


_assert_split_invariants()
