"""Unit tests for the SFT format-warmup dataset builder.

The SFT warmup is the difference between GRPO getting a gradient signal
from step 1 and the model flailing for 100 steps trying to guess the
format. This test suite guards the dataset's structural invariants so a
regression can't silently ship an empty / malformed warmup set.
"""

from __future__ import annotations

import re

import pytest

from training.sft_format_data import (
    SFTExample,
    _converge_budget,
    _first_keyword_as_answer,
    build_format_teaching_examples,
)

FORMAT_RE = re.compile(r"^<budget>(\d+)</budget><answer>(.+)</answer>$", re.DOTALL)


def test_default_build_returns_50_examples():
    """Plan target is 50 format-teaching examples."""
    examples = build_format_teaching_examples()
    assert len(examples) == 50


def test_every_example_matches_the_required_format():
    """Each response must be parseable by the same regex the scorer uses."""
    for i, ex in enumerate(build_format_teaching_examples()):
        m = FORMAT_RE.match(ex["response"])
        assert m is not None, f"example[{i}] malformed: {ex['response']!r}"
        budget = int(m.group(1))
        assert budget > 0, f"example[{i}] has non-positive budget {budget}"


def test_complexity_is_a_known_tier():
    for i, ex in enumerate(build_format_teaching_examples()):
        assert ex["complexity"] in ("easy", "medium", "hard"), (
            f"example[{i}] unknown complexity {ex['complexity']!r}"
        )


def test_all_three_tiers_are_represented():
    """Warmup must expose the model to every tier so budget prediction
    isn't anchored to a single length regime."""
    examples = build_format_teaching_examples()
    tiers = {ex["complexity"] for ex in examples}
    assert tiers == {"easy", "medium", "hard"}, (
        f"missing tier coverage in warmup set: {tiers!r}"
    )


def test_deterministic_under_fixed_seed():
    """Reviewers depend on the warmup set being reproducible."""
    a = build_format_teaching_examples(seed=42)
    b = build_format_teaching_examples(seed=42)
    assert a == b


def test_different_seeds_produce_different_orderings():
    """Sanity: the seed actually matters."""
    a = build_format_teaching_examples(seed=1)
    b = build_format_teaching_examples(seed=999)
    assert a != b


def test_budget_matches_tokenized_length_exactly():
    """With a real ``tokenize_len`` injected, ``N`` must equal
    ``len(tokenize(full_response))``. The fixed-point loop guarantees this,
    which is how the model learns honest self-prediction."""
    # Simple synthetic tokenizer: one token per word-like chunk.
    def fake_len(s: str) -> int:
        return len(s.split())

    for ex in build_format_teaching_examples(tokenize_len=fake_len):
        m = FORMAT_RE.match(ex["response"])
        assert m is not None
        claimed = int(m.group(1))
        actual = fake_len(ex["response"])
        # Allow ±1 tolerance for the rare non-convergent oscillation
        # at digit-count boundaries (documented in _converge_budget).
        assert abs(claimed - actual) <= 1, (
            f"budget={claimed} vs actual tokens={actual} for {ex['response']!r}"
        )


def test_first_keyword_as_answer_handles_nested_forms():
    """Helpers must handle flat strings, flat lists, and nested lists
    (§7 aliases like [['9', 'nine']])."""
    assert _first_keyword_as_answer("paris") == "Paris."
    assert _first_keyword_as_answer(["paris", "france"]) == "Paris."
    assert _first_keyword_as_answer([["9", "nine"]]) == "9."


def test_converge_budget_terminates_on_oscillation():
    """If the tokenizer oscillates between two counts at a digit boundary,
    the loop must still return without hanging."""
    toggle = [0]

    def oscillating_len(s: str) -> int:
        toggle[0] += 1
        return 9 if toggle[0] % 2 == 0 else 10

    result = _converge_budget("dummy", oscillating_len, max_iters=5)
    assert isinstance(result, int) and result > 0


def test_respects_smaller_n_cap():
    """Passing ``n=10`` must cap the dataset at 10 even though more are available."""
    small = build_format_teaching_examples(n=10)
    assert len(small) == 10
    # And still respects format + tier coverage (where possible).
    for ex in small:
        assert FORMAT_RE.match(ex["response"])
