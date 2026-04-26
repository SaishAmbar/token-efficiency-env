"""Regression tests for ``judge._parse_score`` out-of-range handling.

The old implementation parsed any number it could find and left clamping
to ``score_correctness``, which silently mapped ``"Score: 7"`` → ``1.0``.
That created a reward-hacking surface: if a trainee can bias the judge
toward X/10 phrasing, it collects free perfect-correctness rewards.

The hardened parser now treats out-of-range numbers as a parse failure
(``None``) so ``score_correctness`` falls through to the deterministic
keyword scorer for that call. These tests pin that behaviour.
"""

from __future__ import annotations

from token_efficiency_env.judge import _parse_score


class TestInRangeScores:
    """In-range replies still parse correctly — no regression."""

    def test_parses_score_prefix_in_range(self) -> None:
        assert _parse_score("Score: 0.85") == 0.85

    def test_parses_zero(self) -> None:
        assert _parse_score("Score: 0.00") == 0.0

    def test_parses_one(self) -> None:
        assert _parse_score("Score: 1.00") == 1.0

    def test_parses_integer_zero(self) -> None:
        assert _parse_score("Score: 0") == 0.0

    def test_parses_integer_one(self) -> None:
        assert _parse_score("Score: 1") == 1.0

    def test_parses_trailing_explanation(self) -> None:
        # Judge sometimes adds a sentence after the number; the fallback
        # regex should still find the leading decimal.
        assert _parse_score("The answer partially covers the facts. 0.7") == 0.7


class TestOutOfRangeRejection:
    """Out-of-range numbers MUST fall through to keyword scoring."""

    def test_score_prefix_out_of_range_rejected(self) -> None:
        # This is the X/10 case that used to clamp to 1.0.
        assert _parse_score("Score: 7") is None

    def test_score_prefix_out_of_range_decimal_rejected(self) -> None:
        assert _parse_score("Score: 7.5") is None

    def test_score_prefix_negative_rejected(self) -> None:
        # Negative numbers — judge leaked a penalty — must not clamp to 0.
        assert _parse_score("Score: -0.5") is None

    def test_large_number_rejected(self) -> None:
        assert _parse_score("I give this a 100 out of 100.") is None

    def test_x_out_of_ten_rejected(self) -> None:
        # No "Score:" prefix, first-match regex would otherwise pick up
        # "8" but 8 is out of range → fall through.
        assert _parse_score("8 out of 10.") is None


class TestEmptyAndMalformedReplies:
    """Empty / pure-prose replies still return None."""

    def test_empty_string(self) -> None:
        assert _parse_score("") is None

    def test_whitespace_only(self) -> None:
        assert _parse_score("   \n\n") is None

    def test_prose_without_number(self) -> None:
        assert _parse_score("The answer is correct.") is None


class TestScoreCorrectnessFallback:
    """End-to-end: an out-of-range judge reply falls through to keywords.

    We can't easily mock the HF InferenceClient here, but we can exercise
    the same code path by directly poking the value of ``_parse_score``'s
    return. The important invariant is: if the judge reply parses to a
    number outside [0, 1], the correctness score we surface must be the
    keyword score, NOT 1.0.
    """

    def test_keyword_score_used_when_parse_fails(self) -> None:
        from token_efficiency_env.judge import _keyword_score

        # Answer misses every keyword → keyword score is 0.0.
        assert _keyword_score("unrelated answer", ["france", "paris"]) == 0.0
        # Answer matches one of two → 0.5.
        assert _keyword_score("paris is nice", ["france", "paris"]) == 0.5
