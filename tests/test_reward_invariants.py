"""In-process tests for the reward function and anti-hacking cliffs.

These tests instantiate ``TokenEfficiencyEnvironment`` directly (no HTTP/WS
layer). They run with the offline ``KeywordJudge`` (set by ``conftest.py``)
so they pass on CI without an HF_TOKEN and complete in well under a second.

Promoted from ``_phase3_smoketest.py`` (the original Phase 3 sanity script).

Design choices being asserted:
  * The 6-component reward formula is bounded in [-1.0, 1.0] for non-cliff
    paths and produces sensible scores for hand-picked correct/wrong answers.
  * Anti-hacking cliffs short-circuit to fixed reward values with the right
    error tag and an empty ``reward_components`` dict.
  * The parrot guard does NOT fire on legitimate verbose-but-correct answers
    that happen to share words with the prompt.
"""

from __future__ import annotations

import pytest

from token_efficiency_env.judge import matches_keyword
from token_efficiency_env.models import TokenEfficiencyAction
from token_efficiency_env.scorer import score_answer
from token_efficiency_env.server import token_efficiency_env_environment as ee
from token_efficiency_env.server.token_efficiency_env_environment import (
    TokenEfficiencyEnvironment,
)


def _fresh_env(task_idx: int) -> TokenEfficiencyEnvironment:
    """Build a env locked to a specific task without going through reset()."""
    env = TokenEfficiencyEnvironment()
    env.current_task = ee.PROMPT_BANK[task_idx]
    env.episode_count = 1
    return env


# ─── Indices into the canonical PROMPT_BANK in prompts.py ───────────────
# Looked up by prompt text so the tests can't silently drift if the bank
# is reordered (as happened when the §7 numeric-alias migration landed).
TASK_BOIL = next(
    i for i, p in enumerate(ee.PROMPT_BANK)
    if p["prompt"] == "What is the boiling point of water in Celsius?"
)
TASK_PARIS = next(
    i for i, p in enumerate(ee.PROMPT_BANK)
    if p["prompt"] == "What is the capital of France?"
)


# ─── Happy paths ────────────────────────────────────────────────────────
def test_perfect_concise_correct_answer_is_high_reward():
    env = _fresh_env(TASK_PARIS)
    obs = env.step(TokenEfficiencyAction(
        raw_response="<budget>20</budget><answer>Paris.</answer>"
    ))
    assert obs.error == ""
    assert obs.reward is not None
    assert obs.reward >= 0.7, f"perfect answer should score high, got {obs.reward}"
    assert obs.reward_components, "components dict should be non-empty on a clean step"
    # The 6 weighted components must always be present. The scorer also stuffs a
    # debug `final_reward` key into details (mirrors the top-level `reward`),
    # so we assert subset-not-equal to stay forward-compatible.
    expected = {
        "correctness", "efficiency", "self_assessment",
        "redundancy", "keyword_verification", "format_quality",
    }
    assert expected.issubset(obs.reward_components), (
        f"missing components: {expected - set(obs.reward_components)}"
    )


def test_wrong_answer_is_low_reward():
    env = _fresh_env(TASK_PARIS)
    obs = env.step(TokenEfficiencyAction(
        raw_response="<budget>20</budget><answer>Tokyo.</answer>"
    ))
    assert obs.error == ""
    assert obs.reward is not None
    assert obs.reward < 0.5, f"wrong answer should score low, got {obs.reward}"
    assert obs.reward_components["correctness"] < 0.5


def test_verbose_but_correct_does_not_trip_parrot_guard():
    """Repeating words from the question is fine as long as the answer
    isn't *literally* the question."""
    env = _fresh_env(TASK_BOIL)
    obs = env.step(TokenEfficiencyAction(
        raw_response="<budget>40</budget><answer>"
                     "The boiling point of water is 100 degrees Celsius.</answer>"
    ))
    assert obs.error == "", f"parrot guard misfired: {obs.error}"
    assert obs.reward is not None and obs.reward > 0.0


def test_reward_is_clamped_to_unit_interval():
    """All non-cliff rewards must end up in [-1.0, 1.0]."""
    env = _fresh_env(TASK_PARIS)
    obs = env.step(TokenEfficiencyAction(
        raw_response="<budget>5</budget><answer>Paris.</answer>"
    ))
    assert obs.reward is not None
    assert -1.0 <= obs.reward <= 1.0


# ─── Anti-hacking cliffs (parametrised) ─────────────────────────────────
@pytest.mark.parametrize("raw_response, expected_error, expected_reward", [
    pytest.param(
        "<answer>Paris.</answer>",
        "bad_format", -1.0,
        id="missing-budget-tag",
    ),
    pytest.param(
        "Paris is the capital.",
        "bad_format", -1.0,
        id="no-tags-at-all",
    ),
    pytest.param(
        "<budget>5</budget><answer>.</answer>",
        "empty", -1.0,
        id="single-character-answer",
    ),
    pytest.param(
        "<budget>5</budget><answer> </answer>",
        "empty", -1.0,
        id="whitespace-only-answer",
    ),
])
def test_cliffs_short_circuit_with_correct_error_tag(
    raw_response: str, expected_error: str, expected_reward: float
):
    env = _fresh_env(TASK_PARIS)
    obs = env.step(TokenEfficiencyAction(raw_response=raw_response))
    assert obs.error == expected_error
    assert obs.reward == expected_reward
    assert obs.reward_components == {}, \
        "components dict should be empty when a cliff fires"


def test_parrot_cliff_fires_when_answer_is_the_question():
    env = _fresh_env(TASK_PARIS)
    parroted = env.current_task["prompt"]
    obs = env.step(TokenEfficiencyAction(
        raw_response=f"<budget>20</budget><answer>{parroted}</answer>"
    ))
    assert obs.error == "parrot"
    assert obs.reward == -0.5
    assert obs.reward_components == {}


def test_overshoot_heavily_penalises_self_assessment():
    """Predicting budget=5 then writing 30+ tokens should crush self_assessment
    (and likely efficiency as well) so total reward stays modest even when
    correctness is high.

    Note: there is no `overshoot` cliff in the env. Going over the predicted
    budget is paid through the `self_assessment` and `efficiency` components.
    The hard cliff is `too_long` at 500 tokens — well above what this test produces.
    """
    env = _fresh_env(TASK_PARIS)
    verbose = (
        "The capital city of the country known as France is the beautiful "
        "city of Paris located in the Ile-de-France region in the north of "
        "the country."
    )
    obs = env.step(TokenEfficiencyAction(
        raw_response=f"<budget>5</budget><answer>{verbose}</answer>"
    ))
    assert obs.error == "", f"unexpected cliff: {obs.error!r}"
    assert obs.reward_components["self_assessment"] < 0.5, (
        "huge overshoot must heavily penalise self_assessment, "
        f"got {obs.reward_components}"
    )


# ─── Keyword matcher (regression for the prompts.py-author intent) ──────
@pytest.mark.parametrize(
    "answer,keyword,should_match",
    [
        # Pure-numeric keywords use STRICT word-boundary matching to avoid
        # false positives like "100" → "1000" or "30" → "300".
        ("the answer is 100", "100", True),
        ("the answer is 1000", "100", False),
        ("about 30 percent", "30", True),
        ("about 300 percent", "30", False),
        # Word stems use PREFIX matching, so prompts.py author conventions
        # like "antibod" → "antibodies", "purchas" → "purchasing",
        # "intelligen" → "intelligence" all do the right thing.
        ("vaccines produce antibodies", "antibod", True),
        ("rising prices reduce purchasing power", "purchas", True),
        ("artificial intelligence is...", "intelligen", True),
        ("plants need water", "plant", True),
        ("plant biology", "plant", True),
        # But prefix is at WORD START — so "light" should NOT secretly catch
        # "sunlight" (this is why the photosynthesis prompt now uses
        # "sunlight" directly as a keyword instead of "light").
        ("powered by sunlight", "light", False),
        ("the sunlight reflects", "sunlight", True),
        # Whole-word case: matches the keyword exactly.
        ("the immune system", "immune", True),
        ("written by Shakespeare", "shakespeare", True),
    ],
)
def test_matches_keyword_prefix_and_numeric_policies(answer, keyword, should_match):
    assert matches_keyword(answer.lower(), keyword) is should_match


# ─── §2 Bigram redundancy ramp (VULNERABILITY_FIX_PLAN.md §2.4) ─────────
# The pre-v0.3.0 scorer applied the bigram penalty uniformly, so a 2-word
# answer like "Paris, France" lost points for having exactly one repeated
# bigram (trivially 100% of its bigrams). The ramp reintroduces the bigram
# signal smoothly between BIGRAM_MIN_WORDS=8 and BIGRAM_FULL_WORDS=16.
class TestBigramRamp:
    """Parametrised over the ramp boundaries so a future tuning can't
    silently reintroduce the short-answer penalty."""

    @staticmethod
    def _redundancy(response: str) -> float:
        # Minimal inputs; correctness comes from KeywordJudge, bounded [0,1].
        result = score_answer(
            prompt="",
            response=response,
            allocated_budget=10,
            tokens_used=10,
            complexity="medium",
            expected_keywords=[],
        )
        return float(result["details"]["redundancy"])

    def test_two_word_concise_answer_is_not_penalised(self):
        """'Paris, France' used to score 0.70; under the ramp it scores 1.0
        because the bigram term is fully gated out below 8 words."""
        assert self._redundancy("Paris, France") >= 0.99

    def test_single_word_answer_unchanged(self):
        """One-word path never looked at bigrams; still 1.0."""
        assert self._redundancy("Yes.") >= 0.99

    @pytest.mark.parametrize("word_count", [2, 5, 7])
    def test_below_ramp_floor_ignores_bigrams(self, word_count: int):
        """Anything below BIGRAM_MIN_WORDS (=8) should score purely on
        unique_ratio — i.e. a non-repeating short answer must be 1.0."""
        response = " ".join(f"word{i}" for i in range(word_count))
        assert self._redundancy(response) >= 0.99, (
            f"{word_count}-word non-repeating answer must not be penalised"
        )

    def test_full_ramp_penalises_obvious_padding(self):
        """16 identical words → ramp is fully active, penalty bites hard."""
        # 16 copies of 'the' → all 15 bigrams are ('the the'), raw_penalty=1.0
        response = " ".join(["the"] * 16)
        # unique_ratio = 1/16 ≈ 0.0625, full bigram penalty (ramp=1.0) → 0.0
        # redundancy = 0.0625 * 0.7 + (1 - 1.0) * 0.3 ≈ 0.04
        assert self._redundancy(response) < 0.1, (
            "fully-ramped repetition must crater redundancy"
        )

    def test_ramp_is_monotone_nondecreasing_in_penalty(self):
        """The penalty only grows as word count grows past 8. In other
        words, for the same 'worst-case' bigram pattern, 8 words must score
        at least as high as 12 words, which must score at least as high as
        16 words."""
        def rep(n: int) -> str:
            return " ".join(["the"] * n)
        r8 = self._redundancy(rep(8))   # ramp = 0.0
        r12 = self._redundancy(rep(12)) # ramp ≈ 0.5
        r16 = self._redundancy(rep(16)) # ramp = 1.0
        assert r8 >= r12 >= r16, (
            f"bigram ramp must be non-decreasing in word count, "
            f"got r8={r8:.3f}, r12={r12:.3f}, r16={r16:.3f}"
        )


# ─── §7 Numeric keyword aliases (list-of-lists schema) ──────────────────
# Before v0.3.0, prompts like "What is 15% of 200?" only accepted the digit
# form '30', so a natural "thirty" answer scored 0/1 on keyword_verification.
# The new schema lets a single keyword entry be a LIST of equivalent forms;
# matching any form counts it once.
class TestNumericKeywordAliases:
    """Verifies the list-of-lists keyword schema from §7."""

    @staticmethod
    def _keyword_score(response: str, keywords) -> float:
        result = score_answer(
            prompt="",
            response=response,
            allocated_budget=10,
            tokens_used=10,
            complexity="easy",
            expected_keywords=keywords,
        )
        return float(result["details"]["keyword_verification"])

    def test_digit_form_matches_numeric_alias_group(self):
        assert self._keyword_score(
            "The answer is 30 percent.", [["30", "thirty"]]
        ) == 1.0

    def test_word_form_matches_numeric_alias_group(self):
        """The previously-broken case: "thirty" alone must count the '30'
        keyword as satisfied."""
        assert self._keyword_score(
            "The answer is thirty percent.", [["30", "thirty"]]
        ) == 1.0

    def test_neither_form_present_scores_zero(self):
        assert self._keyword_score(
            "The answer is forty percent.", [["30", "thirty"]]
        ) == 0.0

    def test_single_form_match_counts_as_one_keyword_not_two(self):
        """Critical invariant: each alias group counts as ONE keyword in
        the denominator, otherwise a correct single-form answer would
        score 0.5 instead of 1.0."""
        # 1 alias group → denominator = 1, digit match → 1/1 = 1.0
        assert self._keyword_score("About 30%", [["30", "thirty"]]) == 1.0

    def test_mixed_schema_str_and_list_both_work(self):
        """Backwards-compatibility: a flat string keyword coexists with
        a list-of-forms in the same prompt."""
        score = self._keyword_score(
            "The value is thirty units.",
            ["unit", ["30", "thirty"]],  # both should match
        )
        assert score == 1.0

    def test_numeric_strict_still_applies_inside_alias_group(self):
        """The strict-numeric rule ("30" must NOT match "300") from the
        Phase-6.5 fix must still hold inside an alias group."""
        assert self._keyword_score(
            "Approximately 300 units",
            [["30", "thirty"]],
        ) == 0.0, (
            "strict numeric matching must still prevent 30 → 300 false positives"
        )


def test_photosynthesis_keywords_match_a_natural_answer():
    """The exact answer the user typed in the dashboard. Regression for
    the keyword bug that scored 0/3 because 'light' wouldn't match
    'sunlight' and 'energy' wasn't in the answer at all."""
    photo_idx = next(
        i for i, p in enumerate(ee.PROMPT_BANK) if p["prompt"] == "What is photosynthesis?"
    )
    env = _fresh_env(photo_idx)
    answer = (
        "Photosynthesis is the process by which green plants, algae, and "
        "some bacteria use sunlight, water, and carbon dioxide to create "
        "their own food (sugar) and release oxygen."
    )
    obs = env.step(TokenEfficiencyAction(
        raw_response=f"<budget>40</budget><answer>{answer}</answer>"
    ))
    assert obs.error == ""
    assert obs.reward_components["keyword_verification"] >= 0.99, (
        f"keyword score should be ~1.0 for this answer, got "
        f"{obs.reward_components['keyword_verification']}"
    )
