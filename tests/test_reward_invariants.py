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

from token_efficiency_env.models import TokenEfficiencyAction
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
# 0 = "What is the boiling point of water in Celsius?" (easy)
# 1 = "What is the capital of France? One word." (easy)
TASK_BOIL = 0
TASK_PARIS = 1


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
