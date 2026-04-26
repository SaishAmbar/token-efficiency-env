"""Anti-cliff regression tests for the v0.3.0 vulnerability fixes.

Covers the specific exploits called out by
``test_check_docs/VULNERABILITY_FIX_PLAN.md`` §1 (hidden chain-of-thought)
and §5 (parrot → Jaccard). Every case here is one the prior v0.2.0 scorer
either silently let through or scored unfairly; each test encodes the
"correct" behaviour so a future refactor can't regress the guarantees.

Design choices:

* Each test instantiates ``TokenEfficiencyEnvironment`` directly and pins
  ``current_task`` to a known PROMPT_BANK entry. This matches the pattern
  used in ``test_reward_invariants.py`` and keeps the suite fast, offline,
  and free of HTTP/WebSocket flakiness.
* The keyword judge is wired via ``conftest.py`` (``JUDGE_BACKEND=keyword``)
  so correctness is deterministic.
* Where the plan offered a choice (§1 Layer A "count raw" vs Layer C
  "strict regex"), we encoded *Layer C* as the currently-shipped behaviour
  — leaking text outside the tags must produce ``bad_format`` at −1.0, not
  a scaled-down reward. If Layer C is ever loosened, the Layer A invariant
  (``tokens_used`` tracks raw length, not inner answer) is still asserted
  separately in ``test_raw_token_count_*``.
"""

from __future__ import annotations

import pytest

from token_efficiency_env.models import TokenEfficiencyAction
from token_efficiency_env.server import token_efficiency_env_environment as ee
from token_efficiency_env.server.token_efficiency_env_environment import (
    TokenEfficiencyEnvironment,
    _is_parrot,
)


def _fresh_env(task_idx: int) -> TokenEfficiencyEnvironment:
    env = TokenEfficiencyEnvironment()
    env.current_task = ee.PROMPT_BANK[task_idx]
    env.episode_count = 1
    return env


# Canonical indices — mirrors test_reward_invariants.py
TASK_BOIL = 0           # "What is 15% of 200?" (easy)  [NOTE: prompts.py order]
TASK_PARIS = 1          # "What is the capital of France?" (easy)
TASK_TRANSFORMERS = 16  # "Explain how transformers work..." (hard)


# ─── §1: Hidden chain-of-thought exploit ────────────────────────────────
class TestHiddenCoTExploit:
    """The prior parser used ``re.search``, so a model could emit hundreds of
    tokens of scratch reasoning before the ``<budget>`` tag and be scored only
    on the inner ``<answer>``. The v0.3.0 fix (§1 Layer C) uses a strict
    end-anchored regex; if any non-whitespace leaks outside the tags, the
    response must fail with ``bad_format``.
    """

    def test_leading_cot_before_budget_triggers_bad_format(self):
        """500-ish tokens of hidden reasoning before the tags → bad_format cliff."""
        env = _fresh_env(TASK_PARIS)
        hidden_cot = (
            "Hmm let me think step by step. The capital of France is famous "
            "for the Eiffel Tower and for being a global cultural hub. I "
            "should probably answer in one word to save tokens. Let me also "
            "consider whether the user wants the administrative capital or "
            "the cultural one — in this case they coincide. Final answer: "
        ) * 3  # ~300 words of scratch reasoning
        raw = f"{hidden_cot}<budget>3</budget><answer>Paris.</answer>"
        obs = env.step(TokenEfficiencyAction(raw_response=raw))

        assert obs.error == "bad_format", (
            f"hidden CoT must be rejected as bad_format, got error={obs.error!r}"
        )
        assert obs.reward == -1.0
        assert obs.reward_components == {}

    def test_trailing_text_after_answer_triggers_bad_format(self):
        """Text after the closing ``</answer>`` is the symmetric exploit."""
        env = _fresh_env(TASK_PARIS)
        raw = (
            "<budget>5</budget><answer>Paris.</answer>"
            " and here is why I think so: Paris has been the capital since "
            "the 10th century, when the Capetian dynasty..."
        )
        obs = env.step(TokenEfficiencyAction(raw_response=raw))

        assert obs.error == "bad_format"
        assert obs.reward == -1.0

    def test_text_between_tags_triggers_bad_format(self):
        """Anything between ``</budget>`` and ``<answer>`` must also be rejected."""
        env = _fresh_env(TASK_PARIS)
        raw = "<budget>5</budget> Reasoning: it is Paris. <answer>Paris.</answer>"
        obs = env.step(TokenEfficiencyAction(raw_response=raw))

        assert obs.error == "bad_format"
        assert obs.reward == -1.0

    def test_clean_shell_with_leading_whitespace_is_accepted(self):
        """The strict regex still allows a leading newline — explicit design
        choice: the format is about preventing hidden content, not punishing
        cosmetics."""
        env = _fresh_env(TASK_PARIS)
        raw = "\n  <budget>5</budget><answer>Paris.</answer>\n"
        obs = env.step(TokenEfficiencyAction(raw_response=raw))

        assert obs.error == "", (
            f"clean shell with surrounding whitespace must pass, got {obs.error!r}"
        )
        assert obs.reward is not None and obs.reward > 0.0


# ─── §1 Layer A: tokens_used counts the whole shell, not just the answer ───
class TestRawTokenCount:
    """If Layer C is ever softened, Layer A must still close the numeric side:
    ``tokens_used`` tracks the raw response so wrapper / scratch tokens can't
    be smuggled for free.
    """

    def test_tokens_used_includes_wrapper_tags(self):
        env = _fresh_env(TASK_PARIS)
        obs = env.step(TokenEfficiencyAction(
            raw_response="<budget>5</budget><answer>Paris.</answer>"
        ))
        assert obs.error == ""
        # The inner answer "Paris." alone is 2–3 Qwen tokens; the full shell is
        # at least ~10. Assert the wrapper is being paid for.
        assert obs.answer_token_count is not None
        assert obs.tokens_used > obs.answer_token_count, (
            "tokens_used must cover the full <budget>...</answer> shell, "
            f"got tokens_used={obs.tokens_used} vs answer_token_count="
            f"{obs.answer_token_count}"
        )

    def test_answer_token_count_diagnostic_is_populated(self):
        """§1 Layer B: inner-only count is exposed for dashboards."""
        env = _fresh_env(TASK_PARIS)
        obs = env.step(TokenEfficiencyAction(
            raw_response="<budget>5</budget><answer>Paris.</answer>"
        ))
        assert obs.answer_token_count is not None
        assert obs.answer_token_count >= 1


# ─── §5: Parrot guard uses Jaccard similarity ───────────────────────────
class TestParrotJaccard:
    """The old substring guard was both over- and under-triggered. The
    v0.3.0 Jaccard implementation:
      * skips short prompts (< 4 unique tokens) — can't be parroted reliably,
      * flags answers that share ≥ 85% of the prompt's tokens.
    """

    def test_literal_echo_is_flagged_as_parrot(self):
        """A verbatim echo of the prompt is still the canonical parrot case."""
        env = _fresh_env(TASK_TRANSFORMERS)
        parroted = env.current_task["prompt"]
        obs = env.step(TokenEfficiencyAction(
            raw_response=f"<budget>20</budget><answer>{parroted}</answer>"
        ))
        assert obs.error == "parrot"
        assert obs.reward == -0.5

    def test_natural_restated_answer_is_not_parrot(self):
        """Plan §5.4: restating the question words naturally must pass."""
        env = _fresh_env(TASK_TRANSFORMERS)
        natural = (
            "Transformers are a neural-net architecture that uses self-"
            "attention to weight tokens across layers, replacing recurrence "
            "with parallel position-aware context."
        )
        obs = env.step(TokenEfficiencyAction(
            raw_response=f"<budget>60</budget><answer>{natural}</answer>"
        ))
        assert obs.error == "", (
            f"natural paraphrase must not trip parrot guard, got {obs.error!r}"
        )

    def test_padded_paraphrase_over_threshold_is_parrot(self):
        """Plan §5.4: echoing most of the prompt plus a few new words =
        Jaccard ≥ 0.85 → cliff.
        """
        env = _fresh_env(TASK_TRANSFORMERS)
        # Prompt: "Explain how transformers work in machine learning."
        # Tokens after normalisation: {explain, how, transformers, work, in, machine, learning}
        # Answer below shares all 7 and adds only 1 new token → Jaccard ≈ 7/8 = 0.875
        padded = "Explain how transformers work in machine learning architecture"
        obs = env.step(TokenEfficiencyAction(
            raw_response=f"<budget>20</budget><answer>{padded}</answer>"
        ))
        assert obs.error == "parrot", (
            f"padded echo (Jaccard ~0.88) should be a parrot, got {obs.error!r}"
        )

    def test_short_prompt_is_exempt_from_parrot_guard(self):
        """Prompts with < 4 unique content words cannot be reliably parroted;
        the guard must not misfire on them even for near-verbatim answers."""
        # "What is 15% of 200?" normalises to {what, is, 15, of, 200} → 5 tokens,
        # just over the floor. Construct a pure short-prompt scenario by patching.
        env = _fresh_env(TASK_PARIS)
        env.current_task = {
            "prompt": "Capital of France?",
            "complexity": "easy",
            "expected_keywords": ["paris"],
        }
        obs = env.step(TokenEfficiencyAction(
            raw_response="<budget>5</budget><answer>The capital of France is Paris.</answer>"
        ))
        assert obs.error == "", (
            f"short-prompt answers must be exempt from the parrot guard, "
            f"got error={obs.error!r}"
        )


# ─── _is_parrot() unit tests (isolated from the env wiring) ─────────────
class TestIsParrotUnit:
    """Direct-call tests so the Jaccard thresholds stay pinned even if the
    env-level wiring changes.
    """

    @pytest.mark.parametrize("prompt, answer, should_fire", [
        pytest.param(
            "Explain how transformers work in machine learning.",
            "Explain how transformers work in machine learning.",
            True, id="literal-echo",
        ),
        pytest.param(
            "Explain how transformers work in machine learning.",
            "Transformers are a neural architecture using attention over tokens.",
            False, id="natural-paraphrase",
        ),
        pytest.param(
            "Capital of France?",
            "The capital of France is Paris.",
            False, id="short-prompt-exempt",
        ),
        pytest.param(
            "",
            "anything",
            False, id="empty-prompt",
        ),
        pytest.param(
            "What are the causes and effects of climate change?",
            "The causes and effects of climate change are what matter.",
            True, id="high-overlap-padding",
        ),
    ])
    def test_jaccard_thresholds(self, prompt: str, answer: str, should_fire: bool):
        result = _is_parrot(prompt, answer)
        if should_fire:
            assert result >= 0.85, f"expected parrot cliff, got jaccard={result}"
        else:
            assert result == 0.0, f"expected no cliff, got jaccard={result}"
