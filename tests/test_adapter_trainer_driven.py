"""Regression tests for the InProcessRewardAdapter curriculum-bypass fix.

The in-process adapter used to mutate ``env.current_task`` and bump
``env.episode_count`` without going through ``reset()`` — but ``step()``
still appended every reward to ``env.recent_rewards`` and the observation
still reported ``avg_reward_50`` derived from that deque. Since the
trainer (not the env) picks which prompts to score, ``avg_reward_50``
was effectively noise: the running mean of rewards on prompts the env's
sampler never saw.

The fix flips a new ``_trainer_driven_mode`` flag on each env owned by
the adapter, which gates the two ``recent_rewards.append(...)`` calls in
``step()`` / ``_fail()``. This file pins the invariant.
"""

from __future__ import annotations

import pytest

from token_efficiency_env.server.token_efficiency_env_environment import (
    TokenEfficiencyEnvironment,
)
from training.reward_adapter import InProcessRewardAdapter


@pytest.fixture
def adapter() -> InProcessRewardAdapter:
    return InProcessRewardAdapter(num_slots=1)


def test_envs_are_trainer_driven_mode(adapter: InProcessRewardAdapter) -> None:
    """Every slot env gets its curriculum side-effects turned off."""
    for env in adapter._envs:
        assert env._trainer_driven_mode is True


def test_step_does_not_mutate_recent_rewards(adapter: InProcessRewardAdapter) -> None:
    """After a batch of scored completions the env's deque stays empty."""
    env = adapter._envs[0]
    assert len(env.recent_rewards) == 0

    # Pick a real starter-bank prompt so the adapter finds its task.
    prompt = "What is the capital of France?"
    completion = "<budget>5</budget><answer>Paris.</answer>"

    rewards = adapter([prompt] * 3, [completion] * 3)

    assert len(rewards) == 3
    # If ANY reward leaked into the deque, the fix regressed.
    assert len(env.recent_rewards) == 0


def test_step_does_not_mutate_recent_rewards_on_cliff(
    adapter: InProcessRewardAdapter,
) -> None:
    """Bad-format cliff path MUST also respect trainer-driven mode.

    The fix had to gate TWO append sites: the success path in ``step()``
    AND the ``_fail`` helper used by every cliff. If the cliff path were
    missed, a batch of all-empty completions would still fill the deque.
    """
    env = adapter._envs[0]
    assert len(env.recent_rewards) == 0

    prompt = "What is the capital of France?"
    empty_completion = ""  # trips the bad_format cliff

    rewards = adapter([prompt] * 4, [empty_completion] * 4)

    # All four should be negative rewards from the cliff.
    assert all(r <= 0.0 for r in rewards)
    # And recent_rewards must still be empty.
    assert len(env.recent_rewards) == 0


def test_server_mode_still_updates_recent_rewards() -> None:
    """Default (server) env MUST still fill recent_rewards — curriculum
    for the WS / HF Space path is unchanged by this fix."""
    from token_efficiency_env.models import TokenEfficiencyAction
    from token_efficiency_env.server import token_efficiency_env_environment as ee

    env = TokenEfficiencyEnvironment()
    assert env._trainer_driven_mode is False

    env.current_task = ee.PROMPT_BANK[0]
    env.episode_count = 1

    env.step(TokenEfficiencyAction(raw_response="<budget>5</budget><answer>x</answer>"))
    assert len(env.recent_rewards) == 1
