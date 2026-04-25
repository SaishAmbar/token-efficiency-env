"""Smoke tests for the Phase 7 training scaffold.

These tests cover the offline pieces only:

    * ``training/config.py``           — dataclass defaults + cost estimate
    * ``training/prompts_split.py``    — split is deterministic, no overlap,
                                         tier invariants hold
    * ``training/reward_adapter.py``   — InProcessRewardAdapter scores a
                                         batch, logs every row, parrot
                                         cliff still fires through the
                                         adapter, unknown prompts don't
                                         crash the trainer

They run with the offline ``KeywordJudge`` backend (set by
``conftest.py``), so no HF_TOKEN is needed and the whole file completes in
under a second on CI.

The WebSocket adapter (``WSRewardAdapter``) is NOT exercised here:
spawning real uvicorn workers inside pytest is flaky and the dashboard +
``test_e2e_websocket.py`` already cover that wire path. The two adapters
share their public ``__call__(prompts, completions) -> list[float]``
contract, so a regression in one will surface in either suite.
"""

from __future__ import annotations

import pytest

from training.config import TrainingConfig
from training.prompts_split import (
    HOLDOUT_INDICES,
    TRAIN_INDICES,
    holdout_prompts,
    train_prompts,
)
from training.reward_adapter import (
    InProcessRewardAdapter,
    RewardLog,
    _completion_text,
    build_reward_func,
)


# ─── Config ───────────────────────────────────────────────────────────
def test_config_defaults_are_self_consistent():
    cfg = TrainingConfig()
    # GRPO requires per_device_train_batch_size to divide evenly into
    # num_generations (TRL invariant). Defaults must not violate it.
    assert cfg.num_generations % cfg.per_device_train_batch_size == 0
    # Cost estimate matches the formula in describe().
    assert cfg.estimate_judge_calls() == cfg.num_generations * cfg.max_steps
    # Banner doesn't crash and mentions the model so a reader can sanity check.
    text = cfg.describe()
    assert cfg.model_name in text
    assert "judge calls" in text


def test_config_cost_estimate_scales_linearly():
    cfg_small = TrainingConfig(num_generations=4, max_steps=10)
    cfg_big = TrainingConfig(num_generations=8, max_steps=100)
    assert cfg_big.estimate_judge_calls() == 20 * cfg_small.estimate_judge_calls()


# ─── Split ─────────────────────────────────────────────────────────────
def test_split_no_overlap():
    assert set(TRAIN_INDICES).isdisjoint(set(HOLDOUT_INDICES))


def test_split_holdout_size_is_six():
    assert len(HOLDOUT_INDICES) == 6
    assert len(holdout_prompts()) == 6


def test_split_holdout_covers_all_three_tiers():
    tiers = {p["complexity"] for p in holdout_prompts()}
    assert tiers == {"easy", "medium", "hard"}


def test_train_prompts_dont_share_objects_with_bank():
    """Mutating a returned dict must not corrupt PROMPT_BANK."""
    from token_efficiency_env.prompts import PROMPT_BANK

    sample = train_prompts()[0]
    original_prompt = PROMPT_BANK[TRAIN_INDICES[0]]["prompt"]
    sample["prompt"] = "MUTATED"
    assert PROMPT_BANK[TRAIN_INDICES[0]]["prompt"] == original_prompt


# ─── Reward adapter — happy path ───────────────────────────────────────
def test_in_process_adapter_scores_a_known_correct_answer():
    """Paris answer to the Paris question ⇒ reward should be high (>0.7)."""
    log = RewardLog()
    adapter = InProcessRewardAdapter(num_slots=2, log=log)

    prompt = "What is the capital of France?"
    completion = "<budget>3</budget><answer>Paris</answer>"
    rewards = adapter([prompt], [completion])

    assert len(rewards) == 1
    assert rewards[0] > 0.7, f"expected high reward for Paris, got {rewards[0]}"
    assert len(log) == 1
    row = log.to_records()[0]
    assert row["prompt"] == prompt
    assert "correctness" in row["components"]
    assert row["error"] == ""


def test_in_process_adapter_handles_a_full_grpo_batch():
    """8 completions for the same prompt — what GRPO actually sends."""
    log = RewardLog()
    adapter = InProcessRewardAdapter(num_slots=8, log=log)

    prompt = "What is the capital of France?"
    completions = [f"<budget>{i+2}</budget><answer>Paris.</answer>" for i in range(8)]

    rewards = adapter([prompt] * 8, completions)

    assert len(rewards) == 8
    # All Paris answers should land in the high reward zone.
    assert all(r > 0.5 for r in rewards), rewards
    assert len(log) == 8
    # Every slot should have been used at least once across the batch.
    used_slots = {row["slot"] for row in log.to_records()}
    assert used_slots, "no slot info logged"


# ─── Reward adapter — cliffs survive the adapter ──────────────────────
def test_parrot_cliff_still_fires_through_adapter():
    """Adapter must not paper over the env's anti-hacking cliffs."""
    log = RewardLog()
    adapter = InProcessRewardAdapter(num_slots=1, log=log)

    prompt = "What is the capital of France?"
    # Echo the prompt back, exactly the case the parrot guard targets.
    completion = (
        "<budget>10</budget>"
        "<answer>What is the capital of France? Paris is.</answer>"
    )
    rewards = adapter([prompt], [completion])

    assert rewards[0] == pytest.approx(-0.5), (
        f"parrot cliff should yield -0.5, got {rewards[0]}"
    )
    assert log.to_records()[0]["error"] == "parrot"


def test_bad_format_cliff_still_fires_through_adapter():
    log = RewardLog()
    adapter = InProcessRewardAdapter(num_slots=1, log=log)

    rewards = adapter(
        ["What is the capital of France?"],
        ["Paris"],  # no <budget>/<answer> tags at all
    )
    assert rewards[0] == pytest.approx(-1.0)
    assert log.to_records()[0]["error"] == "bad_format"


# ─── Reward adapter — defensive paths ──────────────────────────────────
def test_unknown_prompt_returns_zero_not_crash():
    """If TRL ever feeds us a prompt outside PROMPT_BANK, log + 0.0,
    don't kill the entire training run."""
    adapter = InProcessRewardAdapter(num_slots=1)
    rewards = adapter(
        ["This question is not in the prompt bank at all."],
        ["<budget>3</budget><answer>nope</answer>"],
    )
    assert rewards == [0.0]


def test_mismatched_lengths_raise_loudly():
    adapter = InProcessRewardAdapter(num_slots=1)
    with pytest.raises(ValueError, match="len\\(prompts\\)"):
        adapter(["a", "b"], ["only one"])


# ─── Completion text normalisation ─────────────────────────────────────
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("plain string", "plain string"),
        ([{"role": "assistant", "content": "chat reply"}], "chat reply"),
        ({"role": "assistant", "content": "lone dict"}, "lone dict"),
        # Multi-message conversation: take the LAST message.
        (
            [
                {"role": "user", "content": "ignored"},
                {"role": "assistant", "content": "final reply"},
            ],
            "final reply",
        ),
    ],
)
def test_completion_text_normaliser(raw, expected):
    assert _completion_text(raw) == expected


# ─── build_reward_func factory ─────────────────────────────────────────
def test_build_reward_func_picks_in_process_by_default():
    cfg = TrainingConfig()
    func, adapter = build_reward_func(cfg)
    assert isinstance(adapter, InProcessRewardAdapter)
    rewards = func(
        ["What is the capital of France?"],
        ["<budget>3</budget><answer>Paris.</answer>"],
    )
    assert rewards[0] > 0.5
    adapter.close()


def test_build_reward_func_rejects_unknown_backend():
    cfg = TrainingConfig(reward_backend="kafka")  # not a thing
    with pytest.raises(ValueError, match="Unknown reward_backend"):
        build_reward_func(cfg)
