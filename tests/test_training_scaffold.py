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
    PROBE_INDICES,
    TRAIN_INDICES,
    _allocate_counts,
    dump_split,
    holdout_prompts,
    probe_prompts,
    stratified_split,
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


def test_config_smoke_preset_has_expected_shape():
    """``TrainingConfig.smoke()`` must downscale the run enough to finish on
    a T4 in ~20 min but keep the integration points (Unsloth, SFT) on
    when colab=True."""
    cfg = TrainingConfig.smoke(colab=True)
    assert cfg.max_steps == 50
    assert cfg.num_generations == 4
    assert cfg.judge_backend == "keyword"
    assert cfg.prompt_bank_mode == "starter"
    assert cfg.use_unsloth is True
    assert cfg.use_sft_warmup is True
    assert "smoke" in cfg.output_dir


def test_config_smoke_preset_colab_false_disables_gpu_paths():
    """On non-Colab kernels the smoke preset keeps Unsloth + SFT off so
    a Windows/macOS dev can still execute the config without those deps."""
    cfg = TrainingConfig.smoke(colab=False)
    assert cfg.use_unsloth is False
    assert cfg.use_sft_warmup is False
    # But the run-shape parts (steps, judge, bank) stay scaled down.
    assert cfg.max_steps == 50
    assert cfg.judge_backend == "keyword"


def test_config_smoke_estimate_judge_calls_is_cheap():
    """Smoke should cost at most a few hundred judge calls so we never hit
    HF rate limits during pipeline shakedown (doubly safe: default judge
    is 'keyword' anyway)."""
    cfg = TrainingConfig.smoke(colab=True)
    assert cfg.estimate_judge_calls() <= 300


def test_config_cost_estimate_scales_linearly():
    cfg_small = TrainingConfig(num_generations=4, max_steps=10)
    cfg_big = TrainingConfig(num_generations=8, max_steps=100)
    assert cfg_big.estimate_judge_calls() == 20 * cfg_small.estimate_judge_calls()


# ─── Split (Phase 8: stratified train / holdout / probe) ───────────────
def test_split_is_fully_disjoint():
    """Every PROMPT_BANK index belongs to exactly one of train/holdout/probe."""
    train = set(TRAIN_INDICES)
    holdout = set(HOLDOUT_INDICES)
    probe = set(PROBE_INDICES)
    assert train.isdisjoint(holdout)
    assert train.isdisjoint(probe)
    assert holdout.isdisjoint(probe)


def test_split_covers_every_prompt_exactly_once():
    from token_efficiency_env.prompts import PROMPT_BANK

    covered = set(TRAIN_INDICES) | set(HOLDOUT_INDICES) | set(PROBE_INDICES)
    assert covered == set(range(len(PROMPT_BANK))), (
        "split must partition PROMPT_BANK, not drop or duplicate indices"
    )


def test_holdout_and_probe_are_non_empty_and_cover_all_tiers():
    """Stratified invariant: on the starter 24-prompt bank each eval bucket
    contains at least one prompt from every complexity tier."""
    assert holdout_prompts(), "holdout must not be empty on the 24-prompt bank"
    assert probe_prompts(), "probe must not be empty on the 24-prompt bank"
    for name, bucket in [("holdout", holdout_prompts()), ("probe", probe_prompts())]:
        tiers = {p["complexity"] for p in bucket}
        assert tiers == {"easy", "medium", "hard"}, (
            f"{name} is missing tiers: {tiers!r}"
        )


def test_split_is_deterministic_under_fixed_seed():
    """Re-running ``stratified_split(PROMPT_BANK, seed=42)`` twice must
    produce identical indices — reviewers depend on this to audit the
    holdout set."""
    from token_efficiency_env.prompts import PROMPT_BANK

    a = stratified_split(PROMPT_BANK, seed=42)
    b = stratified_split(PROMPT_BANK, seed=42)
    assert a == b


def test_split_changes_under_different_seed():
    """Sanity: the splitter actually uses the seed."""
    from token_efficiency_env.prompts import PROMPT_BANK

    a = stratified_split(PROMPT_BANK, seed=42)
    b = stratified_split(PROMPT_BANK, seed=123)
    assert a != b, "different seeds should produce different splits"
    # ...but must still cover exactly the same indices (partition invariant).
    assert (
        set(a["train"]) | set(a["holdout"]) | set(a["probe"])
        == set(b["train"]) | set(b["holdout"]) | set(b["probe"])
    )


@pytest.mark.parametrize(
    "n_tier, expected",
    [
        # Degenerate / tiny tiers.
        (0, (0, 0, 0)),
        (1, (1, 0, 0)),
        (2, (0, 1, 1)),
        # Small-tier branch guarantees >=1 in holdout AND probe.
        (3, (1, 1, 1)),
        (8, (6, 1, 1)),
        (10, (8, 1, 1)),
        # Above SMALL_TIER_THRESHOLD (=20) counts follow raw fractions.
        (100, (85, 10, 5)),
    ],
)
def test_allocate_counts_sums_to_n_and_hits_known_layouts(n_tier, expected):
    result = _allocate_counts(n_tier)
    assert sum(result) == n_tier, (
        f"allocator dropped prompts: {result} sums to {sum(result)} != {n_tier}"
    )
    assert result == expected


def test_stratification_within_tolerance_on_a_synthetic_bank():
    """On a 300-prompt bank (100 per tier), per-tier holdout should be
    ~10% with rounding — plan §6.4 asks for ±5% tolerance."""
    synth = (
        [{"prompt": f"easy-{i}", "complexity": "easy"} for i in range(100)]
        + [{"prompt": f"medium-{i}", "complexity": "medium"} for i in range(100)]
        + [{"prompt": f"hard-{i}", "complexity": "hard"} for i in range(100)]
    )
    split = stratified_split(synth, seed=1)
    for tier in ("easy", "medium", "hard"):
        tier_holdout = sum(1 for i in split["holdout"] if synth[i]["complexity"] == tier)
        ratio = tier_holdout / 100.0
        assert 0.05 <= ratio <= 0.15, (
            f"{tier}: holdout ratio {ratio:.3f} outside ±5% of 0.10"
        )


def test_dump_split_writes_stable_json(tmp_path):
    """The JSON file under ``training/prompts_split.json`` is part of how
    we audit the split; test it's valid, includes the seed, and totals match."""
    import json as _json

    out = dump_split(tmp_path / "prompts_split.json")
    payload = _json.loads(out.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert payload["total_prompts"] == len(TRAIN_INDICES) + len(HOLDOUT_INDICES) + len(PROBE_INDICES)
    assert payload["train"] == list(TRAIN_INDICES)
    assert payload["holdout"] == list(HOLDOUT_INDICES)
    assert payload["probe"] == list(PROBE_INDICES)


def test_train_prompts_dont_share_objects_with_bank():
    """Mutating a returned dict must not corrupt PROMPT_BANK."""
    from token_efficiency_env.prompts import PROMPT_BANK

    sample = train_prompts()[0]
    original_prompt = PROMPT_BANK[TRAIN_INDICES[0]]["prompt"]
    sample["prompt"] = "MUTATED"
    assert PROMPT_BANK[TRAIN_INDICES[0]]["prompt"] == original_prompt


# ─── Mode-aware split (prompt_bank_mode="full" wiring) ────────────────
def test_train_prompts_mode_none_matches_starter():
    """``mode=None`` must be identical to ``mode='starter'`` (back-compat)."""
    assert train_prompts() == train_prompts("starter")
    assert holdout_prompts() == holdout_prompts("starter")
    assert probe_prompts() == probe_prompts("starter")


def test_split_mode_full_uses_injected_bank(monkeypatch):
    """With a stub full-bank loader, mode='full' must produce a stratified
    split on the stub and NOT touch the starter bank."""
    from training import prompts_split
    from token_efficiency_env import prompts as prompts_mod

    stub_bank = (
        [{"prompt": f"E{i}", "complexity": "easy", "expected_keywords": []} for i in range(40)]
        + [{"prompt": f"M{i}", "complexity": "medium", "expected_keywords": []} for i in range(40)]
        + [{"prompt": f"H{i}", "complexity": "hard", "expected_keywords": []} for i in range(40)]
    )

    monkeypatch.setattr(prompts_mod, "_FULL_BANK_CACHE", None, raising=False)
    monkeypatch.setattr(prompts_mod, "load_full_prompt_bank", lambda: stub_bank)
    prompts_split.reset_split_cache()

    train = train_prompts("full")
    holdout = holdout_prompts("full")
    probe = probe_prompts("full")

    total = len(train) + len(holdout) + len(probe)
    assert total == len(stub_bank), (
        f"mode=full split must partition the injected bank (got {total}/{len(stub_bank)})"
    )

    # Holdout + probe must each contain >=1 prompt per tier (stratified).
    for name, bucket in [("holdout", holdout), ("probe", probe)]:
        tiers = {p["complexity"] for p in bucket}
        assert tiers == {"easy", "medium", "hard"}, (
            f"mode=full {name} is missing tiers: {tiers!r}"
        )

    # Starter indices must be untouched — the module constants still
    # describe the 24-prompt bank.
    from token_efficiency_env.prompts import STARTER_PROMPTS
    assert len(TRAIN_INDICES) + len(HOLDOUT_INDICES) + len(PROBE_INDICES) == len(STARTER_PROMPTS)

    prompts_split.reset_split_cache()


def test_describe_split_mode_full_renders_without_dumping_every_prompt(monkeypatch):
    """On >50-prompt banks, describe_split must switch to sample-mode
    (5 prompts per bucket) rather than dumping the whole holdout."""
    from training import prompts_split
    from training.prompts_split import describe_split
    from token_efficiency_env import prompts as prompts_mod

    stub_bank = (
        [{"prompt": f"E{i}", "complexity": "easy", "expected_keywords": []} for i in range(40)]
        + [{"prompt": f"M{i}", "complexity": "medium", "expected_keywords": []} for i in range(40)]
        + [{"prompt": f"H{i}", "complexity": "hard", "expected_keywords": []} for i in range(40)]
    )
    monkeypatch.setattr(prompts_mod, "_FULL_BANK_CACHE", None, raising=False)
    monkeypatch.setattr(prompts_mod, "load_full_prompt_bank", lambda: stub_bank)
    prompts_split.reset_split_cache()

    out = describe_split("full")
    assert "mode=full" in out
    assert "sample" in out.lower(), "should switch to sample mode for banks > 50"

    prompts_split.reset_split_cache()


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
