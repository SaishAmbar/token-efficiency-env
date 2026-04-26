"""Tests for training.compare_report — the before/after judge-ready report."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make sure the repo root is importable regardless of where pytest was invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.compare_report import (  # noqa: E402
    build_comparison,
    plot_before_after,
    plot_reward_curve,
    render_markdown_report,
)


# ─── Test fixtures ────────────────────────────────────────────────────────
def _mk_row(
    *,
    prompt: str,
    complexity: str,
    reward: float,
    correctness: float,
    tokens_used: int,
    allocated_budget: int = 20,
    overshoot: bool = False,
    error: str = "",
    completion: str = "",
) -> SimpleNamespace:
    """Shape-compatible stand-in for PerPromptResult (same field names)."""
    return SimpleNamespace(
        prompt=prompt,
        complexity=complexity,
        completion=completion or f"<budget>10</budget><answer>{prompt[:20]}</answer>",
        reward=reward,
        correctness=correctness,
        tokens_used=tokens_used,
        allocated_budget=allocated_budget,
        overshoot=overshoot,
        error=error,
        components={"correctness": correctness},
    )


def _mk_summary(label: str, rows) -> SimpleNamespace:
    """Shape-compatible stand-in for EvalSummary."""
    n = len(rows)
    return SimpleNamespace(
        label=label,
        n_prompts=n,
        mean_reward=sum(r.reward for r in rows) / n,
        mean_correctness=sum(r.correctness for r in rows) / n,
        mean_tokens_used=sum(r.tokens_used for r in rows) / n,
        mean_allocated_budget=sum(r.allocated_budget for r in rows) / n,
        overshoot_rate=sum(1 for r in rows if r.overshoot) / n,
        cliff_rate=sum(1 for r in rows if r.error) / n,
        per_prompt=rows,
    )


@pytest.fixture
def baseline_trained_pair():
    """Toy pair where trained clearly wins on tokens + cliff rate."""
    baseline_rows = [
        _mk_row(prompt="Capital of France?", complexity="easy",
                reward=-0.4, correctness=0.3, tokens_used=80,
                error="bad_format", completion="France's capital, oh..."),
        _mk_row(prompt="Square root of 144?", complexity="easy",
                reward=-0.2, correctness=0.4, tokens_used=60),
        _mk_row(prompt="Solve x^2=4", complexity="medium",
                reward=-0.1, correctness=0.5, tokens_used=90,
                error="parrot"),
        _mk_row(prompt="Integrate x^2 dx", complexity="hard",
                reward=0.0, correctness=0.2, tokens_used=120,
                error="too_long"),
    ]
    trained_rows = [
        _mk_row(prompt="Capital of France?", complexity="easy",
                reward=0.8, correctness=1.0, tokens_used=8,
                completion="<budget>5</budget><answer>Paris.</answer>"),
        _mk_row(prompt="Square root of 144?", complexity="easy",
                reward=0.6, correctness=0.9, tokens_used=10),
        _mk_row(prompt="Solve x^2=4", complexity="medium",
                reward=0.3, correctness=0.7, tokens_used=25),
        _mk_row(prompt="Integrate x^2 dx", complexity="hard",
                reward=0.1, correctness=0.5, tokens_used=40),
    ]
    return _mk_summary("baseline", baseline_rows), _mk_summary("trained", trained_rows)


# ─── build_comparison ─────────────────────────────────────────────────────
def test_build_comparison_overall_deltas_match_direction(baseline_trained_pair):
    baseline, trained = baseline_trained_pair
    comp = build_comparison(baseline, trained)

    overall = comp["overall"]
    assert overall["trained"]["mean_reward"] > overall["baseline"]["mean_reward"]
    assert overall["delta"]["mean_reward"] > 0
    # Tokens went DOWN after training → delta is negative.
    assert overall["delta"]["mean_tokens_used"] < 0
    # Cliff rate went down (3/4 → 0/4) → delta is negative.
    assert overall["delta"]["cliff_rate"] < 0


def test_build_comparison_per_tier_has_all_tiers_from_data(baseline_trained_pair):
    baseline, trained = baseline_trained_pair
    comp = build_comparison(baseline, trained)

    # Fixture has easy, medium, hard.
    assert set(comp["per_tier"].keys()) == {"easy", "medium", "hard"}

    easy = comp["per_tier"]["easy"]
    # Easy tier should show the biggest token reduction (80,60 → 8,10).
    assert easy["baseline"]["mean_tokens_used"] == 70
    assert easy["trained"]["mean_tokens_used"] == 9
    assert easy["delta"]["mean_tokens_used"] == -61


def test_build_comparison_tiers_excluded_when_no_data(baseline_trained_pair):
    baseline, trained = baseline_trained_pair
    # Drop all 'hard' rows from both summaries.
    baseline.per_prompt = [r for r in baseline.per_prompt if r.complexity != "hard"]
    trained.per_prompt = [r for r in trained.per_prompt if r.complexity != "hard"]
    baseline.n_prompts = len(baseline.per_prompt)
    trained.n_prompts = len(trained.per_prompt)

    comp = build_comparison(baseline, trained)
    assert "hard" not in comp["per_tier"]
    assert "easy" in comp["per_tier"]
    assert "medium" in comp["per_tier"]


# ─── render_markdown_report ───────────────────────────────────────────────
def test_render_markdown_report_writes_expected_sections(
    baseline_trained_pair, tmp_path
):
    baseline, trained = baseline_trained_pair
    out = render_markdown_report(
        baseline, trained,
        output_path=tmp_path / "report.md",
        run_label="smoke",
        config_banner="model=Qwen/Qwen2.5-3B\nsteps=50",
    )
    text = out.read_text(encoding="utf-8")

    assert "# Before / After Comparison" in text
    assert "**Run mode:** `smoke`" in text
    assert "## Training config" in text
    assert "model=Qwen/Qwen2.5-3B" in text
    assert "## Headline metrics" in text
    assert "## Per-tier breakdown" in text
    assert "## Qualitative samples" in text
    # All three tiers should appear in the per-tier table.
    assert "easy" in text and "medium" in text and "hard" in text
    # At least one sample (the cliff → clean prompt) should be rendered.
    assert "Capital of France" in text


def test_render_markdown_report_handles_missing_overlap(tmp_path):
    """When baseline and trained evaluated different prompts, samples
    section should gracefully degrade to a placeholder line."""
    b = _mk_summary("baseline", [
        _mk_row(prompt="A?", complexity="easy", reward=-0.3,
                correctness=0.5, tokens_used=50),
    ])
    t = _mk_summary("trained", [
        _mk_row(prompt="B?", complexity="easy", reward=0.5,
                correctness=0.9, tokens_used=10),
    ])
    out = render_markdown_report(b, t, output_path=tmp_path / "r.md")
    text = out.read_text(encoding="utf-8")
    assert "no qualitative samples" in text


# ─── Plots — skip gracefully if matplotlib is unavailable ─────────────────
matplotlib = pytest.importorskip("matplotlib")


def test_plot_before_after_writes_nonempty_png(baseline_trained_pair, tmp_path):
    baseline, trained = baseline_trained_pair
    out = plot_before_after(baseline, trained, output_path=tmp_path / "plot.png")
    assert out.exists()
    assert out.stat().st_size > 1000  # a few KB of PNG bytes, at least


def test_plot_reward_curve_writes_png_when_log_nonempty(tmp_path):
    reward_log = [{"reward": float(i) * 0.01} for i in range(200)]
    out = plot_reward_curve(
        reward_log,
        output_path=tmp_path / "curve.png",
        num_generations=8,
    )
    assert out is not None
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_reward_curve_returns_none_for_empty_log(tmp_path):
    out = plot_reward_curve(
        [],
        output_path=tmp_path / "curve.png",
        num_generations=8,
    )
    assert out is None
    assert not (tmp_path / "curve.png").exists()
