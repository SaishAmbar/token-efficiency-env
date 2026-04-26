"""Judge-ready before/after comparison report.

The notebook runs two :func:`notebooks.eval_baseline_vs_trained.evaluate`
passes — once against the untrained base model, once against the trained
LoRA adapter — and hands us two :class:`EvalSummary` objects. This module
turns those into the concrete files a hackathon judge (or anyone reading
the repo on GitHub) can look at without running anything:

* ``training/before_after_report.md`` — the prose+tables version. Ready
  to paste into a submission blurb or blog post.
* ``training/before_after_plot.png`` — 5-metric side-by-side bar chart.
* ``training/reward_curve.png``       — rolling training-time reward.

The module is import-safe: all heavy deps (matplotlib, pandas) are pulled
in lazily, so ``training.compare_report`` can be imported by tests even
on a machine where matplotlib isn't usable (headless CI, etc.).

Design notes
------------

* **Per-tier breakdown** matters. The headline numbers can hide huge
  wins on one tier and no-ops on another; judges find the per-tier
  table more convincing than the aggregate.
* **Qualitative samples** matter even more. A row that says
  "baseline: 84 tokens of waffle, trained: ``<budget>5</budget>
  <answer>Paris.</answer>``" is a stronger argument than any chart.
* Reports are **deterministic** modulo input — same baseline + trained
  → byte-identical markdown (no timestamps in the middle of tables) so
  diffs are reviewable.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# Tier display order — easy → medium → hard reads naturally in tables.
_TIER_ORDER = ("easy", "medium", "hard")


# ─── Aggregation ─────────────────────────────────────────────────────────
def _per_tier_means(
    per_prompt: Sequence[Any],
) -> Dict[str, Dict[str, float]]:
    """Compute mean metrics per complexity tier.

    Accepts ``per_prompt`` as either a sequence of ``PerPromptResult``
    dataclasses or of plain dicts (so we can test this with synthetic
    data without importing the eval module's dataclass).
    """
    by_tier: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_prompt:
        # Dataclass → asdict; plain dict → passthrough; any other object
        # (e.g. SimpleNamespace used in tests) → vars().
        if hasattr(row, "__dataclass_fields__"):
            r = asdict(row)
        elif isinstance(row, Mapping):
            r = dict(row)
        else:
            r = dict(vars(row))
        tier = r.get("complexity") or "unknown"
        by_tier[tier].append(r)

    out: Dict[str, Dict[str, float]] = {}
    for tier, rows in by_tier.items():
        if not rows:
            continue
        out[tier] = {
            "n_prompts":         len(rows),
            "mean_reward":       statistics.fmean(r["reward"] for r in rows),
            "mean_correctness":  statistics.fmean(r["correctness"] for r in rows),
            "mean_tokens_used":  statistics.fmean(r["tokens_used"] for r in rows),
            "mean_allocated_budget": statistics.fmean(r["allocated_budget"] for r in rows),
            "overshoot_rate":    sum(1 for r in rows if r.get("overshoot")) / len(rows),
            "cliff_rate":        sum(1 for r in rows if r.get("error")) / len(rows),
        }
    return out


def build_comparison(baseline: Any, trained: Any) -> Dict[str, Any]:
    """Aggregate a baseline/trained pair into a single comparison dict.

    Returns
    -------
    dict
        ``{"overall": {baseline, trained, delta}, "per_tier": {...}}`` —
        a structure both the markdown renderer and a JSON dump can consume.
    """
    def _overall(summary: Any) -> Dict[str, float]:
        return {
            "n_prompts":             summary.n_prompts,
            "mean_reward":           summary.mean_reward,
            "mean_correctness":      summary.mean_correctness,
            "mean_tokens_used":      summary.mean_tokens_used,
            "mean_allocated_budget": summary.mean_allocated_budget,
            "overshoot_rate":        summary.overshoot_rate,
            "cliff_rate":            summary.cliff_rate,
        }

    b_overall = _overall(baseline)
    t_overall = _overall(trained)
    delta = {k: t_overall[k] - b_overall[k] for k in b_overall if k != "n_prompts"}

    b_tier = _per_tier_means(baseline.per_prompt)
    t_tier = _per_tier_means(trained.per_prompt)

    per_tier: Dict[str, Dict[str, Any]] = {}
    for tier in _TIER_ORDER:
        if tier not in b_tier and tier not in t_tier:
            continue
        per_tier[tier] = {
            "baseline": b_tier.get(tier, {}),
            "trained":  t_tier.get(tier, {}),
            "delta": {
                k: (t_tier.get(tier, {}).get(k, 0.0) - b_tier.get(tier, {}).get(k, 0.0))
                for k in ("mean_reward", "mean_correctness", "mean_tokens_used",
                          "overshoot_rate", "cliff_rate")
            },
        }

    return {
        "overall": {
            "baseline": b_overall,
            "trained": t_overall,
            "delta": delta,
        },
        "per_tier": per_tier,
    }


# ─── Qualitative samples ─────────────────────────────────────────────────
def _find_sample_pairs(
    baseline: Any,
    trained: Any,
    *,
    max_samples: int = 3,
) -> List[Dict[str, str]]:
    """Pick ``max_samples`` (prompt, baseline_answer, trained_answer) rows
    where the trained model's output is meaningfully shorter.

    Heuristic: prefer rows where baseline was a cliff (bad format) AND
    trained wasn't, because those are the most striking before/after
    stories for a reader. Fall back to largest-token-reduction rows.
    """
    b_by_prompt = {row.prompt: row for row in baseline.per_prompt}
    t_by_prompt = {row.prompt: row for row in trained.per_prompt}

    pairs = []
    for prompt, t_row in t_by_prompt.items():
        b_row = b_by_prompt.get(prompt)
        if b_row is None:
            continue
        pairs.append({
            "prompt": prompt,
            "complexity": t_row.complexity,
            "baseline": b_row.completion,
            "baseline_tokens": b_row.tokens_used,
            "baseline_cliff": bool(b_row.error),
            "trained": t_row.completion,
            "trained_tokens": t_row.tokens_used,
            "trained_cliff": bool(t_row.error),
            "token_delta": b_row.tokens_used - t_row.tokens_used,
        })

    # Prioritise cliff → clean, then by token reduction.
    pairs.sort(
        key=lambda p: (
            not (p["baseline_cliff"] and not p["trained_cliff"]),
            -p["token_delta"],
        )
    )
    return pairs[:max_samples]


# ─── Markdown rendering ──────────────────────────────────────────────────
def _fmt_delta(value: float, *, higher_is_better: bool = True, pct: bool = False) -> str:
    """Format a delta with an arrow indicator matched to desired direction."""
    arrow = (
        "▲" if (value > 0) == higher_is_better else ("▼" if value != 0 else "–")
    )
    if pct:
        return f"{arrow} {value * 100:+.1f}%"
    return f"{arrow} {value:+.3f}"


def _overall_table(comp: Mapping[str, Any]) -> str:
    b, t, d = (comp["overall"][k] for k in ("baseline", "trained", "delta"))
    rows = [
        # (name, baseline, trained, delta, higher_is_better, is_pct, abs_fmt)
        ("Mean reward",      b["mean_reward"],       t["mean_reward"],       d["mean_reward"],       True,  False, "{:+.3f}"),
        ("Correctness",      b["mean_correctness"],  t["mean_correctness"],  d["mean_correctness"],  True,  False, "{:.3f}"),
        ("Mean tokens used", b["mean_tokens_used"],  t["mean_tokens_used"],  d["mean_tokens_used"],  False, False, "{:.1f}"),
        ("Cliff rate",       b["cliff_rate"],        t["cliff_rate"],        d["cliff_rate"],        False, True,  None),
        ("Overshoot rate",   b["overshoot_rate"],    t["overshoot_rate"],    d["overshoot_rate"],    False, True,  None),
    ]
    lines = [
        "_Arrows: ▲ = improvement, ▼ = regression (independent of sign; lower tokens = good)._",
        "",
        "| Metric           | Baseline  | Trained   | Δ             |",
        "|------------------|-----------|-----------|---------------|",
    ]
    for name, bv, tv, dv, higher_better, is_pct, abs_fmt in rows:
        if is_pct:
            bstr, tstr = f"{bv * 100:.1f}%", f"{tv * 100:.1f}%"
        else:
            bstr, tstr = abs_fmt.format(bv), abs_fmt.format(tv)
        lines.append(
            f"| {name:<16} | {bstr:<9} | {tstr:<9} | "
            f"{_fmt_delta(dv, higher_is_better=higher_better, pct=is_pct):<13} |"
        )
    return "\n".join(lines)


def _tier_table(comp: Mapping[str, Any]) -> str:
    if not comp["per_tier"]:
        return "_(no per-tier data — baseline/trained were empty or missing complexity tags)_"

    lines = [
        "| Tier   | N | Reward (B→T) | Tokens (B→T) | Cliff (B→T) |",
        "|--------|---|--------------|--------------|-------------|",
    ]
    for tier in _TIER_ORDER:
        if tier not in comp["per_tier"]:
            continue
        pt = comp["per_tier"][tier]
        b, t = pt["baseline"], pt["trained"]
        n = b.get("n_prompts") or t.get("n_prompts") or 0
        lines.append(
            f"| {tier:<6} | {n} | "
            f"{b.get('mean_reward', 0):+.2f} → {t.get('mean_reward', 0):+.2f} | "
            f"{b.get('mean_tokens_used', 0):.0f} → {t.get('mean_tokens_used', 0):.0f} | "
            f"{b.get('cliff_rate', 0) * 100:.0f}% → {t.get('cliff_rate', 0) * 100:.0f}% |"
        )
    return "\n".join(lines)


def _samples_section(samples: Sequence[Mapping[str, Any]]) -> str:
    if not samples:
        return "_(no qualitative samples — insufficient overlap between baseline/trained prompts)_"
    parts = []
    for i, s in enumerate(samples, 1):
        parts.append(
            f"### Sample {i} — *{s['complexity']}* prompt\n"
            f"> **Question:** {s['prompt']}\n\n"
            f"**Baseline** ({s['baseline_tokens']} tokens"
            f"{', **cliff**' if s['baseline_cliff'] else ''}):\n"
            f"```\n{s['baseline'].strip()[:400]}\n```\n\n"
            f"**Trained** ({s['trained_tokens']} tokens"
            f"{', **cliff**' if s['trained_cliff'] else ''}):\n"
            f"```\n{s['trained'].strip()[:400]}\n```\n"
        )
    return "\n".join(parts)


def render_markdown_report(
    baseline: Any,
    trained: Any,
    *,
    output_path: Path | str = "training/before_after_report.md",
    run_label: str = "full",
    config_banner: Optional[str] = None,
) -> Path:
    """Write a judge-ready before/after comparison report.

    Parameters
    ----------
    baseline, trained : EvalSummary
        Results of running ``evaluate()`` on the base and trained models.
    output_path : Path | str
        Where to write the markdown file.
    run_label : str
        "smoke" | "full" — shown in the report header so readers can tell
        which artifact they're looking at.
    config_banner : str, optional
        Output of ``TrainingConfig.describe()``. Included verbatim so the
        report is self-contained.
    """
    comp = build_comparison(baseline, trained)
    samples = _find_sample_pairs(baseline, trained)

    header = [
        "# Before / After Comparison — TokenEfficiencyEnv",
        "",
        f"**Run mode:** `{run_label}`  ",
        f"**Prompts evaluated:** {baseline.n_prompts}",
        "",
    ]
    if config_banner:
        header += ["## Training config", "", "```", config_banner, "```", ""]

    headline = [
        "## Headline metrics",
        "",
        _overall_table(comp),
        "",
    ]

    tier = [
        "## Per-tier breakdown",
        "",
        _tier_table(comp),
        "",
    ]

    qual = [
        "## Qualitative samples (most striking before/after)",
        "",
        _samples_section(samples),
        "",
    ]

    footer = [
        "## Plots",
        "",
        "![Before/After bar chart](before_after_plot.png)",
        "![Training reward curve](reward_curve.png)",
        "",
        "_Generated by `training/compare_report.py`. To regenerate, rerun the last cells of `notebooks/train_grpo.ipynb`._",
        "",
    ]

    body = "\n".join(header + headline + tier + qual + footer)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    logger.info("compare_report: wrote %s", out.resolve())
    return out


# ─── Plots ───────────────────────────────────────────────────────────────
def _use_agg_backend() -> None:
    """Force a non-interactive matplotlib backend so tests don't need a display."""
    import matplotlib  # type: ignore[import-not-found]
    # Only switch when nothing's been set up yet; avoids blowing up an
    # active Jupyter/notebook display.
    if matplotlib.get_backend().lower() != "agg":
        try:
            matplotlib.use("Agg", force=False)
        except Exception:  # pragma: no cover — already attached to a GUI backend
            pass


def plot_before_after(
    baseline: Any,
    trained: Any,
    *,
    output_path: Path | str = "training/before_after_plot.png",
) -> Path:
    """Save a 5-metric bar chart comparing baseline vs trained."""
    _use_agg_backend()
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    metrics = [
        ("reward",         baseline.mean_reward,       trained.mean_reward,       True),
        ("correctness",    baseline.mean_correctness,  trained.mean_correctness,  True),
        ("tokens",         baseline.mean_tokens_used,  trained.mean_tokens_used,  False),
        ("cliff rate",     baseline.cliff_rate,        trained.cliff_rate,        False),
        ("overshoot",      baseline.overshoot_rate,    trained.overshoot_rate,    False),
    ]
    labels = [m[0] for m in metrics]
    b_vals = [m[1] for m in metrics]
    t_vals = [m[2] for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(labels))
    width = 0.4
    ax.bar([i - width / 2 for i in x], b_vals, width=width, label="baseline", color="#b0b0b0")
    ax.bar([i + width / 2 for i in x], t_vals, width=width, label="trained",  color="#3b82f6")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.set_title("Before / After — TokenEfficiencyEnv")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("compare_report: wrote %s", out.resolve())
    return out


def plot_reward_curve(
    reward_log: Any,
    *,
    output_path: Path | str = "training/reward_curve.png",
    window: int = 25,
    num_generations: int = 8,
) -> Optional[Path]:
    """Save the rolling training-reward trajectory. ``None`` if log is empty."""
    _use_agg_backend()
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    import pandas as pd  # type: ignore[import-not-found]

    records = reward_log.to_records() if hasattr(reward_log, "to_records") else list(reward_log)
    if not records:
        logger.warning("compare_report: reward_log is empty, skipping curve plot")
        return None

    df = pd.DataFrame(records)
    df["step"] = df.index // max(num_generations, 1)
    rolling = df.groupby("step")["reward"].mean().rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rolling.index, rolling.values, color="#3b82f6")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(f"reward (rolling mean, window={window})")
    ax.set_title("Training-time reward trajectory")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("compare_report: wrote %s", out.resolve())
    return out


__all__ = [
    "build_comparison",
    "render_markdown_report",
    "plot_before_after",
    "plot_reward_curve",
]
