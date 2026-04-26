"""One-shot smoke of training.compare_report with synthetic data.

Useful as a visual sanity check: run this and eyeball
``training/before_after_report.md`` (+ the two PNGs) to make sure the
rendering looks right BEFORE you commit to a multi-hour Colab training
run that would otherwise be the only way to exercise the report path.

    python scripts/smoke_compare_report.py

Does NOT touch real models, datasets, or HF endpoints.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from training.compare_report import (  # noqa: E402
    plot_before_after,
    plot_reward_curve,
    render_markdown_report,
)


@dataclass
class FakePerPrompt:
    prompt: str
    complexity: str
    completion: str
    reward: float
    correctness: float
    tokens_used: int
    allocated_budget: int
    overshoot: bool
    error: str
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class FakeSummary:
    label: str
    n_prompts: int
    mean_reward: float
    mean_correctness: float
    mean_tokens_used: float
    mean_allocated_budget: float
    overshoot_rate: float
    cliff_rate: float
    per_prompt: List[FakePerPrompt] = field(default_factory=list)


def _summary(label: str, rows: List[FakePerPrompt]) -> FakeSummary:
    n = len(rows)
    return FakeSummary(
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


def main() -> int:
    baseline_rows = [
        FakePerPrompt("What is the capital of France?", "easy",
                      "The capital of France is the wonderful city of Paris, located...",
                      -0.50, 0.20, 84, 20, True, "too_long"),
        FakePerPrompt("What is 12 * 12?", "easy",
                      "I don't know",
                      -0.30, 0.00, 12, 20, False, "bad_format"),
        FakePerPrompt("Who wrote Hamlet?", "easy",
                      "Shakespeare wrote the play Hamlet in the early 1600s",
                      -0.10, 0.60, 45, 30, False, ""),
        FakePerPrompt("Solve x^2 = 49 for positive x.", "medium",
                      "<budget>5</budget>x=7",
                      -0.20, 0.50, 18, 5, True, "bad_format"),
        FakePerPrompt("Derivative of sin(x)", "medium",
                      "The derivative is cos(x)",
                      0.10, 0.70, 28, 30, False, ""),
        FakePerPrompt("Explain the halting problem.", "hard",
                      "The halting problem is a fundamental result in computability theory...",
                      -0.05, 0.40, 130, 100, True, "too_long"),
    ]
    trained_rows = [
        FakePerPrompt("What is the capital of France?", "easy",
                      "<budget>5</budget><answer>Paris.</answer>",
                      0.85, 1.00, 8, 5, False, ""),
        FakePerPrompt("What is 12 * 12?", "easy",
                      "<budget>5</budget><answer>144.</answer>",
                      0.80, 1.00, 8, 5, False, ""),
        FakePerPrompt("Who wrote Hamlet?", "easy",
                      "<budget>6</budget><answer>Shakespeare.</answer>",
                      0.72, 1.00, 9, 6, False, ""),
        FakePerPrompt("Solve x^2 = 49 for positive x.", "medium",
                      "<budget>6</budget><answer>7.</answer>",
                      0.60, 0.90, 9, 6, False, ""),
        FakePerPrompt("Derivative of sin(x)", "medium",
                      "<budget>7</budget><answer>cos(x).</answer>",
                      0.55, 0.95, 10, 7, False, ""),
        FakePerPrompt("Explain the halting problem.", "hard",
                      "<budget>40</budget><answer>Whether a program halts is undecidable for arbitrary inputs (Turing, 1936).</answer>",
                      0.25, 0.70, 45, 40, True, ""),
    ]

    baseline = _summary("baseline", baseline_rows)
    trained = _summary("trained", trained_rows)

    report_dir = _REPO_ROOT / "training"
    report_dir.mkdir(parents=True, exist_ok=True)

    md = render_markdown_report(
        baseline, trained,
        output_path=report_dir / "before_after_report_sample.md",
        run_label="sample (synthetic)",
        config_banner=(
            "model=Qwen/Qwen2.5-3B-Instruct\n"
            "lora_r=16 lora_alpha=32\n"
            "max_steps=300 num_generations=8\n"
            "(synthetic numbers — for rendering sanity only)"
        ),
    )
    png = plot_before_after(
        baseline, trained,
        output_path=report_dir / "before_after_plot_sample.png",
    )
    curve_log = [{"reward": 0.01 * i - 0.4} for i in range(300)]
    curve = plot_reward_curve(
        curve_log,
        output_path=report_dir / "reward_curve_sample.png",
        num_generations=8,
    )

    print("Wrote sample artifacts:")
    print(f"  - {md.relative_to(_REPO_ROOT)}")
    print(f"  - {png.relative_to(_REPO_ROOT)}")
    if curve is not None:
        print(f"  - {curve.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
