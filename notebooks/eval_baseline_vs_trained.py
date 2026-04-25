"""Standalone eval: run a model against the held-out prompts and report metrics.

Used twice by ``notebooks/train_grpo.ipynb``:

    1. Once before training, against the BASE model — establishes a baseline.
    2. Once after training, against the LoRA-merged model — quantifies the
       win (lower tokens, non-decreasing correctness).

Can also be run directly from the CLI for ad-hoc checks::

    python -m notebooks.eval_baseline_vs_trained \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --output baseline.json

The output JSON is the input to the comparison plots in the notebook.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("eval_baseline_vs_trained")

SYSTEM_PROMPT = (
    "You are a concise assistant being evaluated on token efficiency. "
    "Predict the smallest number of answer tokens you'll need (your "
    "'budget'), then answer the question.\n\n"
    "Format strictly as: <budget>N</budget><answer>your answer</answer>\n"
    "Be as short as possible while still being correct. Do NOT exceed your "
    "predicted budget."
)


@dataclass
class PerPromptResult:
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
class EvalSummary:
    label: str
    n_prompts: int
    mean_reward: float
    mean_correctness: float
    mean_tokens_used: float
    mean_allocated_budget: float
    overshoot_rate: float
    cliff_rate: float
    per_prompt: List[PerPromptResult] = field(default_factory=list)


def _build_chat_prompt(tokenizer, question: str) -> str:
    """Render the chat template down to a single string the model can complete."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _generate_one(model, tokenizer, prompt_text: str, *, max_new_tokens: int, temperature: float) -> str:
    """Single forward pass; returns just the assistant's continuation."""
    import torch  # local import — eval module shouldn't pull torch unless run

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    do_sample = temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=0.95 if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # Strip everything up to (and including) the rendered prompt so we
    # only pass the assistant's continuation back to the env.
    decoded_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    if full.startswith(decoded_prompt):
        return full[len(decoded_prompt):].strip()
    return full.strip()


def evaluate(
    model,
    tokenizer,
    prompts: Sequence[Dict],
    *,
    label: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    samples_per_prompt: int = 1,
) -> EvalSummary:
    """Run ``model`` against ``prompts`` and return aggregated metrics.

    Each prompt is decoded ``samples_per_prompt`` times; the returned
    per-prompt result holds the FIRST sample (for printing) but the means
    are over all samples to keep stochastic eval honest.
    """
    from training.reward_adapter import InProcessRewardAdapter, RewardLog

    log = RewardLog()
    adapter = InProcessRewardAdapter(num_slots=1, log=log)

    per_prompt: List[PerPromptResult] = []
    cliff_count = 0

    for task in prompts:
        question = task["prompt"]
        rendered = _build_chat_prompt(tokenizer, question)

        completions = [
            _generate_one(
                model, tokenizer, rendered,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for _ in range(samples_per_prompt)
        ]

        # Score every sample so the means use all of them.
        rewards = adapter([question] * len(completions), completions)

        # The matching log rows are the LAST `len(completions)` rows.
        recent = log.to_records()[-len(completions):]
        first = recent[0]
        components = first["components"]

        result = PerPromptResult(
            prompt=question,
            complexity=task.get("complexity", ""),
            completion=completions[0],
            reward=rewards[0],
            correctness=float(components.get("correctness", 0.0)),
            tokens_used=int(first.get("tokens_used", 0) or 0),
            allocated_budget=int(first.get("allocated_budget", 0) or 0),
            overshoot=(
                first.get("tokens_used", 0) is not None
                and first.get("allocated_budget", 0) is not None
                and first["tokens_used"] > first["allocated_budget"]
            ),
            error=first.get("error", ""),
            components=components,
        )
        per_prompt.append(result)
        if result.error:
            cliff_count += 1

    n = len(per_prompt)
    mean_reward = statistics.fmean(r.reward for r in per_prompt) if n else 0.0
    mean_correctness = statistics.fmean(r.correctness for r in per_prompt) if n else 0.0
    mean_tokens = statistics.fmean(r.tokens_used for r in per_prompt) if n else 0.0
    mean_budget = statistics.fmean(r.allocated_budget for r in per_prompt) if n else 0.0
    overshoot_rate = (
        sum(1 for r in per_prompt if r.overshoot) / n if n else 0.0
    )

    return EvalSummary(
        label=label,
        n_prompts=n,
        mean_reward=mean_reward,
        mean_correctness=mean_correctness,
        mean_tokens_used=mean_tokens,
        mean_allocated_budget=mean_budget,
        overshoot_rate=overshoot_rate,
        cliff_rate=cliff_count / n if n else 0.0,
        per_prompt=per_prompt,
    )


def summary_to_dict(summary: EvalSummary) -> Dict:
    """JSON-serialisable form (PerPromptResult → dict)."""
    d = asdict(summary)
    return d


def print_summary(summary: EvalSummary) -> None:
    """Human-readable table the notebook prints for both baseline and trained."""
    print(f"\n{'=' * 60}")
    print(f"{summary.label}  (n={summary.n_prompts})")
    print(f"{'-' * 60}")
    print(f"  mean reward       : {summary.mean_reward:.4f}")
    print(f"  mean correctness  : {summary.mean_correctness:.4f}")
    print(f"  mean tokens used  : {summary.mean_tokens_used:.1f}")
    print(f"  mean budget       : {summary.mean_allocated_budget:.1f}")
    print(f"  overshoot rate    : {summary.overshoot_rate:.1%}")
    print(f"  cliff rate        : {summary.cliff_rate:.1%}")
    print(f"{'=' * 60}\n")


# ─── CLI entrypoint (rare; the notebook calls evaluate() directly) ─────
def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", default="eval.json")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from training.prompts_split import holdout_prompts

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    summary = evaluate(
        model, tokenizer, holdout_prompts(),
        label=args.label,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print_summary(summary)
    Path(args.output).write_text(json.dumps(summary_to_dict(summary), indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
