"""Single source of truth for every Phase 7 training knob.

The notebook instantiates one ``TrainingConfig`` at the top, then passes the
relevant slice to each downstream component. Keeping every knob here (instead
of scattered across notebook cells) makes runs reproducible and lets
``tests/test_training_scaffold.py`` import the *same* defaults the notebook
uses.

Cost guard
----------
GRPO calls the judge ``num_generations × max_steps`` times. The
``estimate_judge_calls`` method prints that number up front so you can spot
"this run will hit the HF Inference rate limit" *before* hitting Train.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    # ─── Trainee model ────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    # max input prompt length (tokens). Our prompts are short questions;
    # 256 is a comfortable upper bound that stays well under Qwen's 32k.
    max_prompt_length: int = 256
    # max length the model is allowed to generate per rollout.
    max_completion_length: int = 256

    # ─── LoRA (PEFT) ──────────────────────────────────────────────────
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # ─── GRPO ─────────────────────────────────────────────────────────
    # Number of completions sampled per prompt. Group-relative advantage
    # only has signal when this is >1; TRL's default is 8.
    num_generations: int = 8
    # Total GRPO optimisation steps. 300 is the "see it move on a single GPU
    # in 1-2h" budget — bump it up once you trust the config.
    max_steps: int = 300
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 4
    # batch size = num_generations * per_device_train_batch_size, and TRL
    # requires per_device_train_batch_size % num_generations == 0. We use 1
    # because anything bigger blows past 16GB with Qwen-3B + LoRA + grad
    # accumulation. Increase only if you've got a 24GB+ card.
    per_device_train_batch_size: int = 1
    # Sampling temperature for rollouts. >0 is required for GRPO to get
    # diverse completions to compare against each other.
    temperature: float = 0.9
    # Where TRL writes checkpoints + logs.
    output_dir: str = "outputs/grpo_qwen2.5_3b"

    # ─── Reward adapter / env pool ────────────────────────────────────
    # "in_process" = cheap, default, no servers. "ws" = use server_pool.
    reward_backend: str = "in_process"
    # Only used when reward_backend == "ws". One server per concurrent
    # rollout slot keeps each session's curriculum/recent_rewards isolated.
    num_env_servers: int = 1  # in_process needs no parallel servers
    base_port: int = 8100  # WS pool ports = base_port .. base_port + N - 1

    # ─── Judge ────────────────────────────────────────────────────────
    # "huggingface" makes correctness real (and costs API calls).
    # "keyword" is the free, deterministic offline judge — fine for a dry
    # run that just verifies the trainer doesn't crash.
    judge_backend: str = "huggingface"

    # ─── Eval ─────────────────────────────────────────────────────────
    # Greedy decoding for eval so before/after numbers are reproducible.
    eval_temperature: float = 0.0
    # How many completions per held-out prompt to average over (with t>0).
    # Kept at 1 because eval_temperature defaults to 0; bump to 4-8 if you
    # switch to stochastic eval.
    eval_samples_per_prompt: int = 1

    # ─── Reproducibility ──────────────────────────────────────────────
    seed: int = 42

    # ─── Helpers ──────────────────────────────────────────────────────
    def estimate_judge_calls(self) -> int:
        """How many judge API calls this config will make during training.

        Each GRPO step samples ``num_generations`` completions per prompt,
        and each completion → one judge call. ``per_device_train_batch_size``
        of 1 means one prompt per step.
        """
        return self.num_generations * self.max_steps

    def describe(self) -> str:
        """Short banner the notebook prints up front."""
        return (
            f"TrainingConfig\n"
            f"  model        : {self.model_name}\n"
            f"  LoRA         : r={self.lora_r}, α={self.lora_alpha}, "
            f"targets={len(self.lora_target_modules)} modules\n"
            f"  GRPO         : steps={self.max_steps}, "
            f"num_generations={self.num_generations}, lr={self.learning_rate}\n"
            f"  reward       : backend={self.reward_backend} "
            f"({self.num_env_servers} env(s)), judge={self.judge_backend}\n"
            f"  cost estimate: ~{self.estimate_judge_calls():,} judge calls\n"
            f"  seed         : {self.seed}\n"
            f"  output_dir   : {self.output_dir}"
        )
