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

    # ─── SFT format warmup (optional; highly recommended before GRPO) ─
    # A fresh Qwen2.5-3B has never seen our
    # ``<budget>N</budget><answer>...</answer>`` format. On GRPO step 1
    # that means every rollout cliffs out at "bad_format" (-0.8 reward),
    # the reward signal has zero variance, and the policy gradient is
    # zero — i.e. the first ~50-100 steps of GRPO are wasted while the
    # model stumbles onto the format by chance.
    #
    # A 1-epoch SFT warmup on ~50 synthetic ``(question, formatted_answer)``
    # pairs teaches the format in ~2 min on a T4. After SFT, the model
    # produces valid format >95% of the time → GRPO gets a real reward
    # signal from step 1 and can actually learn *what* to answer, not
    # *how* to format.
    #
    # Off by default so the test suite and any pure-keyword offline run
    # stays identical; the notebook flips this to True on Colab.
    use_sft_warmup: bool = False
    sft_warmup_epochs: int = 1
    sft_warmup_examples: int = 50
    sft_warmup_lr: float = 2e-4
    sft_warmup_batch_size: int = 4

    # ─── Unsloth acceleration (optional; Linux + CUDA only) ───────────
    # Route model loading + LoRA through Unsloth's ``FastLanguageModel``
    # for ~2× training throughput and ~40% lower VRAM (per Unsloth/TRL
    # cookbook numbers). Kept off by default so the HF+TRL path — which
    # runs anywhere torch does — stays the tested baseline. See
    # ``requirements-train.txt`` for the platform gate.
    use_unsloth: bool = False
    # Unsloth loads in 4-bit by default to fit bigger models on small
    # GPUs. Flip to False if you're on a >=24 GB card and want bf16.
    unsloth_load_in_4bit: bool = True
    # Unsloth's max-seq override. Needs to be >= max_prompt_length +
    # max_completion_length; set explicitly so we don't silently truncate.
    unsloth_max_seq_length: int = 2048

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
    # Run the periodic probe-set eval every N training steps (Phase 8 /
    # Stage 6.C). Set to 0 to disable mid-training probing; the final
    # holdout eval still runs at the end.
    eval_steps: int = 50

    # ─── Prompt bank (Phase 8 / Stage 6.A+B) ──────────────────────────
    # "starter" = the 24-prompt hand-curated bank (default, offline-safe).
    # "full" = the programmatic ~2.3k-prompt bank loaded from HF datasets
    # (requires ``datasets`` + network; see
    # ``token_efficiency_env.prompt_bank_loader``).
    prompt_bank_mode: str = "starter"
    # Fraction of the bank reserved for periodic on-policy probing. The
    # ``training.prompts_split`` module reads these at import time; we
    # keep them on ``TrainingConfig`` so they're documented in one place
    # and overridable from the notebook.
    train_fraction: float = 0.85
    holdout_fraction: float = 0.10
    probe_fraction: float = 0.05

    # ─── Reproducibility ──────────────────────────────────────────────
    seed: int = 42

    # ─── Presets ──────────────────────────────────────────────────────
    @classmethod
    def smoke(cls, *, colab: bool = False) -> "TrainingConfig":
        """Pre-set for a ~15-20 min end-to-end smoke run (plan §5).

        Purpose: verify the **full** pipeline (bootstrap → SFT warmup →
        GRPO → eval → report) runs without OOM, TRL API drift, or
        judge-API failures **before** committing to the 1-2h full run
        (Step 6).

        What it changes vs the full-run defaults:

        * ``max_steps = 50``        (vs 300) — finishes in ~15-20 min on T4.
        * ``num_generations = 4``   (vs 8)   — halves VRAM + judge calls.
        * ``judge_backend = "keyword"``      — zero HF API calls, zero
                                               risk of rate-limits during
                                               pipeline shakedown.
        * ``prompt_bank_mode = "starter"``   — skips the 2.3k loader
                                               download; smoke wants to
                                               test *code paths*, not
                                               data scale.
        * ``eval_steps = 25``                — one mid-training probe so
                                               we see the trajectory, not
                                               just start/end.
        * Output dir is namespaced ``..._smoke`` so smoke checkpoints
          don't clobber the real run's artifacts.

        Unsloth + SFT warmup stay ON (when ``colab=True``) — those are
        the integration points most likely to break, which is exactly
        what smoke mode is meant to catch.
        """
        return cls(
            max_steps=50,
            num_generations=4,
            judge_backend="keyword",
            prompt_bank_mode="starter",
            use_unsloth=colab,
            use_sft_warmup=colab,
            output_dir="outputs/grpo_qwen2.5_3b_smoke",
            eval_steps=25,
        )

    # ─── Helpers ──────────────────────────────────────────────────────
    def estimate_judge_calls(self) -> int:
        """How many judge API calls this config will make during training.

        Each GRPO step samples ``num_generations`` completions per prompt,
        and each completion → one judge call. ``per_device_train_batch_size``
        of 1 means one prompt per step.
        """
        return self.num_generations * self.max_steps

    def __post_init__(self) -> None:
        if self.prompt_bank_mode not in ("starter", "full"):
            raise ValueError(
                f"prompt_bank_mode must be 'starter' or 'full', "
                f"got {self.prompt_bank_mode!r}"
            )
        total = self.train_fraction + self.holdout_fraction + self.probe_fraction
        if not 0.999 <= total <= 1.001:
            raise ValueError(
                f"train/holdout/probe fractions must sum to 1.0, "
                f"got {total:.4f}"
            )
        if self.eval_steps < 0:
            raise ValueError(f"eval_steps must be >= 0, got {self.eval_steps}")
        if self.sft_warmup_epochs < 1:
            raise ValueError(
                f"sft_warmup_epochs must be >= 1, got {self.sft_warmup_epochs}"
            )
        if self.sft_warmup_examples < 1:
            raise ValueError(
                f"sft_warmup_examples must be >= 1, got {self.sft_warmup_examples}"
            )

    def describe(self) -> str:
        """Short banner the notebook prints up front."""
        accel = (
            f"Unsloth (4bit={self.unsloth_load_in_4bit}, "
            f"max_seq={self.unsloth_max_seq_length})"
            if self.use_unsloth
            else "vanilla HF + PEFT"
        )
        sft = (
            f"SFT warmup ({self.sft_warmup_examples} ex, "
            f"{self.sft_warmup_epochs} epoch(s), lr={self.sft_warmup_lr})"
            if self.use_sft_warmup
            else "off (GRPO starts cold)"
        )
        return (
            f"TrainingConfig\n"
            f"  model        : {self.model_name}\n"
            f"  LoRA         : r={self.lora_r}, α={self.lora_alpha}, "
            f"targets={len(self.lora_target_modules)} modules\n"
            f"  acceleration : {accel}\n"
            f"  SFT warmup   : {sft}\n"
            f"  GRPO         : steps={self.max_steps}, "
            f"num_generations={self.num_generations}, lr={self.learning_rate}\n"
            f"  prompt bank  : mode={self.prompt_bank_mode}, "
            f"split={self.train_fraction:.2f}/{self.holdout_fraction:.2f}/"
            f"{self.probe_fraction:.2f} (train/holdout/probe)\n"
            f"  reward       : backend={self.reward_backend} "
            f"({self.num_env_servers} env(s)), judge={self.judge_backend}\n"
            f"  eval         : every {self.eval_steps} step(s) "
            f"(0 = off), temperature={self.eval_temperature}\n"
            f"  cost estimate: ~{self.estimate_judge_calls():,} judge calls\n"
            f"  seed         : {self.seed}\n"
            f"  output_dir   : {self.output_dir}"
        )
