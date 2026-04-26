"""
TokenEfficiencyEnv — GRPO Training Script
==========================================

Trains an LLM to answer questions concisely using fewer tokens via
Group Relative Policy Optimization (GRPO) with TRL.

Stack: OpenEnv (environment) + TRL (training) + Unsloth (efficiency)

Following the pattern from OpenEnv tutorial/04-training.md (Wordle GRPO).

Usage:
    # Full training (requires GPU — Colab T4 or better):
    python train_grpo.py

    # Quick smoke test (2 steps, CPU-compatible):
    python train_grpo.py --dry-run

    # Custom settings:
    python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 1 --steps 200

Requirements:
    pip install trl transformers datasets torch accelerate
    pip install unsloth  # optional, for efficiency
"""

import sys
import os
import re
import argparse
import logging
from collections import Counter

# ─── Add project paths ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "token_efficiency_env"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OpenEnv", "src"))

from prompts import PROMPT_BANK
from scorer import score_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("train_grpo")

# ═══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT — teaches the model the <budget><answer> format
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a concise, accurate question answerer.

## RESPONSE FORMAT (MANDATORY)
You MUST respond in this exact format — no other text:
<budget>N</budget><answer>your answer here</answer>

Where:
- <budget>N</budget> = the number of tokens you plan to use (integer)
- <answer>...</answer> = your concise answer

## BUDGET GUIDELINES
- Easy factual questions (e.g. "Capital of France?"): budget 15–30
- Medium explanation questions (e.g. "How does gravity work?"): budget 60–90
- Hard detailed questions (e.g. "Explain transformers in ML"): budget 120–160

## RULES
1. Be CORRECT — accuracy is most important (40% of score)
2. Be CONCISE — use as few tokens as possible (20% of score)
3. Set a REASONABLE budget that matches question difficulty (10% of score)
4. PREDICT your own usage — set budget close to what you'll actually use (10% of score)
5. Avoid REPETITION — don't repeat the same words or phrases (10% of score)
6. Include KEY FACTS — mention the essential terms and concepts (5% of score)
7. Use CLEAN formatting — no extra whitespace or restating the question (5% of score)

## EXAMPLES
Question: "What is the capital of France?"
Response: <budget>15</budget><answer>Paris.</answer>

Question: "Explain what gravity is."
Response: <budget>70</budget><answer>Gravity is a fundamental force of attraction between objects with mass. Greater mass produces stronger gravitational pull, keeping planets in orbit and holding us on Earth's surface.</answer>
"""


# ═══════════════════════════════════════════════════════════════════
#  LOCAL ENVIRONMENT (no server needed — direct Python calls)
# ═══════════════════════════════════════════════════════════════════

MAX_TOKEN_LIMIT = 200
MIN_BUDGET = 1
MAX_BUDGET = 200
MAX_ANSWER_TOKENS = 500
REPETITION_THRESHOLD = 0.6

CURRICULUM_PHASES = [
    {"name": "Phase 1", "weights": {"easy": 1.0, "medium": 0.0, "hard": 0.0}, "threshold": 0.4},
    {"name": "Phase 2", "weights": {"easy": 0.6, "medium": 0.4, "hard": 0.0}, "threshold": 0.5},
    {"name": "Phase 3", "weights": {"easy": 0.3, "medium": 0.4, "hard": 0.3}, "threshold": 0.6},
    {"name": "Phase 4", "weights": {"easy": 0.2, "medium": 0.4, "hard": 0.4}, "threshold": None},
]


class LocalEnv:
    """Lightweight local environment for training (no server overhead)."""

    def __init__(self):
        self.pools = {
            "easy": [p for p in PROMPT_BANK if p["complexity"] == "easy"],
            "medium": [p for p in PROMPT_BANK if p["complexity"] == "medium"],
            "hard": [p for p in PROMPT_BANK if p["complexity"] == "hard"],
        }
        self.phase = 0
        self.recent_rewards = []
        self.episode = 0
        self.current_task = None

    def _select_question(self):
        import random
        weights = CURRICULUM_PHASES[self.phase]["weights"]
        candidates = []
        for comp, w in weights.items():
            if w > 0:
                candidates.extend(random.choices(self.pools[comp], k=max(1, int(w * 10))))
        return random.choice(candidates) if candidates else random.choice(PROMPT_BANK)

    def _maybe_advance(self):
        if self.phase >= len(CURRICULUM_PHASES) - 1:
            return
        if len(self.recent_rewards) < 30:
            return
        avg = sum(self.recent_rewards[-50:]) / min(len(self.recent_rewards), 50)
        thr = CURRICULUM_PHASES[self.phase]["threshold"]
        if thr and avg >= thr:
            self.phase += 1
            logger.info("🎓 CURRICULUM ADVANCE → %s (avg=%.4f)", CURRICULUM_PHASES[self.phase]["name"], avg)

    def reset(self):
        self._maybe_advance()
        self.current_task = self._select_question()
        self.episode += 1
        return {
            "prompt": self.current_task["prompt"],
            "complexity": self.current_task["complexity"],
            "keywords": self.current_task.get("expected_keywords", []),
            "token_limit": MAX_TOKEN_LIMIT,
        }

    def step(self, raw_response: str):
        """Process model response and return reward + per-component scores."""
        task = self.current_task

        # Parse format
        budget_match = re.search(r"<budget>(\d+)</budget>", raw_response)
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_response, re.DOTALL)

        if not budget_match or not answer_match:
            self.recent_rewards.append(-1.0)
            return {"reward": -1.0, "details": {}, "error": "bad_format"}

        budget = max(MIN_BUDGET, min(MAX_BUDGET, int(budget_match.group(1))))
        answer = answer_match.group(1).strip()

        if not answer:
            self.recent_rewards.append(-1.0)
            return {"reward": -1.0, "details": {}, "error": "empty"}

        words = answer.lower().split()
        if len(words) > 3:
            wc = Counter(words)
            top_w, top_c = wc.most_common(1)[0]
            if top_c / len(words) > REPETITION_THRESHOLD:
                self.recent_rewards.append(-0.5)
                return {"reward": -0.5, "details": {}, "error": "repetition"}

        tokens_used = max(1, int(len(answer.split()) * 1.3))
        if tokens_used > MAX_ANSWER_TOKENS:
            self.recent_rewards.append(-0.5)
            return {"reward": -0.5, "details": {}, "error": "too_long"}

        result = score_answer(
            prompt=task["prompt"],
            response=answer,
            allocated_budget=budget,
            tokens_used=tokens_used,
            complexity=task.get("complexity", "medium"),
            expected_keywords=task.get("expected_keywords", []),
        )

        self.recent_rewards.append(result["reward"])
        if len(self.recent_rewards) > 100:
            self.recent_rewards = self.recent_rewards[-50:]

        return result


# ═══════════════════════════════════════════════════════════════════
#  REWARD FUNCTIONS (one per scorer component — Guide §7)
# ═══════════════════════════════════════════════════════════════════

def reward_correctness(completions, **kwargs):
    """40% weight — how correct is the answer?"""
    rewards = kwargs.get("correctness_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]


def reward_efficiency(completions, **kwargs):
    """20% weight — tokens used vs self-allocated budget."""
    rewards = kwargs.get("efficiency_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]


def reward_budget_reasonableness(completions, **kwargs):
    """10% weight — does budget match question complexity?"""
    rewards = kwargs.get("budget_reasonableness_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]


def reward_redundancy(completions, **kwargs):
    """10% weight — unique information density."""
    rewards = kwargs.get("redundancy_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]


def reward_self_assessment(completions, **kwargs):
    """10% weight — budget prediction accuracy."""
    rewards = kwargs.get("self_assessment_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]


def reward_keyword(completions, **kwargs):
    """5% weight — keyword verification."""
    rewards = kwargs.get("keyword_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]


def reward_format(completions, **kwargs):
    """5% weight — format quality."""
    rewards = kwargs.get("format_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]


# ═══════════════════════════════════════════════════════════════════
#  ROLLOUT FUNCTION (called by GRPOTrainer each step)
# ═══════════════════════════════════════════════════════════════════

env = LocalEnv()


def rollout_func(prompts, trainer=None):
    """
    Rollout function for GRPO training with environment interaction.

    For each prompt:
    1. env.reset() → get a question
    2. Format prompt with system prompt
    3. Generate model response
    4. env.step() → get 7-component reward
    5. Return all rewards for GRPOTrainer
    """
    from trl.experimental.openenv import generate_rollout_completions

    episode_prompt_ids = []
    episode_completion_ids = []
    episode_logprobs = []

    correctness_rewards = []
    efficiency_rewards = []
    budget_r_rewards = []
    redundancy_rewards = []
    self_assessment_rewards = []
    keyword_rewards = []
    format_rewards = []

    tokenizer = trainer.processing_class

    for prompt_text in prompts:
        # 1. Reset environment → get question
        obs = env.reset()
        question = obs["prompt"]

        # 2. Format prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # 3. Generate model response
        rollout_outputs = generate_rollout_completions(trainer, [formatted_prompt])[0]
        episode_prompt_ids.append(rollout_outputs["prompt_ids"])
        episode_completion_ids.append(rollout_outputs["completion_ids"])
        episode_logprobs.append(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # 4. Step environment → get 7-component reward
        result = env.step(completion_text)
        details = result.get("details", {})

        # 5. Collect per-component rewards
        correctness_rewards.append(details.get("correctness", 0.0))
        efficiency_rewards.append(details.get("efficiency", 0.0))
        budget_r_rewards.append(details.get("budget_reasonableness", 0.0))
        redundancy_rewards.append(details.get("redundancy", 0.0))
        self_assessment_rewards.append(details.get("self_assessment", 0.0))
        keyword_rewards.append(details.get("keyword_verification", 0.0))
        format_rewards.append(details.get("format_quality", 0.0))

        if env.episode % 10 == 0:
            avg = sum(env.recent_rewards[-50:]) / max(len(env.recent_rewards[-50:]), 1)
            logger.info(
                "EP %d | reward=%.3f | avg=%.3f | phase=%s | Q: %s",
                env.episode, result["reward"], avg,
                CURRICULUM_PHASES[env.phase]["name"],
                question[:40],
            )

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "correctness_reward": correctness_rewards,
        "efficiency_reward": efficiency_rewards,
        "budget_reasonableness_reward": budget_r_rewards,
        "redundancy_reward": redundancy_rewards,
        "self_assessment_reward": self_assessment_rewards,
        "keyword_reward": keyword_rewards,
        "format_reward": format_rewards,
    }


# ═══════════════════════════════════════════════════════════════════
#  MAIN — configure and run training
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train TokenEfficiencyEnv with GRPO")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Base model name")
    parser.add_argument("--output-dir", default="token-efficiency-grpo", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=200, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--dataset-size", type=int, default=500, help="Number of training prompts")
    parser.add_argument("--num-generations", type=int, default=4, help="GRPO group size")
    parser.add_argument("--dry-run", action="store_true", help="Quick 2-step smoke test")
    parser.add_argument("--push-to-hub", action="store_true", help="Push trained model to HF Hub")
    parser.add_argument("--use-unsloth", action="store_true", help="Use Unsloth for efficiency")
    args = parser.parse_args()

    # ─── Dry run overrides ──────────────────────────────────────
    if args.dry_run:
        args.steps = 2
        args.dataset_size = 10
        args.grad_accum = 2
        args.num_generations = 2
        logger.info("🧪 DRY RUN MODE — 2 steps, minimal config")

    logger.info("=" * 60)
    logger.info("TokenEfficiencyEnv GRPO Training")
    logger.info("=" * 60)
    logger.info("Model: %s", args.model)
    logger.info("Output: %s", args.output_dir)
    logger.info("Steps: %d | Batch: %d | Grad Accum: %d", args.steps, args.batch_size, args.grad_accum)
    logger.info("Dataset size: %d | Generations: %d", args.dataset_size, args.num_generations)

    # ─── Tokenizer ──────────────────────────────────────────────
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── Dataset ────────────────────────────────────────────────
    from datasets import Dataset

    # Repeat a generic prompt — the actual question comes from env.reset()
    dataset = Dataset.from_dict({
        "prompt": ["Answer the following question concisely."] * args.dataset_size
    })

    logger.info("Dataset created: %d prompts", len(dataset))

    # ─── GRPO Config ────────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        num_train_epochs=args.epochs,
        max_steps=args.steps,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=min(20, args.steps // 5),
        num_generations=args.num_generations,
        max_completion_length=256,
        max_prompt_length=512,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        output_dir=args.output_dir,
        logging_steps=1,
        save_steps=max(10, args.steps // 10),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=args.push_to_hub,
    )

    # ─── Trainer ────────────────────────────────────────────────
    model_name = args.model

    # Optional: Use Unsloth for memory efficiency (Guide §10, §12)
    if args.use_unsloth:
        try:
            from unsloth import FastLanguageModel
            logger.info("🚀 Using Unsloth for efficient training")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=768,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                use_gradient_checkpointing="unsloth",
            )
            model_name = model  # Pass the model object instead of name
        except ImportError:
            logger.warning("Unsloth not installed — falling back to standard training")

    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=[
            reward_correctness,
            reward_efficiency,
            reward_budget_reasonableness,
            reward_redundancy,
            reward_self_assessment,
            reward_keyword,
            reward_format,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    # ─── Train! ─────────────────────────────────────────────────
    logger.info("🏋️ Starting GRPO training...")
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s | VRAM: %.1f GB", gpu.name, gpu.total_memory / 1e9)

    trainer_stats = trainer.train()

    # ─── Save ───────────────────────────────────────────────────
    logger.info("💾 Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        logger.info("📤 Pushing to HF Hub...")
        trainer.push_to_hub()

    # ─── Summary ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("✅ Training complete!")
    logger.info("   Runtime: %.1f seconds", trainer_stats.metrics.get("train_runtime", 0))
    logger.info("   Final loss: %.4f", trainer_stats.metrics.get("train_loss", 0))
    logger.info("   Model saved to: %s", args.output_dir)
    logger.info("   Env episodes: %d | Final phase: %s",
                env.episode, CURRICULUM_PHASES[env.phase]["name"])

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_reserved() / 1e9
        logger.info("   Peak GPU memory: %.1f GB", peak_mem)

    logger.info("=" * 60)
    logger.info("Next step: run evaluate_before_after.py to compare base vs trained model")


if __name__ == "__main__":
    main()
