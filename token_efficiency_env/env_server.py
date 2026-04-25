"""
TokenEfficiencyEnv — Core RL Environment

Features aligned with Hackathon Self-Serve Guide:
    - Curriculum learning: adaptive difficulty progression (Guide §6)
    - Anti-reward-hacking: budget clamping, empty answer detection,
      repetition guard, answer length sanity (Guide §8)
    - Process-aware feedback: 7-component scoring with per-component
      monitoring (Guide §9, §15)
    - Structured logging for reward tracking (Guide §15)
"""

import re
import sys
import os
import time
import logging
import random

sys.path.insert(0, os.path.abspath('../OpenEnv/src'))

from openenv.core import Environment
from pydantic import BaseModel
from transformers import AutoTokenizer
from prompts import PROMPT_BANK
from scorer import score_answer

# ─── Logging Setup (Guide §15: Monitor the right things) ───────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TokenEfficiencyEnv")

# ─── Tokenizer ──────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# ─── Constants ──────────────────────────────────────────────────────
MAX_TOKEN_LIMIT = 200           # absolute max tokens allowed
MIN_BUDGET = 1                  # minimum budget model can allocate
MAX_BUDGET = MAX_TOKEN_LIMIT    # maximum budget model can allocate
MAX_ANSWER_TOKENS = 500         # sanity cap — anything above this is suspicious
STEP_TIMEOUT_SECONDS = 30       # max time for a single step
REPETITION_THRESHOLD = 0.6     # if >60% of words are the same → penalty


# ─── Curriculum Configuration (Guide §6: Keep task simple first) ────
# Phase transitions based on average reward over last N episodes
CURRICULUM_PHASES = [
    {
        "name": "Phase 1: Easy Only",
        "weights": {"easy": 1.0, "medium": 0.0, "hard": 0.0},
        "advance_threshold": 0.4,   # advance when avg reward > 0.4
    },
    {
        "name": "Phase 2: Easy + Medium",
        "weights": {"easy": 0.6, "medium": 0.4, "hard": 0.0},
        "advance_threshold": 0.5,
    },
    {
        "name": "Phase 3: Mixed",
        "weights": {"easy": 0.3, "medium": 0.4, "hard": 0.3},
        "advance_threshold": 0.6,
    },
    {
        "name": "Phase 4: Full Difficulty",
        "weights": {"easy": 0.2, "medium": 0.4, "hard": 0.4},
        "advance_threshold": None,  # final phase — no more advancement
    },
]

REWARD_WINDOW = 50  # number of recent episodes for average reward calc


class TokenEfficiencyAction(BaseModel):
    raw_response: str


class TokenEfficiencyObservation(BaseModel):
    prompt: str
    episode_token_limit: int = MAX_TOKEN_LIMIT


class TokenEfficiencyEnv(Environment):
    """
    RL environment for training LLMs to answer correctly using fewer tokens.

    The model receives a question and must:
    1. Self-allocate a token budget via <budget>N</budget>
    2. Provide a concise answer via <answer>text</answer>
    3. Be scored on 7 dimensions (correctness, efficiency, budget
       reasonableness, redundancy, self-assessment, keyword match,
       format quality)

    Supports curriculum learning — starts with easy questions and
    progresses to harder ones as performance improves.
    """

    def __init__(self):
        self.current_task = None
        self.episode_token_limit = MAX_TOKEN_LIMIT

        # ─── Curriculum state ───────────────────────────────────
        self.episode_count = 0
        self.current_phase = 0
        self.recent_rewards = []   # rolling window for avg reward

        # ─── Prompt pools by complexity ─────────────────────────
        self.prompt_pools = {
            "easy": [p for p in PROMPT_BANK if p["complexity"] == "easy"],
            "medium": [p for p in PROMPT_BANK if p["complexity"] == "medium"],
            "hard": [p for p in PROMPT_BANK if p["complexity"] == "hard"],
        }

        logger.info("Environment initialised | Phase: %s",
                     CURRICULUM_PHASES[self.current_phase]["name"])

    # ─── Curriculum: select question based on current phase ─────────
    def _select_question(self):
        """Pick a question weighted by the current curriculum phase."""
        phase = CURRICULUM_PHASES[self.current_phase]
        weights = phase["weights"]

        # Build weighted pool
        candidates = []
        for complexity, weight in weights.items():
            if weight > 0:
                pool = self.prompt_pools[complexity]
                count = max(1, int(weight * 10))  # rough weighting
                candidates.extend(random.choices(pool, k=count))

        return random.choice(candidates) if candidates else random.choice(PROMPT_BANK)

    def _maybe_advance_phase(self):
        """Check if we should advance to the next curriculum phase."""
        if self.current_phase >= len(CURRICULUM_PHASES) - 1:
            return  # already at final phase

        if len(self.recent_rewards) < REWARD_WINDOW:
            return  # not enough data yet

        avg_reward = sum(self.recent_rewards[-REWARD_WINDOW:]) / REWARD_WINDOW
        threshold = CURRICULUM_PHASES[self.current_phase]["advance_threshold"]

        if threshold is not None and avg_reward >= threshold:
            self.current_phase += 1
            logger.info(
                "🎓 CURRICULUM ADVANCE | Now at: %s | Avg reward was: %.4f",
                CURRICULUM_PHASES[self.current_phase]["name"], avg_reward
            )

    # ─── Core Environment Methods ───────────────────────────────────
    def reset(self):
        """Start a new episode. Picks a question based on curriculum phase."""
        self._maybe_advance_phase()
        self.current_task = self._select_question()
        self.episode_count += 1

        logger.info(
            "Episode %d | Phase: %s | Complexity: %s | Prompt: %s",
            self.episode_count,
            CURRICULUM_PHASES[self.current_phase]["name"],
            self.current_task["complexity"],
            self.current_task["prompt"][:60]
        )

        return TokenEfficiencyObservation(
            prompt=self.current_task["prompt"],
            episode_token_limit=self.episode_token_limit
        )

    def step(self, action: TokenEfficiencyAction):
        """
        Process the model's response and return reward.

        Anti-hacking protections (Guide §8):
            - Budget clamping to [1, 200]
            - Empty/whitespace answer detection
            - Repetition guard (same word repeated)
            - Answer length sanity check
            - Step timeout
        """
        start_time = time.time()
        raw = action.raw_response

        # ─── Parse format ───────────────────────────────────────
        budget_match = re.search(r"<budget>(\d+)</budget>", raw)
        answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)

        if not budget_match or not answer_match:
            logger.warning("Episode %d | BAD FORMAT | reward=-1.0", self.episode_count)
            self.recent_rewards.append(-1.0)
            return {
                "observation": {"error": "bad format"},
                "reward": -1.0,
                "done": True,
                "info": {
                    "error": "model did not follow <budget>N</budget><answer>text</answer> format",
                    "episode": self.episode_count,
                    "phase": CURRICULUM_PHASES[self.current_phase]["name"],
                }
            }

        allocated_budget = int(budget_match.group(1))
        answer = answer_match.group(1).strip()

        # ─── Anti-Hacking: Budget clamping ──────────────────────
        original_budget = allocated_budget
        allocated_budget = max(MIN_BUDGET, min(MAX_BUDGET, allocated_budget))
        if original_budget != allocated_budget:
            logger.warning(
                "Episode %d | Budget clamped: %d → %d",
                self.episode_count, original_budget, allocated_budget
            )

        # ─── Anti-Hacking: Empty answer detection ───────────────
        if not answer or answer.isspace():
            logger.warning("Episode %d | EMPTY ANSWER | reward=-1.0", self.episode_count)
            self.recent_rewards.append(-1.0)
            return {
                "observation": {"error": "empty answer"},
                "reward": -1.0,
                "done": True,
                "info": {
                    "error": "answer was empty or whitespace only",
                    "episode": self.episode_count,
                }
            }

        # ─── Anti-Hacking: Repetition guard ─────────────────────
        words = answer.lower().split()
        if len(words) > 3:
            from collections import Counter
            word_counts = Counter(words)
            most_common_word, most_common_count = word_counts.most_common(1)[0]
            if most_common_count / len(words) > REPETITION_THRESHOLD:
                logger.warning(
                    "Episode %d | REPETITION DETECTED | '%s' repeated %d/%d times",
                    self.episode_count, most_common_word, most_common_count, len(words)
                )
                self.recent_rewards.append(-0.5)
                return {
                    "observation": {"error": "repetitive answer"},
                    "reward": -0.5,
                    "done": True,
                    "info": {
                        "error": f"repetitive answer: '{most_common_word}' repeated {most_common_count}/{len(words)} times",
                        "episode": self.episode_count,
                    }
                }

        # ─── Count tokens ──────────────────────────────────────
        tokens_used = len(tokenizer.encode(answer))

        # ─── Anti-Hacking: Answer length sanity ────────────────
        if tokens_used > MAX_ANSWER_TOKENS:
            logger.warning(
                "Episode %d | ANSWER TOO LONG | %d tokens (max %d)",
                self.episode_count, tokens_used, MAX_ANSWER_TOKENS
            )
            self.recent_rewards.append(-0.5)
            return {
                "observation": {"error": "answer too long"},
                "reward": -0.5,
                "done": True,
                "info": {
                    "error": f"answer used {tokens_used} tokens, max allowed is {MAX_ANSWER_TOKENS}",
                    "episode": self.episode_count,
                }
            }

        # ─── Timeout check ─────────────────────────────────────
        elapsed = time.time() - start_time
        if elapsed > STEP_TIMEOUT_SECONDS:
            logger.warning(
                "Episode %d | TIMEOUT | %.1fs > %ds",
                self.episode_count, elapsed, STEP_TIMEOUT_SECONDS
            )

        # ─── Score the answer (7-component) ────────────────────
        score_result = score_answer(
            prompt=self.current_task["prompt"],
            response=answer,
            allocated_budget=allocated_budget,
            tokens_used=tokens_used,
            complexity=self.current_task.get("complexity", "medium"),
            expected_keywords=self.current_task.get("expected_keywords", []),
        )

        reward = score_result["reward"]
        details = score_result["details"]

        # ─── Track reward for curriculum ───────────────────────
        self.recent_rewards.append(reward)
        # Keep only the last REWARD_WINDOW * 2 to avoid memory issues
        if len(self.recent_rewards) > REWARD_WINDOW * 2:
            self.recent_rewards = self.recent_rewards[-REWARD_WINDOW:]

        # ─── Monitoring log (Guide §15) ────────────────────────
        avg_recent = (
            sum(self.recent_rewards[-REWARD_WINDOW:]) /
            min(len(self.recent_rewards), REWARD_WINDOW)
        )

        logger.info(
            "Episode %d | reward=%.4f | avg_50=%.4f | "
            "correct=%.2f | effic=%.2f | budget_r=%.2f | "
            "redund=%.2f | self_a=%.2f | keyword=%.2f | format=%.2f | "
            "budget=%d | used=%d | complexity=%s",
            self.episode_count, reward, avg_recent,
            details.get("correctness", 0),
            details.get("efficiency", 0),
            details.get("budget_reasonableness", 0),
            details.get("redundancy", 0),
            details.get("self_assessment", 0),
            details.get("keyword_verification", 0),
            details.get("format_quality", 0),
            allocated_budget, tokens_used,
            self.current_task.get("complexity", "unknown")
        )

        # ─── Return result ─────────────────────────────────────
        return {
            "observation": {
                "answer": answer,
                "tokens_used": tokens_used,
                "allocated_budget": allocated_budget,
            },
            "reward": reward,
            "done": True,
            "info": {
                "episode": self.episode_count,
                "phase": CURRICULUM_PHASES[self.current_phase]["name"],
                "complexity": self.current_task.get("complexity", "unknown"),
                "tokens_used": tokens_used,
                "allocated_budget": allocated_budget,
                "avg_reward_50": round(avg_recent, 4),
                "reward_components": details,
            }
        }

    def state(self):
        """Return current environment state for debugging/logging."""
        avg_recent = 0.0
        if self.recent_rewards:
            window = self.recent_rewards[-REWARD_WINDOW:]
            avg_recent = sum(window) / len(window)

        return {
            "current_task": self.current_task,
            "episode_count": self.episode_count,
            "current_phase": CURRICULUM_PHASES[self.current_phase]["name"],
            "avg_reward_50": round(avg_recent, 4),
            "total_episodes": self.episode_count,
        }