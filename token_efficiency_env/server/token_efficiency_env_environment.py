"""
TokenEfficiencyEnvironment — OpenEnv-compatible HTTP environment.

Wraps the real environment logic (curriculum, anti-hacking, 6-component
scoring) and exposes it via the ``openenv.core.env_server`` interface so the
HTTP/WebSocket server in ``server/app.py`` can host it.

Each WebSocket session gets its own instance of this class (via the factory
mode in ``app.py``), so curriculum state is per-trainer and never bleeds
across sessions.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from collections import Counter, deque
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Imports work both when the package is installed under
# ``token_efficiency_env`` and when uvicorn loads ``server.app:app`` from
# /app/env (where ``server`` is a top-level package).
try:
    from ..models import TokenEfficiencyAction, TokenEfficiencyObservation
    from ..prompts import PROMPT_BANK
    from ..scorer import score_answer
except ImportError:  # pragma: no cover — exercised only at container start
    from models import TokenEfficiencyAction, TokenEfficiencyObservation  # type: ignore[no-redef]
    from prompts import PROMPT_BANK  # type: ignore[no-redef]
    from scorer import score_answer  # type: ignore[no-redef]


# ─── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("TOKEN_EFFICIENCY_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TokenEfficiencyEnv")


# ─── Constants ──────────────────────────────────────────────────────────
MAX_TOKEN_LIMIT = 200
MIN_BUDGET = 1
MAX_BUDGET = MAX_TOKEN_LIMIT
MAX_ANSWER_TOKENS = 500
REPETITION_THRESHOLD = 0.6
REWARD_WINDOW = 50
MIN_ANSWER_CHARS = 2  # Phase 3: catches ".", "a", " " hacks that pass `if not answer`


def _normalize_for_parrot(text: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace.

    Used by the parrot guard to compare prompt and answer fairly. We strip
    punctuation so ``"What is the capital of France?"`` and the model's echoed
    ``"what is the capital of france"`` register as identical.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# ─── Curriculum (advance_threshold = None ⇒ final phase) ────────────────
CURRICULUM_PHASES = [
    {
        "name": "Phase 1: Easy Only",
        "weights": {"easy": 1.0, "medium": 0.0, "hard": 0.0},
        "advance_threshold": 0.4,
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
        "advance_threshold": None,
    },
]


# ─── Lazy tokenizer (loaded on first use, then cached) ──────────────────
_TOKENIZER = None


def _get_tokenizer():
    """Load the Qwen2.5-3B tokenizer on first use.

    Loading at import time forces a network download for any process that
    imports this module (tests, scorer-only usage, etc.) and makes container
    start-up race the HF Space health check, so we defer it.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        cache_dir = os.environ.get("HF_HOME", None)
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            cache_dir=cache_dir,
        )
    return _TOKENIZER


def _count_tokens(text: str) -> int:
    """Exact token count using the Qwen2.5-3B tokenizer."""
    return len(_get_tokenizer().encode(text))


# ─── Environment ────────────────────────────────────────────────────────
class TokenEfficiencyEnvironment(Environment):
    """OpenEnv environment for training LLMs to be token-efficient.

    The agent receives a question via ``reset()``. It must respond with
    ``<budget>N</budget><answer>text</answer>``. ``step()`` parses the
    response, runs anti-hacking guards, and scores the answer with the
    6-component scorer.

    State (per session):
        - episode_count, current_phase, recent_rewards (rolling, maxlen=50)
        - current_task (the question selected by the most recent reset)

    Concurrency:
        SUPPORTS_CONCURRENT_SESSIONS = True. Each WebSocket session gets its
        own instance via the factory in ``server/app.py``, so curriculum
        state is isolated per trainer.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.episode_count = 0
        self.current_phase = 0
        self.recent_rewards: deque[float] = deque(maxlen=REWARD_WINDOW)
        self.current_task: Optional[dict] = None

        self._prompt_pools = {
            "easy": [p for p in PROMPT_BANK if p["complexity"] == "easy"],
            "medium": [p for p in PROMPT_BANK if p["complexity"] == "medium"],
            "hard": [p for p in PROMPT_BANK if p["complexity"] == "hard"],
        }

        logger.info(
            "Environment initialised | episode_id=%s | phase=%s",
            self._state.episode_id,
            CURRICULUM_PHASES[self.current_phase]["name"],
        )

    # ─── Curriculum helpers ────────────────────────────────────────
    def _select_question(self) -> dict:
        weights = CURRICULUM_PHASES[self.current_phase]["weights"]
        complexities = [c for c, w in weights.items() if w > 0]
        if not complexities:
            return random.choice(PROMPT_BANK)
        chosen_complexity = random.choices(
            complexities, weights=[weights[c] for c in complexities], k=1
        )[0]
        return random.choice(self._prompt_pools[chosen_complexity])

    def _maybe_advance_phase(self) -> None:
        if self.current_phase >= len(CURRICULUM_PHASES) - 1:
            return
        if len(self.recent_rewards) < REWARD_WINDOW:
            return

        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
        threshold = CURRICULUM_PHASES[self.current_phase]["advance_threshold"]
        if threshold is not None and avg_reward >= threshold:
            self.current_phase += 1
            logger.info(
                "Curriculum advance | now=%s | avg_reward=%.4f",
                CURRICULUM_PHASES[self.current_phase]["name"],
                avg_reward,
            )

    def _avg_reward(self) -> float:
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    def _phase_name(self) -> str:
        return CURRICULUM_PHASES[self.current_phase]["name"]

    # ─── Environment interface ─────────────────────────────────────
    def reset(  # type: ignore[override]
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TokenEfficiencyObservation:
        if seed is not None:
            random.seed(seed)

        self._maybe_advance_phase()
        self.current_task = self._select_question()
        self.episode_count += 1

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        logger.info(
            "Reset | episode=%d | phase=%s | complexity=%s | prompt=%s",
            self.episode_count,
            self._phase_name(),
            self.current_task["complexity"],
            self.current_task["prompt"][:60],
        )

        return TokenEfficiencyObservation(
            prompt=self.current_task["prompt"],
            episode_token_limit=MAX_TOKEN_LIMIT,
            complexity=self.current_task["complexity"],
            phase=self._phase_name(),
            episode=self.episode_count,
            avg_reward_50=round(self._avg_reward(), 4),
            done=False,
            reward=None,
        )

    def step(  # type: ignore[override]
        self,
        action: TokenEfficiencyAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TokenEfficiencyObservation:
        if self.current_task is None:
            # Defensive: trainer called step() before reset(). Auto-reset.
            self.reset()

        self._state.step_count += 1
        start_time = time.time()
        raw = action.raw_response

        # ─── Parse format ─────────────────────────────────────────
        budget_match = re.search(r"<budget>(\d+)</budget>", raw)
        answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)

        if not budget_match or not answer_match:
            return self._fail(
                error="bad_format",
                reward=-1.0,
                log_msg="BAD FORMAT — missing <budget> or <answer>",
            )

        allocated_budget = int(budget_match.group(1))
        answer = answer_match.group(1).strip()

        # ─── Budget clamp ─────────────────────────────────────────
        original_budget = allocated_budget
        allocated_budget = max(MIN_BUDGET, min(MAX_BUDGET, allocated_budget))
        if original_budget != allocated_budget:
            logger.warning(
                "Episode %d | Budget clamped: %d → %d",
                self.episode_count,
                original_budget,
                allocated_budget,
            )

        # ─── Empty / near-empty answer ────────────────────────────
        # MIN_ANSWER_CHARS=2 catches ".", "a", " ", "" and similar
        # one-character cop-outs that the old `if not answer` missed.
        if len(answer) < MIN_ANSWER_CHARS:
            return self._fail(
                error="empty",
                reward=-1.0,
                log_msg=f"EMPTY/NEAR-EMPTY ANSWER (len={len(answer)})",
                answer=answer,
                allocated_budget=allocated_budget,
            )

        # ─── Parrot guard ─────────────────────────────────────────
        # Hard-fail if the answer literally contains the prompt as a
        # substring (after normalising case and punctuation). This catches
        # the "echo the question back" hack that the LLM judge would
        # otherwise score 0.0 on but that the other components might
        # accidentally reward.
        prompt_norm = _normalize_for_parrot(self.current_task["prompt"])
        answer_norm = _normalize_for_parrot(answer)
        if prompt_norm and prompt_norm in answer_norm:
            return self._fail(
                error="parrot",
                reward=-0.5,
                log_msg="PARROT — answer contains prompt verbatim",
                answer=answer,
                allocated_budget=allocated_budget,
            )

        # ─── Repetition guard ─────────────────────────────────────
        words = answer.lower().split()
        if len(words) > 3:
            top_word, top_count = Counter(words).most_common(1)[0]
            if top_count / len(words) > REPETITION_THRESHOLD:
                return self._fail(
                    error="repetition",
                    reward=-0.5,
                    log_msg=f"REPETITION: '{top_word}' = {top_count}/{len(words)}",
                    answer=answer,
                    allocated_budget=allocated_budget,
                )

        # ─── Token count + length sanity ──────────────────────────
        try:
            tokens_used = _count_tokens(answer)
        except Exception as exc:  # pragma: no cover
            logger.exception("Tokenizer failed; falling back to word*1.3 estimate: %s", exc)
            tokens_used = max(1, int(len(words) * 1.3))

        if tokens_used > MAX_ANSWER_TOKENS:
            return self._fail(
                error="too_long",
                reward=-0.5,
                log_msg=f"TOO LONG: {tokens_used} > {MAX_ANSWER_TOKENS}",
                answer=answer,
                allocated_budget=allocated_budget,
                tokens_used=tokens_used,
            )

        # ─── Score (6 components) ─────────────────────────────────
        score_result = score_answer(
            prompt=self.current_task["prompt"],
            response=answer,
            allocated_budget=allocated_budget,
            tokens_used=tokens_used,
            complexity=self.current_task.get("complexity", "medium"),
            expected_keywords=self.current_task.get("expected_keywords", []),
        )

        reward = float(score_result["reward"])
        details = {k: float(v) for k, v in score_result["details"].items()}

        elapsed = time.time() - start_time
        self.recent_rewards.append(reward)

        logger.info(
            "Episode %d | reward=%.4f | avg=%.4f | budget=%d | used=%d "
            "| complexity=%s | %.2fs",
            self.episode_count,
            reward,
            self._avg_reward(),
            allocated_budget,
            tokens_used,
            self.current_task.get("complexity", "?"),
            elapsed,
        )

        return TokenEfficiencyObservation(
            prompt=self.current_task["prompt"],
            episode_token_limit=MAX_TOKEN_LIMIT,
            answer=answer,
            allocated_budget=allocated_budget,
            tokens_used=tokens_used,
            complexity=self.current_task.get("complexity", ""),
            phase=self._phase_name(),
            episode=self.episode_count,
            avg_reward_50=round(self._avg_reward(), 4),
            reward_components=details,
            error="",
            reward=reward,
            done=True,
        )

    @property
    def state(self) -> State:
        return self._state

    # ─── Helpers ──────────────────────────────────────────────────
    def _fail(
        self,
        *,
        error: str,
        reward: float,
        log_msg: str,
        answer: str = "",
        allocated_budget: int = 0,
        tokens_used: int = 0,
    ) -> TokenEfficiencyObservation:
        """Build a terminal failure observation and update curriculum state."""
        self.recent_rewards.append(reward)
        logger.warning(
            "Episode %d | %s | reward=%.2f", self.episode_count, log_msg, reward
        )
        prompt = self.current_task["prompt"] if self.current_task else ""
        complexity = self.current_task.get("complexity", "") if self.current_task else ""
        return TokenEfficiencyObservation(
            prompt=prompt,
            episode_token_limit=MAX_TOKEN_LIMIT,
            answer=answer,
            allocated_budget=allocated_budget,
            tokens_used=tokens_used,
            complexity=complexity,
            phase=self._phase_name(),
            episode=self.episode_count,
            avg_reward_50=round(self._avg_reward(), 4),
            reward_components={},
            error=error,
            reward=reward,
            done=True,
        )
