"""Bridge ``TokenEfficiencyEnv`` into TRL's ``reward_funcs`` API.

TRL's GRPOTrainer calls a reward function with the signature::

    reward_func(prompts, completions, **kwargs) -> list[float]

where ``prompts`` and ``completions`` are aligned lists of length
``per_device_train_batch_size * num_generations``. We need to:

    1. For each (prompt, completion) pair, find the matching task in
       ``PROMPT_BANK`` (so we can recover the ``expected_keywords`` and
       ``complexity`` the env needs for scoring).
    2. Run that pair through the env's ``step()``, which gives us all the
       cliffs (bad_format, parrot, repetition, too_long, empty), the
       6-component scoring, and the ``-1.0..~0.97`` reward we want.
    3. Return the floats in the same order, plus log every component to a
       ``RewardLog`` so the notebook can plot them post-training.

Two backends:

    * ``InProcessRewardAdapter`` (default) — instantiates one
      ``TokenEfficiencyEnvironment`` per "rollout slot", drives it directly
      in the trainer's process. No servers, no ports, no async. This is
      what you want 99% of the time.

    * ``WSRewardAdapter`` — talks to a real ``ServerPool`` over the
      WebSocket client. Same logical API; useful for stress-testing the
      deployed server path.

Both expose the same ``__call__`` so the notebook can swap backends with
one config flag.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from token_efficiency_env.models import TokenEfficiencyAction
from token_efficiency_env.prompts import get_prompt_bank
from token_efficiency_env.server.token_efficiency_env_environment import (
    TokenEfficiencyEnvironment,
)

logger = logging.getLogger("token_efficiency_env.reward_adapter")

# ─── Prompt → task lookup ──────────────────────────────────────────────
# We build the lookup dict lazily from get_prompt_bank() rather than
# from the module-level PROMPT_BANK constant. This matters when the
# caller sets PROMPT_BANK_MODE=full (the 2.3k-prompt bank): at import
# time the lazy loader hasn't run yet, so a module-level dict built from
# PROMPT_BANK would only contain the 24 starter prompts, causing every
# full-bank prompt to be logged as "unknown" and scored 0.0.
_PROMPT_TO_TASK_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_PROMPT_TO_TASK_CACHE_LOCK = threading.Lock()


def _build_prompt_lookup() -> Dict[str, Dict[str, Any]]:
    """Build (or return cached) prompt→task lookup for the active bank."""
    global _PROMPT_TO_TASK_CACHE
    with _PROMPT_TO_TASK_CACHE_LOCK:
        if _PROMPT_TO_TASK_CACHE is None:
            _PROMPT_TO_TASK_CACHE = {p["prompt"]: p for p in get_prompt_bank()}
        return _PROMPT_TO_TASK_CACHE


def _invalidate_prompt_lookup() -> None:
    """Call this if the bank mode changes at runtime (e.g. in tests)."""
    global _PROMPT_TO_TASK_CACHE
    with _PROMPT_TO_TASK_CACHE_LOCK:
        _PROMPT_TO_TASK_CACHE = None


def _task_for_prompt(prompt: str) -> Optional[Dict[str, Any]]:
    """Return the active bank entry whose ``prompt`` matches exactly.

    Respects ``PROMPT_BANK_MODE`` (starter vs full). Returns None if
    there's no match (caller decides whether that's fatal or gets 0.0).
    """
    return _build_prompt_lookup().get(prompt)


# ─── Per-step log row ──────────────────────────────────────────────────
@dataclass
class RewardLog:
    """Append-only buffer of (prompt, completion, reward, components) tuples.

    The trainer is single-threaded from TRL's perspective but the in-process
    adapter pools envs under a lock, so the underlying list is guarded by
    its own ``threading.Lock`` to make ``.to_records()`` safe to call from
    notebook cells while training is running.
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def append(self, **row: Any) -> None:
        with self._lock:
            self.rows.append(row)

    def to_records(self) -> List[Dict[str, Any]]:
        """Snapshot a copy of the rows. Safe to call mid-training."""
        with self._lock:
            return list(self.rows)

    def __len__(self) -> int:
        with self._lock:
            return len(self.rows)


# ─── Adapter A: in-process ─────────────────────────────────────────────
class InProcessRewardAdapter:
    """Default reward backend. One env instance per rollout slot.

    A "rollout slot" is one of the ``num_generations`` parallel completions
    GRPO samples for each prompt. Each slot gets its own
    ``TokenEfficiencyEnvironment`` so curriculum/recent_rewards logging
    stays per-slot and doesn't interfere across rollouts.

    Curriculum is owned by the trainer during training: prompts come
    from a stratified split (``train_prompts()``), so we override the
    env's ``current_task`` before each ``step()`` and never rely on
    ``_select_question``. Each env is put into ``_trainer_driven_mode``
    on construction so ``step()`` also skips appending rewards to
    ``recent_rewards`` — otherwise the server-side ``avg_reward_50``
    stat would fill with numbers from prompts the env's own sampler
    never picked, and any dashboard reading ``obs.avg_reward_50`` or
    ``obs.phase`` would be reading noise. In trainer-driven mode those
    two fields are intentionally pinned (``avg_reward_50=0.0``,
    ``phase="foundation"``); the authoritative running mean lives in
    ``RewardLog`` instead.
    """

    def __init__(self, num_slots: int = 1, log: Optional[RewardLog] = None) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be >=1, got {num_slots}")
        self.num_slots = num_slots
        # Explicit `is None` check: an empty RewardLog is falsy via __len__,
        # so `log or RewardLog()` would silently swap the caller's instance
        # for a new one and every appended row would land in a log nobody
        # holds a reference to. Took one test failure to find this.
        self.log = RewardLog() if log is None else log
        self._envs: List[TokenEfficiencyEnvironment] = []
        for _ in range(num_slots):
            env = TokenEfficiencyEnvironment()
            env._trainer_driven_mode = True
            self._envs.append(env)
        # Each slot is a single-threaded conversation with one env, but TRL
        # may call us from a worker pool; serialise per-slot access.
        self._slot_locks = [threading.Lock() for _ in range(num_slots)]
        # Round-robin slot assignment for batches that don't divide evenly.
        self._next_slot = 0
        self._next_slot_lock = threading.Lock()

    def _claim_slot(self) -> int:
        with self._next_slot_lock:
            slot = self._next_slot
            self._next_slot = (self._next_slot + 1) % self.num_slots
            return slot

    def _score_one(self, prompt: str, completion: str) -> float:
        """Run a single (prompt, completion) pair through the env."""
        task = _task_for_prompt(prompt)
        if task is None:
            # Defensive: trainer fed us an unknown prompt. Log loudly and
            # return 0.0 rather than crash the run. Should never happen
            # because the trainer dataset is built from PROMPT_BANK.
            logger.warning("Unknown prompt fed to reward adapter: %r", prompt[:80])
            self.log.append(
                prompt=prompt, completion=completion, reward=0.0,
                components={}, error="unknown_prompt",
                tokens_used=0, allocated_budget=0, complexity="", slot=-1,
            )
            return 0.0

        slot = self._claim_slot()
        env = self._envs[slot]
        with self._slot_locks[slot]:
            # Bypass _select_question — trainer owns prompt selection.
            # The env's step() needs current_task set so it can read
            # complexity + expected_keywords; we don't go through reset().
            env.current_task = task
            env.episode_count += 1  # so logs are still useful
            obs = env.step(TokenEfficiencyAction(raw_response=completion))

        reward = float(obs.reward) if obs.reward is not None else 0.0
        self.log.append(
            prompt=prompt,
            completion=completion,
            reward=reward,
            components=dict(obs.reward_components),
            tokens_used=obs.tokens_used,
            allocated_budget=obs.allocated_budget,
            complexity=obs.complexity,
            error=obs.error or "",
            slot=slot,
        )
        return reward

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[Any],
        **kwargs: Any,
    ) -> List[float]:
        """TRL-shaped reward function.

        ``completions`` may arrive as either:
          * a list of plain strings (TRL ``conversational=False``), or
          * a list of ``[{"role": "assistant", "content": "..."}]`` dicts
            (TRL ``conversational=True``).
        We normalise to strings before scoring.
        """
        if len(prompts) != len(completions):
            raise ValueError(
                f"reward_adapter: len(prompts)={len(prompts)} != "
                f"len(completions)={len(completions)}"
            )

        rewards: List[float] = []
        for p, c in zip(prompts, completions):
            text = _completion_text(c)
            rewards.append(self._score_one(p, text))
        return rewards

    def close(self) -> None:
        """No-op; here so ``WSRewardAdapter`` and this can share the API."""
        pass


# ─── Adapter B: WebSocket pool ─────────────────────────────────────────
class WSRewardAdapter:
    """Reward backend that talks to a running ``ServerPool``.

    Heavier than the in-process adapter (one HTTP+WS round-trip per
    completion). Use this when you want to validate the HF Space code path
    or stress-test the server, NOT for routine training.
    """

    def __init__(
        self,
        urls: Sequence[str],
        log: Optional[RewardLog] = None,
    ) -> None:
        if not urls:
            raise ValueError("WSRewardAdapter needs at least one server URL")
        self.urls = list(urls)
        self.log = RewardLog() if log is None else log  # see InProcessRewardAdapter for the reason
        # Lazy import — keeps openenv.core out of the import path for the
        # in-process happy path that doesn't need it.
        from token_efficiency_env.client import TokenEfficiencyEnv

        self._client_cls = TokenEfficiencyEnv
        self._clients: List[Any] = []
        self._client_locks: List[threading.Lock] = []
        self._next_slot = 0
        self._next_slot_lock = threading.Lock()
        self._open_clients()

    def _open_clients(self) -> None:
        for url in self.urls:
            client = self._client_cls(base_url=url).sync()
            client.__enter__()  # sync context, kept open for the run
            self._clients.append(client)
            self._client_locks.append(threading.Lock())

    def _claim_slot(self) -> int:
        with self._next_slot_lock:
            slot = self._next_slot
            self._next_slot = (self._next_slot + 1) % len(self._clients)
            return slot

    def _score_one(self, prompt: str, completion: str) -> float:
        task = _task_for_prompt(prompt)
        if task is None:
            logger.warning("Unknown prompt fed to WS reward adapter: %r", prompt[:80])
            self.log.append(
                prompt=prompt, completion=completion, reward=0.0,
                components={}, error="unknown_prompt",
                tokens_used=0, allocated_budget=0, complexity="", slot=-1,
            )
            return 0.0

        slot = self._claim_slot()
        client = self._clients[slot]
        with self._client_locks[slot]:
            # We can't force the server to pick `task`, so we burn a reset
            # to align curriculum bookkeeping but score against the
            # trainer's prompt manually. The actual scoring still happens
            # server-side via the step() call below.
            #
            # NOTE: this means the SERVER's curriculum drifts independently
            # of our trainer-side prompt sampling. That's fine for stress
            # testing but is the reason in_process is the default.
            client.reset()
            result = client.step(TokenEfficiencyAction(raw_response=completion))

        obs = result.observation
        reward = float(result.reward) if result.reward is not None else 0.0
        self.log.append(
            prompt=prompt,
            completion=completion,
            reward=reward,
            components=dict(obs.reward_components),
            tokens_used=obs.tokens_used,
            allocated_budget=obs.allocated_budget,
            complexity=obs.complexity,
            error=obs.error or "",
            slot=slot,
            backend="ws",
        )
        return reward

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[Any],
        **kwargs: Any,
    ) -> List[float]:
        return [
            self._score_one(p, _completion_text(c))
            for p, c in zip(prompts, completions)
        ]

    def close(self) -> None:
        for client in self._clients:
            try:
                client.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                logger.warning("Error closing WS client: %s", exc)
        self._clients.clear()


# ─── Helpers ───────────────────────────────────────────────────────────
def _completion_text(completion: Any) -> str:
    """Normalise TRL completions to a plain string.

    TRL passes either ``str`` (non-chat) or
    ``[{"role": "assistant", "content": "..."}]`` (chat). We accept both
    plus a few fallback shapes so the adapter doesn't fight the trainer's
    config.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])
    return str(completion)


def build_reward_func(
    config: "TrainingConfig",  # noqa: F821 — forward ref to avoid cycle
    log: Optional[RewardLog] = None,
):
    """Factory: returns the reward callable that matches ``config.reward_backend``.

    Returns a tuple ``(reward_func, adapter)`` so the caller can ``close()``
    the adapter at the end of training (matters for ``ws`` backend; no-op
    for in-process).
    """
    log = RewardLog() if log is None else log
    if config.reward_backend == "in_process":
        adapter = InProcessRewardAdapter(num_slots=config.num_generations, log=log)
        return adapter, adapter

    if config.reward_backend == "ws":
        # The notebook is responsible for spinning up the ServerPool on
        # ``config.base_port .. config.base_port + num_env_servers``;
        # here we just point at it.
        urls = [
            f"http://localhost:{config.base_port + i}"
            for i in range(config.num_env_servers)
        ]
        adapter = WSRewardAdapter(urls=urls, log=log)
        return adapter, adapter

    raise ValueError(
        f"Unknown reward_backend={config.reward_backend!r}; "
        f"expected 'in_process' or 'ws'."
    )
