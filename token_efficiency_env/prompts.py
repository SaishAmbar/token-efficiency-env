"""Prompt bank(s) used by the environment and the trainer.

Two shapes live side-by-side:

* ``STARTER_PROMPTS`` / ``PROMPT_BANK`` (the 24-prompt hand-curated list
  below) — deterministic, offline-safe, and what every test / demo /
  dashboard pins to. Kept for CI, the web dashboard, and the Phase 7
  smoke scaffold.

* ``load_full_prompt_bank()`` — returns a programmatic ~2.3k-prompt bank
  sourced from GSM8K + TriviaQA + ARC + OpenOrca (Stage 6.A of the
  Phase 8 plan in ``test_check_docs/VULNERABILITY_FIX_PLAN.md``). Not
  imported by default; hides the ``datasets`` dependency until a real
  training run asks for it. Call sites select between the two via the
  ``PROMPT_BANK_MODE`` env var:

    - ``starter`` (default) → ``STARTER_PROMPTS``
    - ``full``             → ``load_full_prompt_bank(seed=SPLIT_SEED)``

Schema of every entry (both banks):

    {
        "prompt": str,                                # question text
        "complexity": "easy" | "medium" | "hard",
        "expected_keywords": list[str | list[str]],   # §7 list-of-lists aliases OK
    }
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Sequence, TypedDict, Union

logger = logging.getLogger(__name__)

KeywordEntry = Union[str, List[str]]


class PromptEntry(TypedDict, total=False):
    prompt: str
    complexity: str
    expected_keywords: List[KeywordEntry]


# ─── 1. Starter bank — the canonical 24 hand-curated prompts ────────────
STARTER_PROMPTS: List[PromptEntry] = [
    # ──────────────────────────────────────────────────────────────
    # EASY — short answer expected (1–5 words)
    # expected_keywords: used by keyword verification scorer
    # ──────────────────────────────────────────────────────────────
    {
        "prompt": "What is 15% of 200?",
        "complexity": "easy",
        "expected_keywords": [["30", "thirty"]]
    },
    {
        "prompt": "What is the capital of France?",
        "complexity": "easy",
        "expected_keywords": ["paris"]
    },
    {
        "prompt": "What does CPU stand for?",
        "complexity": "easy",
        "expected_keywords": ["central", "processing", "unit"]
    },
    {
        "prompt": "What is 2 to the power of 8?",
        "complexity": "easy",
        "expected_keywords": [["256", "two hundred fifty six"]]
    },
    {
        "prompt": "What color do you get mixing red and blue?",
        "complexity": "easy",
        "expected_keywords": ["purple"]
    },
    {
        "prompt": "How many sides does a hexagon have?",
        "complexity": "easy",
        "expected_keywords": [["6", "six"]]
    },
    {
        "prompt": "What is the boiling point of water in Celsius?",
        "complexity": "easy",
        "expected_keywords": [["100", "one hundred"]]
    },
    {
        "prompt": "Who wrote Romeo and Juliet?",
        "complexity": "easy",
        "expected_keywords": ["shakespeare"]
    },

    # ──────────────────────────────────────────────────────────────
    # MEDIUM — a few sentences expected
    # ──────────────────────────────────────────────────────────────
    {
        "prompt": "Explain what gravity is.",
        "complexity": "medium",
        "expected_keywords": ["force", "attract", "mass"]
    },
    {
        "prompt": "What is the difference between RAM and ROM?",
        "complexity": "medium",
        "expected_keywords": ["volatile", "memory", "read"]
    },
    {
        "prompt": "How does a vaccine work?",
        "complexity": "medium",
        "expected_keywords": ["immune", "antibod"]
    },
    {
        "prompt": "What causes seasons on Earth?",
        "complexity": "medium",
        "expected_keywords": ["tilt", "axis", "sun"]
    },
    {
        "prompt": "Explain what inflation means.",
        "complexity": "medium",
        "expected_keywords": ["price", "purchas", "money"]
    },
    {
        "prompt": "What is the difference between speed and velocity?",
        "complexity": "medium",
        "expected_keywords": ["direction", "scalar", "vector"]
    },
    {
        "prompt": "How does the internet work in simple terms?",
        "complexity": "medium",
        "expected_keywords": ["network", "data", "server"]
    },
    {
        "prompt": "What is photosynthesis?",
        "complexity": "medium",
        # NB: prefix-stem matching, so "plant" catches "plants",
        # "sunlight" matches itself, "oxygen" appears in essentially every
        # natural answer. Avoids the old "light"/"energy" trap where
        # "sunlight" couldn't satisfy "light" (no word-start boundary)
        # and "create their own food" answers never said "energy".
        "expected_keywords": ["plant", "sunlight", "oxygen"]
    },

    # ──────────────────────────────────────────────────────────────
    # HARD — detailed answer expected
    # ──────────────────────────────────────────────────────────────
    {
        "prompt": "Explain how transformers work in machine learning.",
        "complexity": "hard",
        "expected_keywords": ["attention", "token", "layer"]
    },
    {
        "prompt": "What are the causes and effects of climate change?",
        "complexity": "hard",
        "expected_keywords": ["greenhouse", "carbon", "temperature"]
    },
    {
        "prompt": "Explain the difference between supervised and unsupervised learning.",
        "complexity": "hard",
        "expected_keywords": ["label", "cluster", "train"]
    },
    {
        "prompt": "How does the immune system fight a virus?",
        "complexity": "hard",
        "expected_keywords": ["antibod", "cell", "pathogen"]
    },
    {
        "prompt": "Explain what quantum entanglement is.",
        "complexity": "hard",
        "expected_keywords": ["particle", "state", "measur"]
    },
    {
        "prompt": "What is the significance of the Turing Test?",
        "complexity": "hard",
        "expected_keywords": ["machine", "intelligen", "human"]
    },
    {
        "prompt": "How do neural networks learn from data?",
        "complexity": "hard",
        "expected_keywords": ["weight", "gradient", "backpropagat"]
    },
    {
        "prompt": "Explain the theory of relativity in simple terms.",
        "complexity": "hard",
        "expected_keywords": ["einstein", "space", "time"]
    },
]

# Back-compat alias. Downstream (env, trainer, tests, dashboard) still
# reads ``PROMPT_BANK`` — don't break that contract by removing it.
PROMPT_BANK: List[PromptEntry] = STARTER_PROMPTS


# ─── 2. Programmatic full bank (Phase 8, Stage 6.A) ─────────────────────
# Lives in a separate, importable-on-demand module so the happy-path
# ``import token_efficiency_env`` never pulls in ``datasets`` / network.
def load_full_prompt_bank(
    seed: int = 42,
    max_prompts: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[PromptEntry]:
    """Load the programmatic ~2.3k-prompt bank.

    The actual dataset fetching (GSM8K + TriviaQA + ARC + OpenOrca) is
    implemented in ``token_efficiency_env.prompt_bank_loader`` and
    imported lazily here so that importing this module never requires
    the ``datasets`` package to be installed.

    Parameters:
        seed:        Deterministic shuffle/sample seed.
        max_prompts: Optional cap for quick smoke runs (default = all ~2.3k).
        cache_dir:   Where HuggingFace caches raw dataset files (default =
                     ``HF_HOME`` env var).

    Raises:
        ImportError:  If ``datasets`` / ``num2words`` aren't installed.
                      Install ``pip install -r requirements-train.txt`` or
                      fall back to ``PROMPT_BANK_MODE=starter``.
        RuntimeError: On network failure — dataset downloads are network-
                      bound; callers should either retry or switch modes.
    """
    # Lazy import — this is the whole point of the split between this
    # file (stays offline) and ``prompt_bank_loader.py`` (pulls network).
    from .prompt_bank_loader import load_programmatic_bank  # type: ignore[import-not-found]

    return load_programmatic_bank(
        seed=seed, max_prompts=max_prompts, cache_dir=cache_dir
    )


# ─── 3. Dispatch ────────────────────────────────────────────────────────
_FULL_BANK_CACHE: Optional[List[PromptEntry]] = None


def get_prompt_bank(
    mode: Optional[str] = None,
    *,
    loader: Optional[Callable[[], List[PromptEntry]]] = None,
) -> Sequence[PromptEntry]:
    """Return the active prompt bank, selected by mode / env var.

    Resolution order:
      1. Explicit ``mode`` argument (``"starter"`` | ``"full"``).
      2. ``PROMPT_BANK_MODE`` env var.
      3. Default: ``"starter"`` — so pure-import users never pay a
         network cost.

    The ``loader`` argument is a seam for tests — pass a callable that
    returns a list of PromptEntries and it'll be used instead of
    ``load_full_prompt_bank``. Useful for injecting a stub bank without
    touching the environment variable.
    """
    if mode is None:
        mode = os.environ.get("PROMPT_BANK_MODE", "starter").lower().strip()
    mode = mode.lower().strip()

    if mode in ("starter", "mini", ""):
        return STARTER_PROMPTS

    if mode == "full":
        global _FULL_BANK_CACHE
        if _FULL_BANK_CACHE is None:
            try:
                _FULL_BANK_CACHE = (loader or load_full_prompt_bank)()
            except ImportError as exc:
                logger.warning(
                    "Full prompt bank requested but its dependencies are "
                    "missing (%s); falling back to STARTER_PROMPTS. Install "
                    "requirements-train.txt to enable the full bank.",
                    exc,
                )
                return STARTER_PROMPTS
        return _FULL_BANK_CACHE

    raise ValueError(
        f"Unknown PROMPT_BANK_MODE={mode!r}; "
        "expected 'starter' or 'full'."
    )


def reset_prompt_bank_cache() -> None:
    """Clear the cached full bank. Intended for tests."""
    global _FULL_BANK_CACHE
    _FULL_BANK_CACHE = None


__all__ = [
    "STARTER_PROMPTS",
    "PROMPT_BANK",
    "PromptEntry",
    "KeywordEntry",
    "load_full_prompt_bank",
    "get_prompt_bank",
    "reset_prompt_bank_cache",
]
