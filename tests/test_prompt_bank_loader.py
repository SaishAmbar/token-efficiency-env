"""Tests for the Phase 8 prompt-bank loader scaffolding.

Two concerns:

1. ``prompts.get_prompt_bank(mode=...)`` dispatches correctly between the
   offline starter bank and a "full" bank loader, including the
   fall-back path when the full bank's dependencies aren't installed.

2. ``prompt_bank_loader`` helpers (``_extract_keywords``,
   ``_bucket_by_expected_length``, ``_build_entry``,
   ``_add_numeric_aliases``) produce entries in the exact schema the rest
   of the pipeline expects — so swapping the starter bank for the ~2.3k
   programmatic bank in a training run doesn't silently break the
   scorer.

The heavyweight integration test that actually pulls GSM8K / TriviaQA /
ARC / OpenOrca over the network is marked ``slow`` and disabled by
default. Run it explicitly with ``pytest -m slow`` when you've installed
``requirements-train.txt`` and have network access.
"""

from __future__ import annotations

import importlib
import os

import pytest

from token_efficiency_env import prompts as prompts_mod
from token_efficiency_env.prompts import (
    PROMPT_BANK,
    STARTER_PROMPTS,
    get_prompt_bank,
    reset_prompt_bank_cache,
)


# ─── 1. Dispatch / env-var contract ────────────────────────────────────
def test_starter_is_the_default_and_matches_PROMPT_BANK():
    """Default call returns the 24-prompt starter bank (same object)."""
    bank = get_prompt_bank()
    assert bank is STARTER_PROMPTS
    assert STARTER_PROMPTS is PROMPT_BANK  # back-compat alias


def test_explicit_starter_mode():
    assert get_prompt_bank(mode="starter") is STARTER_PROMPTS
    assert get_prompt_bank(mode="STARTER") is STARTER_PROMPTS  # case-insensitive


def test_env_var_selects_mode(monkeypatch):
    """``PROMPT_BANK_MODE=starter`` env var is honoured."""
    monkeypatch.setenv("PROMPT_BANK_MODE", "starter")
    reset_prompt_bank_cache()
    assert get_prompt_bank() is STARTER_PROMPTS


def test_full_mode_uses_injected_loader():
    """``loader=`` kwarg lets tests bypass network without installing
    ``datasets`` or monkeypatching sys.modules."""
    stub_entries = [
        {"prompt": "Q1?", "complexity": "easy", "expected_keywords": ["a"]},
        {"prompt": "Q2?", "complexity": "medium", "expected_keywords": [["1", "one"]]},
        {"prompt": "Q3?", "complexity": "hard", "expected_keywords": ["b", "c"]},
    ]
    reset_prompt_bank_cache()
    bank = get_prompt_bank(mode="full", loader=lambda: stub_entries)
    assert bank == stub_entries
    # Second call without passing loader again must hit the cache.
    cached = get_prompt_bank(mode="full")
    assert cached is bank
    reset_prompt_bank_cache()


def test_full_mode_falls_back_to_starter_when_dependencies_missing():
    """If the full loader raises ImportError, we must not crash the
    training run — we fall back to the starter bank with a warning."""
    def broken_loader():
        raise ImportError("fake: `datasets` not installed")

    reset_prompt_bank_cache()
    bank = get_prompt_bank(mode="full", loader=broken_loader)
    assert bank is STARTER_PROMPTS
    reset_prompt_bank_cache()


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown PROMPT_BANK_MODE"):
        get_prompt_bank(mode="bogus")


def test_reset_cache_clears_full_bank():
    """Regression guard on the cache: after ``reset_prompt_bank_cache()``
    the next full-mode call must rebuild from the loader."""
    calls = {"n": 0}

    def counting_loader():
        calls["n"] += 1
        return [{"prompt": "X", "complexity": "easy", "expected_keywords": ["x"]}]

    reset_prompt_bank_cache()
    get_prompt_bank(mode="full", loader=counting_loader)
    get_prompt_bank(mode="full", loader=counting_loader)  # cached
    assert calls["n"] == 1

    reset_prompt_bank_cache()
    get_prompt_bank(mode="full", loader=counting_loader)
    assert calls["n"] == 2


# ─── 2. Programmatic loader helpers — schema + determinism ─────────────
# Import the loader module lazily so the ``datasets`` ImportError path
# still gets tested on hosts that don't have it (the helpers below don't
# use ``datasets``; only the public ``load_programmatic_bank`` does).
@pytest.fixture(scope="module")
def loader_mod():
    try:
        return importlib.import_module(
            "token_efficiency_env.prompt_bank_loader"
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"prompt_bank_loader unimportable: {exc}")


def test_extract_keywords_returns_at_most_k_content_tokens(loader_mod):
    kws = loader_mod._extract_keywords(
        "Paris is the capital of France.", k=3
    )
    # No stopwords, no punctuation, at most 3.
    assert 1 <= len(kws) <= 3
    assert all(w.islower() for w in kws)
    assert "is" not in kws
    assert "the" not in kws


def test_extract_keywords_handles_empty_string(loader_mod):
    assert loader_mod._extract_keywords("") == []


def test_extract_keywords_handles_apostrophe_tokens(loader_mod):
    """Regression — tokens that contain surrounding apostrophes used to
    crash the sort with ``ValueError: 'X' is not in list`` because the
    stripped candidate didn't match the unstripped source list. The fix
    records token order on the fly instead of calling ``.index()``.

    Real-world examples: ``"'quoted'"`` wrappers in dataset rows, and
    contractions whose apostrophe the regex happens to catch at a boundary.
    """
    for answer in [
        "Einstein's 'famous' theory of relativity.",
        "The 'quick' brown fox jumps over the 'lazy' dog.",
        "She said 'hello' to the 'stranger'.",
    ]:
        result = loader_mod._extract_keywords(answer, k=3)
        assert isinstance(result, list)
        assert all(isinstance(w, str) for w in result)
        assert len(result) <= 3


def test_numeric_aliases_emit_list_of_lists_for_digits(loader_mod):
    """``"30"`` must become ``["30", "thirty"]`` when num2words is
    installed; otherwise fall back to flat ``"30"``."""
    result = loader_mod._add_numeric_aliases(["30", "paris"])
    # At minimum, the non-numeric keyword is passed through unchanged.
    assert "paris" in result
    # The numeric one is either aliased (list) or passed through (str).
    numeric_entry = next(e for e in result if e != "paris")
    assert numeric_entry == "30" or numeric_entry == ["30", "thirty"]


def test_bucket_by_expected_length_uses_token_count(loader_mod):
    """Short answer → easy, medium-length → medium, long reasoning →
    hard. Exact thresholds depend on Qwen tokenization; we just assert
    the monotonic direction."""
    try:
        easy = loader_mod._bucket_by_expected_length("Paris.")
        medium = loader_mod._bucket_by_expected_length(
            "Paris is the capital and largest city of France, "
            "located in the north-central part of the country on the Seine river."
        )
        hard = loader_mod._bucket_by_expected_length(
            "The Turing Test evaluates a machine's ability to exhibit "
            "intelligent behaviour indistinguishable from a human's. "
            "Proposed by Alan Turing in 1950, it involves a human judge "
            "engaging in natural-language conversations with one human "
            "and one machine, each of which tries to appear human. "
            "If the judge cannot reliably tell which is which, the "
            "machine is said to have passed the test. The test does "
            "not measure intelligence per se, only the capacity for "
            "humanlike conversational behaviour, and has been "
            "criticised on both philosophical and empirical grounds."
        )
    except Exception as exc:
        pytest.skip(f"tokenizer unavailable offline: {exc}")

    # easy should always be "easy"; the longer ones shouldn't be "easy".
    assert easy == "easy"
    assert medium in ("medium", "hard")
    assert hard == "hard"


def test_build_entry_emits_canonical_schema(loader_mod):
    """A constructed entry must match the schema the scorer expects:
    prompt:str, complexity:str in {easy,medium,hard}, expected_keywords:list."""
    try:
        entry = loader_mod._build_entry(
            "What is the boiling point of water in Celsius?",
            "100",
        )
    except Exception as exc:
        pytest.skip(f"tokenizer unavailable offline: {exc}")
    assert isinstance(entry, dict)
    assert set(entry) >= {"prompt", "complexity", "expected_keywords"}
    assert entry["prompt"] == "What is the boiling point of water in Celsius?"
    assert entry["complexity"] in {"easy", "medium", "hard"}
    assert isinstance(entry["expected_keywords"], list)


def test_load_programmatic_bank_raises_importerror_without_datasets(loader_mod, monkeypatch):
    """Without ``datasets`` installed, the public loader must surface a
    clear ImportError so the dispatch fallback in prompts.py fires."""
    import sys

    # Pretend ``datasets`` is not importable, regardless of the real env.
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match="datasets"):
        loader_mod.load_programmatic_bank(seed=0, max_prompts=1)


# ─── 3. The real network integration — opt-in only ────────────────────
@pytest.mark.slow
def test_load_programmatic_bank_integration_smoke(loader_mod):
    """Full network-backed path, tiny slice. Opt-in via `-m slow`."""
    try:
        bank = loader_mod.load_programmatic_bank(
            seed=0, max_prompts=10, sources=("gsm8k",)
        )
    except Exception as exc:
        pytest.skip(f"network / dataset not available: {exc}")
    assert 0 < len(bank) <= 10
    for entry in bank:
        assert "prompt" in entry and entry["prompt"]
        assert entry["complexity"] in {"easy", "medium", "hard"}
        assert isinstance(entry["expected_keywords"], list)
