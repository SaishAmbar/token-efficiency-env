"""Regression tests for the ``judge_stats()`` health helper.

``/judge_stats`` is the operator-visible way to tell whether the LLM
judge is silently degrading to the keyword fallback. The values don't
need to match exactly between runs, but the *shape* of the response
must be stable so dashboards and monitoring queries don't break.
"""

from __future__ import annotations

import os

import pytest

from token_efficiency_env import judge as judge_module
from token_efficiency_env.judge import (
    KeywordJudge,
    judge_stats,
    reset_judge,
)


@pytest.fixture(autouse=True)
def _clean_judge() -> None:
    """Reset the process-level singleton before and after every test.

    Without this, test order starts mattering: one test builds a
    KeywordJudge, the next reads the cached copy expecting
    "uninitialised".
    """
    reset_judge()
    yield
    reset_judge()


def test_uninitialised_shape_has_all_required_fields() -> None:
    """Before any call, we still return a well-formed snapshot."""
    stats = judge_stats()
    assert stats["backend"] == "uninitialised"
    assert stats["calls"] == 0
    assert stats["failures"] == 0
    assert stats["failure_rate"] == 0.0


def test_keyword_judge_stats_shape() -> None:
    """KeywordJudge.stats() matches the protocol contract."""
    kj = KeywordJudge()
    stats = kj.stats()
    assert stats["backend"] == "KeywordJudge"
    assert stats["calls"] == 0
    assert stats["failures"] == 0
    assert stats["failure_rate"] == 0.0

    kj.score_correctness("q", "a", ["x"])
    kj.score_correctness("q", "a", ["y"])
    stats = kj.stats()
    assert stats["calls"] == 2
    assert stats["failures"] == 0


def test_judge_stats_reflects_configured_singleton() -> None:
    """After ``get_judge()`` runs, ``judge_stats()`` returns the real backend."""
    # conftest.py sets JUDGE_BACKEND=keyword, so this avoids any network.
    judge_module.get_judge()
    stats = judge_stats()
    assert stats["backend"] == "KeywordJudge"
    # Fresh singleton: no calls yet.
    assert stats["calls"] == 0


def test_uninitialised_does_not_force_judge_construction() -> None:
    """Reading stats must NOT build a judge (or we'd defeat the point).

    If the server is running with a missing HF_TOKEN, an unconditional
    ``get_judge()`` inside the /judge_stats handler would either raise
    or print a warning every time an operator curls the endpoint. The
    correct behaviour is to report ``backend: "uninitialised"`` until
    something actually needs to score.
    """
    reset_judge()
    _ = judge_stats()
    # The module-level singleton should still be None.
    assert judge_module._JUDGE is None
