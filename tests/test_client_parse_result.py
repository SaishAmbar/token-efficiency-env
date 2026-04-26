"""Regression tests for ``TokenEfficiencyEnv._parse_result``.

Before v0.3.0 the WebSocket client silently dropped the
``answer_token_count`` field from the server's payload, so trainers using
the WS reward backend had no way to see the inner-answer-only token
count diagnostic (Layer B of the hidden-CoT fix). This file pins the
deserialisation shape so a future refactor can't regress it.
"""

from __future__ import annotations

from token_efficiency_env.client import TokenEfficiencyEnv


def _make_client() -> TokenEfficiencyEnv:
    """Build a client without opening a real WebSocket.

    ``_parse_result`` is a pure method on the class; we only need an
    instance to call it bound, not a live connection.
    """
    return TokenEfficiencyEnv.__new__(TokenEfficiencyEnv)


def test_answer_token_count_plumbed_through_when_present() -> None:
    client = _make_client()
    payload = {
        "observation": {
            "prompt": "What is the capital of France?",
            "episode_token_limit": 200,
            "answer": "Paris.",
            "allocated_budget": 3,
            "tokens_used": 12,
            "answer_token_count": 2,
            "complexity": "easy",
            "phase": "foundation",
            "episode": 1,
            "avg_reward_50": 0.0,
            "reward_components": {"correctness": 1.0, "efficiency": 1.0},
            "error": "",
        },
        "reward": 0.95,
        "done": True,
    }

    result = client._parse_result(payload)

    assert result.observation.answer_token_count == 2
    # The other fields are still plumbed correctly — regression guard.
    assert result.observation.tokens_used == 12
    assert result.observation.answer == "Paris."
    assert result.reward == 0.95
    assert result.done is True


def test_answer_token_count_defaults_to_none_when_missing() -> None:
    """Old-server compatibility: missing field must not crash the client."""
    client = _make_client()
    payload = {
        "observation": {
            "prompt": "q",
            "episode_token_limit": 200,
            "answer": "a",
            "allocated_budget": 1,
            "tokens_used": 1,
            # Intentionally no ``answer_token_count`` key.
            "complexity": "easy",
            "phase": "foundation",
            "episode": 1,
            "avg_reward_50": 0.0,
            "reward_components": {},
            "error": "",
        },
        "reward": 0.5,
        "done": True,
    }

    result = client._parse_result(payload)

    assert result.observation.answer_token_count is None
    assert result.observation.tokens_used == 1
