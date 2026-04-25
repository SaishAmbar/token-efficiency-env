"""End-to-end WebSocket round-trip tests against a locally running server.

These are *integration* tests, not unit tests. They require a running
OpenEnv server. To run them:

    # In one shell, start the server (HF_TOKEN optional, falls back to keyword):
    $env:JUDGE_BACKEND = "keyword"
    python -m uvicorn token_efficiency_env.server.app:app --host 127.0.0.1 --port 8000

    # In another shell:
    pytest tests/test_e2e_websocket.py -v

If the server isn't running, every test in this file is skipped.

Promoted from ``_phase4_e2e.py`` (the original Phase 4 sanity script).

KEY FACT (worth re-stating because it cost us hours of debugging):
    OpenEnv's ``/reset`` and ``/step`` HTTP endpoints are STATELESS.
    Each request constructs a fresh Environment instance. Stateful sessions
    (curriculum progression, recent_rewards deque, prompt round-trip
    between reset and step) live behind the WebSocket interface used by
    ``token_efficiency_env.client.TokenEfficiencyEnv``.

    GRPO trainers MUST drive the env through ``TokenEfficiencyEnv``,
    NOT raw HTTP.
"""

from __future__ import annotations

import json

import httpx
import pytest

from token_efficiency_env.client import TokenEfficiencyEnv
from token_efficiency_env.models import TokenEfficiencyAction

BASE = "http://127.0.0.1:8000"


def _server_alive() -> bool:
    try:
        with httpx.Client(timeout=1.5) as c:
            return c.get(f"{BASE}/health").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_alive(),
    reason=f"no OpenEnv server running on {BASE} (start uvicorn first)",
)


# ─── 1. Server liveness + advertised schema (raw HTTP — stateless meta) ─
def test_server_advertises_action_and_observation_fields():
    with httpx.Client(base_url=BASE, timeout=30.0) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/metadata").status_code == 200

        schema = c.get("/schema").json()
        assert isinstance(schema, dict) and schema, "schema must be non-empty"
        schema_str = json.dumps(schema)
        assert "raw_response" in schema_str, "action field 'raw_response' missing"
        assert "reward_components" in schema_str, \
            "observation field 'reward_components' missing"


# ─── 2. WS happy path: prompt MUST round-trip across reset/step ─────────
def test_ws_reset_then_step_keeps_same_prompt():
    """If ``max_concurrent_envs`` is bumped or session affinity breaks, the
    prompt the agent answered will differ from the prompt the env scored.
    This catches that class of bug immediately."""
    with TokenEfficiencyEnv(base_url=BASE).sync() as env:
        reset_prompt = env.reset().observation.prompt
        assert reset_prompt, "reset must return a non-empty prompt"

        res = env.step(TokenEfficiencyAction(
            raw_response="<budget>20</budget><answer>Paris.</answer>"
        ))
        assert res.observation.prompt == reset_prompt, (
            "prompt drifted between reset and step; "
            "max_concurrent_envs likely > 1 (must be 1)"
        )
        assert res.reward is not None
        assert res.observation.allocated_budget == 20
        assert res.observation.tokens_used > 0
        assert isinstance(res.observation.reward_components, dict)
        assert "correctness" in res.observation.reward_components


# ─── 3. WS cliffs over the wire (parametrised) ──────────────────────────
@pytest.mark.parametrize("raw_response, expected_error, expected_reward", [
    pytest.param(
        "<answer>Paris.</answer>",
        "bad_format", -1.0,
        id="ws-bad-format",
    ),
    pytest.param(
        "<budget>5</budget><answer>.</answer>",
        "empty", -1.0,
        id="ws-empty",
    ),
])
def test_ws_cliff_wire_format(raw_response, expected_error, expected_reward):
    with TokenEfficiencyEnv(base_url=BASE).sync() as env:
        env.reset()
        res = env.step(TokenEfficiencyAction(raw_response=raw_response))
        assert res.reward == expected_reward
        assert res.observation.error == expected_error


def test_ws_parrot_cliff_uses_actual_reset_prompt():
    """Parrot guard needs a fresh prompt for each call — using a hardcoded
    one would race the curriculum's question selection."""
    with TokenEfficiencyEnv(base_url=BASE).sync() as env:
        prompt = env.reset().observation.prompt
        res = env.step(TokenEfficiencyAction(
            raw_response=f"<budget>20</budget><answer>{prompt}</answer>"
        ))
        assert res.reward == -0.5
        assert res.observation.error == "parrot"


# ─── 4. WS state persistence across multiple steps ──────────────────────
def test_ws_episode_counter_advances_across_steps():
    """If the WS session were stateless, episode would always be 1.
    Seeing 1..5 proves the env instance is shared for the connection."""
    with TokenEfficiencyEnv(base_url=BASE).sync() as env:
        episodes = []
        for _ in range(5):
            env.reset()
            res = env.step(TokenEfficiencyAction(
                raw_response="<budget>20</budget><answer>Paris.</answer>"
            ))
            episodes.append(res.observation.episode)
        assert episodes == [1, 2, 3, 4, 5], (
            f"WS session lost state; expected [1..5], got {episodes}"
        )


def test_ws_avg_reward_50_evolves_as_rewards_accumulate():
    """A persisting recent_rewards deque means the rolling average shifts
    when individual rewards differ. We use intentionally varied answers so
    the rewards differ regardless of which random prompt the env picks."""
    answers = [
        "<budget>10</budget><answer>Paris.</answer>",          # short, sometimes correct
        "<budget>10</budget><answer>x</answer>",               # empty cliff (-1.0)
        "<budget>10</budget><answer>" + ("hello " * 80) + "</answer>",  # repetition cliff (-0.5)
        "<budget>10</budget><answer>Tokyo Berlin Madrid Lima.</answer>",  # multi-word, varied
        "<budget>10</budget><answer>Paris is in France.</answer>",
    ]
    with TokenEfficiencyEnv(base_url=BASE).sync() as env:
        env.reset()
        avgs = []
        for raw in answers:
            res = env.step(TokenEfficiencyAction(raw_response=raw))
            avgs.append(res.observation.avg_reward_50)
            env.reset()
        assert len(set(avgs)) > 1, (
            f"avg_reward_50 never changed across 5 deliberately-varied steps; "
            f"recent_rewards deque is not persisting ({avgs})"
        )
