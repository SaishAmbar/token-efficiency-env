"""Phase 4 — end-to-end smoke test against a locally running server.

KEY FACT (took some debugging to confirm):
  OpenEnv's HTTP `/reset` and `/step` endpoints are STATELESS by design.
  They construct a fresh Environment instance per request and discard it,
  which means our curriculum, recent_rewards deque, and the link between
  reset's task and step's scoring all break under raw HTTP.

  Stateful sessions live behind the WebSocket interface, accessed via the
  ``openenv.core.EnvClient`` base class. Our ``TokenEfficiencyEnv`` client
  in ``client.py`` extends ``EnvClient`` so it uses WS automatically.

  *** GRPO trainers MUST use TokenEfficiencyEnv, NOT raw HTTP. ***

Prereqs (in another shell):
    $env:HF_TOKEN = '<token>'
    $env:JUDGE_BACKEND = 'huggingface'
    python -m uvicorn token_efficiency_env.server.app:app --host 127.0.0.1 --port 8000

Then in this shell:
    python _phase4_e2e.py

Tests:
    1. Server liveness + advertised schema (raw HTTP — these endpoints are
       legitimately stateless: just metadata).
    2. WS happy path: reset -> step with Paris matched to the SAME prompt.
       Verifies prompt round-trip and reward_components dict serialization.
    3. WS cliffs: bad_format and empty answer hit the right -1.0 cliffs
       with the correct error tag.
    4. WS state persistence: 5 sequential reset+step pairs see episode
       counter advance and avg_reward_50 stabilize on a non-zero value.
       This proves the WS session keeps a single env instance alive.
"""
from __future__ import annotations

import json
import sys

import httpx

from token_efficiency_env.client import TokenEfficiencyEnv
from token_efficiency_env.models import TokenEfficiencyAction

BASE = "http://127.0.0.1:8000"


def hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def must(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        sys.exit(1)


# ─── 1. Server liveness + schema (raw HTTP — stateless metadata only) ─
hr("TEST 1: server liveness and advertised schema")
with httpx.Client(base_url=BASE, timeout=30.0) as c:
    r = c.get("/health")
    must(r.status_code == 200, f"GET /health -> status {r.status_code}")

    r = c.get("/metadata")
    must(r.status_code == 200, f"GET /metadata -> status {r.status_code}")
    print(f"    /metadata: {json.dumps(r.json())[:200]}")

    r = c.get("/schema")
    must(r.status_code == 200, f"GET /schema -> status {r.status_code}")
    schema = r.json()
    must(isinstance(schema, dict) and bool(schema), "schema is non-empty dict")
    print(f"    /schema top-level keys: {sorted(schema.keys())}")
    schema_str = json.dumps(schema)
    must("raw_response" in schema_str,
         "schema mentions our action field 'raw_response'")
    must("reward_components" in schema_str,
         "schema mentions our observation field 'reward_components'")


# ─── 2. WS happy path with prompt round-trip ──────────────────────────
hr("TEST 2: WS reset->step happy path (prompt MUST match across calls)")
with TokenEfficiencyEnv(base_url=BASE).sync() as env:
    res = env.reset()
    reset_prompt = res.observation.prompt
    print(f"    reset.prompt = {reset_prompt!r}")
    must(bool(reset_prompt), "client reset returns a prompt")

    res = env.step(TokenEfficiencyAction(
        raw_response="<budget>20</budget><answer>Paris.</answer>"
    ))
    step_prompt = res.observation.prompt
    print(f"    step.prompt  = {step_prompt!r}")
    print(f"    step.reward  = {res.reward}")
    print(f"    step.reward_components = {res.observation.reward_components}")
    must(reset_prompt == step_prompt,
         f"prompt is preserved across reset/step (the bug we hunted!)")
    must(res.reward is not None, "client step returns a reward")
    must(res.observation.allocated_budget == 20, "client parses allocated_budget")
    must(res.observation.tokens_used > 0, "client parses tokens_used > 0")
    must(isinstance(res.observation.reward_components, dict),
         "client parses reward_components dict")
    must("correctness" in res.observation.reward_components,
         "client preserves the correctness component")


# ─── 3. WS cliffs ─────────────────────────────────────────────────────
hr("TEST 3: WS cliffs - bad_format, empty, parrot")
with TokenEfficiencyEnv(base_url=BASE).sync() as env:
    env.reset()
    res = env.step(TokenEfficiencyAction(raw_response="<answer>Paris.</answer>"))
    must(res.reward == -1.0,
         f"bad_format -> reward=-1.0 (got {res.reward})")
    must(res.observation.error == "bad_format",
         f"error='bad_format' (got {res.observation.error!r})")

    env.reset()
    res = env.step(TokenEfficiencyAction(
        raw_response="<budget>5</budget><answer>.</answer>"
    ))
    must(res.reward == -1.0,
         f"empty -> reward=-1.0 (got {res.reward})")
    must(res.observation.error == "empty",
         f"error='empty' (got {res.observation.error!r})")

    reset_obs = env.reset().observation
    parrot_payload = (
        f"<budget>20</budget><answer>{reset_obs.prompt}</answer>"
    )
    res = env.step(TokenEfficiencyAction(raw_response=parrot_payload))
    must(res.reward == -0.5,
         f"parrot -> reward=-0.5 (got {res.reward})")
    must(res.observation.error == "parrot",
         f"error='parrot' (got {res.observation.error!r})")


# ─── 4. WS state persistence (curriculum + recent_rewards) ────────────
hr("TEST 4: WS state persistence across multiple steps")
with TokenEfficiencyEnv(base_url=BASE).sync() as env:
    episodes = []
    avgs = []
    for i in range(5):
        env.reset()
        res = env.step(TokenEfficiencyAction(
            raw_response="<budget>20</budget><answer>Paris.</answer>"
        ))
        episodes.append(res.observation.episode)
        avgs.append(res.observation.avg_reward_50)
        print(f"    step {i + 1}: episode={res.observation.episode}  "
              f"avg_reward_50={res.observation.avg_reward_50}  "
              f"reward={res.reward}")
    must(episodes == [1, 2, 3, 4, 5],
         f"episode counter advances 1..5 (got {episodes})")
    # If state weren't persisting, every avg would equal that step's reward
    # (a deque of size 1). Persistence means avg is a true rolling mean.
    must(len(set(avgs)) > 1 or avgs[-1] != avgs[0],
         f"avg_reward_50 evolves over steps, proving deque persists "
         f"({avgs})")


print()
print("=" * 72)
print("Phase 4 done - WS client path validated end-to-end.")
