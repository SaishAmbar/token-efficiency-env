# Tests

Two suites live here.

## `test_reward_invariants.py` — fast unit tests

In-process tests that instantiate `TokenEfficiencyEnvironment` directly. They use the offline `KeywordJudge` (set in `conftest.py`), so they need **no network and no `HF_TOKEN`** and complete in well under a second.

What they cover:
- The 6-component reward function returns sensible values for hand-picked correct/wrong answers.
- All non-cliff rewards land in `[-1.0, 1.0]`.
- Each anti-hacking cliff (`bad_format`, `empty`, `parrot`) short-circuits to its expected reward + error tag.
- The parrot guard does *not* misfire on legitimate verbose answers.

Run them with:

```bash
pytest tests/test_reward_invariants.py -v
```

## `test_e2e_websocket.py` — integration tests

Round-trip tests against a *running* OpenEnv server. They prove that the WebSocket session keeps a single env instance alive (so the prompt round-trips between `reset` and `step`, the episode counter advances, and the rolling average accumulates).

Every test in this file is **automatically skipped** if `http://127.0.0.1:8000/health` doesn't respond, so running `pytest` without a server is still green.

To run them:

```bash
# Shell 1 — start the server (keyword judge keeps the test fast and offline):
$env:JUDGE_BACKEND = "keyword"
python -m uvicorn token_efficiency_env.server.app:app --host 127.0.0.1 --port 8000

# Shell 2 — run the suite:
pytest tests/test_e2e_websocket.py -v
```

## Running everything

```bash
pytest tests/ -v
```

The integration suite is skipped when no server is up, so this is safe.

## Why these tests exist

Both suites started life as `_phase3_smoketest.py` and `_phase4_e2e.py` in the repo root — manual scripts written during the Phase 3 and Phase 4 implementation work. They were promoted to a `pytest`-driven suite in Phase 6 so the env's invariants are checked automatically, not just on demand.

The original scripts have been removed; everything they covered (and a bit more) lives here.
