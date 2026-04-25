# `token_efficiency_env` — In-Package Contributor Guide

> This is the **in-package** README, intended for someone editing files inside `token_efficiency_env/`. For the project-level overview and quick start, read the [root `README.md`](../README.md) and [`ARCHITECTURE.md`](../ARCHITECTURE.md) first.

---

## Table of contents

1. [What lives in this package](#1-what-lives-in-this-package)
2. [The wire contract](#2-the-wire-contract)
3. [`server/token_efficiency_env_environment.py` — the environment](#3-servertoken_efficiency_env_environmentpy--the-environment)
4. [`scorer.py` — the 6-component reward](#4-scorerpy--the-6-component-reward)
5. [`judge.py` — the LLM judge](#5-judgepy--the-llm-judge)
6. [`prompts.py` — the question bank](#6-promptspy--the-question-bank)
7. [`models.py` — Pydantic schemas](#7-modelspy--pydantic-schemas)
8. [`client.py` — the typed WebSocket client](#8-clientpy--the-typed-websocket-client)
9. [`server/app.py` — the OpenEnv server wrapper](#9-serverapppy--the-openenv-server-wrapper)
10. [`openenv.yaml` — deployment manifest](#10-openenvyaml--deployment-manifest)
11. [Environment variables](#11-environment-variables)
12. [Common edits and where to make them](#12-common-edits-and-where-to-make-them)
13. [Running the server](#13-running-the-server)
14. [Running the test suite](#14-running-the-test-suite)
15. [Common errors and fixes](#15-common-errors-and-fixes)

---

## 1. What lives in this package

```
token_efficiency_env/
├── __init__.py
├── pyproject.toml          ← package metadata + dependencies
├── openenv.yaml            ← OpenEnv deployment manifest
├── README.md               ← you are here
│
├── prompts.py              ← 24 hand-curated questions w/ keywords
├── scorer.py               ← 6-component reward function
├── judge.py                ← Judge protocol + HF/keyword backends
├── models.py               ← Pydantic Action/Observation schemas
├── client.py               ← typed WebSocket client (use this in trainers)
│
├── env_server.py           ← thin back-compat re-export (deprecated path)
├── env_client.py           ← thin back-compat re-export (deprecated path)
│
└── server/
    ├── app.py              ← FastAPI wrapper that mounts the env
    ├── token_efficiency_env_environment.py   ← THE ENVIRONMENT
    ├── requirements.txt    ← used by the Docker image
    └── Dockerfile          ← used by HF Spaces deploy (Phase 5, deferred)
```

**Where the canonical implementation lives:**

| Concept | File |
|---|---|
| Environment (`reset` / `step` / curriculum / cliffs) | `server/token_efficiency_env_environment.py` |
| Reward function (6 components) | `scorer.py` |
| LLM judge (correctness scoring) | `judge.py` |
| Question bank | `prompts.py` |
| Wire schemas | `models.py` |
| Trainer-facing client | `client.py` |

`env_server.py` and `env_client.py` are **shims** kept only for backward compatibility with old import paths. New code should import from the canonical modules above.

---

## 2. The wire contract

The environment exchanges two Pydantic objects with the trainer over WebSocket:

**Action** (`models.TokenEfficiencyAction`) — what the trainer sends:
```python
class TokenEfficiencyAction(Action):
    raw_response: str
```
Exactly one field: the *raw* string the LLM emitted, expected to be:
```
<budget>N</budget><answer>your answer text</answer>
```

**Observation** (`models.TokenEfficiencyObservation`) — what the env returns from `reset()` and `step()`:

| Field | Type | When populated | Description |
|---|---|---|---|
| `prompt` | `str` | always | The question to answer |
| `episode_token_limit` | `int` | always | Hard cap on tokens the env will tolerate (default 200) |
| `answer` | `str` | after `step` | Parsed text inside `<answer>…</answer>` |
| `allocated_budget` | `int` | after `step` | Parsed `<budget>N</budget>`, clamped to `[1, 200]` |
| `tokens_used` | `int` | after `step` | Qwen tokenizer count of the answer |
| `complexity` | `str` | after `step` | `"easy"` / `"medium"` / `"hard"` |
| `phase` | `str` | always | Curriculum phase name |
| `episode` | `int` | always | Per-session episode counter |
| `avg_reward_50` | `float` | always | Rolling 50-episode mean reward |
| `reward_components` | `dict[str, float]` | after `step` (empty on cliff) | Per-component breakdown |
| `error` | `str` | after `step` | Non-empty when a cliff fired (`bad_format` / `empty` / `parrot` / `repetition` / `too_long`) |

`reward` and `done` are inherited from `openenv.core.env_server.types.Observation`.

---

## 3. `server/token_efficiency_env_environment.py` — the environment

This is the file that implements OpenEnv's `Environment` interface. It's the **canonical** env class — the OpenEnv server, the Gradio web UI, and the in-process tests all instantiate it directly.

**Key responsibilities:**

| Concern | Method / member |
|---|---|
| Pick a curriculum-weighted question | `_select_phase()` + `_sample_task()` |
| Start a new episode | `reset()` |
| Score one response | `step(action)` |
| Run anti-hacking guards | inline at the top of `step()` |
| Track recent rewards for the curriculum | `self.recent_rewards` (a `collections.deque(maxlen=50)`) |
| Lazily load the Qwen tokenizer | `_get_tokenizer()` |

**`reset()`** does:
1. Calls `_select_phase()` to compute the current curriculum index from `mean(recent_rewards)`.
2. Calls `_sample_task()` to pick a question weighted by the phase's mix.
3. Stashes `self.current_task`, increments `self.episode_count`.
4. Returns a `TokenEfficiencyObservation` with `prompt`, `episode_token_limit`, `phase`, `episode`, `avg_reward_50`.

**`step(action)`** does (in order):
1. Regex-parse `<budget>N</budget>` and `<answer>…</answer>`. **Missing/malformed → `bad_format` cliff (`reward = -1.0`)**.
2. Clamp `budget` to `[1, MAX_TOKEN_LIMIT]`.
3. **Empty answer guard**: `len(answer) < MIN_ANSWER_CHARS` (= 2) → `empty` cliff (`-1.0`).
4. **Parrot guard**: if the normalised question is a substring of the normalised answer → `parrot` cliff (`-0.5`). Normalisation strips punctuation and lower-cases, so verbose-but-legitimate answers don't misfire.
5. **Repetition guard**: if a single word covers more than `REPETITION_THRESHOLD` (= 0.6) of the answer → `repetition` cliff (`-0.5`).
6. Count tokens with the Qwen tokenizer.
7. **Length guard**: if `tokens_used > MAX_ANSWER_TOKENS` (= 500) → `too_long` cliff (`-0.5`).
8. Call `scorer.score_answer(...)` for the 6-component reward.
9. Append the reward to `self.recent_rewards` (deque auto-evicts oldest).
10. Return a `TokenEfficiencyObservation` with all fields populated.

**Constants you might tune:**

```python
MAX_TOKEN_LIMIT = 200      # hard ceiling on per-episode token count
REWARD_WINDOW   = 50       # rolling window for curriculum decisions
MIN_ANSWER_CHARS = 2       # below this length, answer is treated as empty
```

---

## 4. `scorer.py` — the 6-component reward

```python
def score_answer(
    prompt: str,
    response: str,
    allocated_budget: int,
    tokens_used: int,
    complexity: str = "medium",
    expected_keywords: Optional[list[str]] = None,
) -> dict
```

Returns `{"reward": float, "details": {component: float, ...}}`.

The components and their weights:

| # | Component | Weight | Formula sketch |
|---|---|---:|---|
| 1 | `correctness` | **0.55** | `judge.get_judge().score_correctness(prompt, response, expected_keywords)` |
| 2 | `efficiency` | **0.15** | `_efficiency_absolute(tokens_used, complexity)` against `COMPLEXITY_IDEAL_TOKENS` |
| 3 | `self_assessment` | **0.15** | `_self_assessment_asymmetric(allocated_budget, tokens_used)` |
| 4 | `redundancy` | **0.05** | `1 - duplicated_word_ratio` |
| 5 | `keyword_verification` | **0.05** | word-boundary overlap with `expected_keywords` |
| 6 | `format_quality` | **0.05** | both tags present, budget is positive int, well-ordered |

```python
reward = (
    0.55 * correctness
  + 0.15 * efficiency
  + 0.15 * self_assessment
  + 0.05 * redundancy
  + 0.05 * keyword_verification
  + 0.05 * format_quality
)
# then clamped to [-1.0, 1.0]
```

**Why these specific shapes:**

- **`efficiency` is absolute, not relative to the model's budget.** If we used `tokens_used / allocated_budget`, the model could max efficiency by predicting `<budget>1000</budget>` and using 1 token. We use `_efficiency_absolute(tokens_used, complexity)` against fixed ideals (`easy=15`, `medium=60`, `hard=130`).
- **`self_assessment` is asymmetric.** Symmetric "match your budget exactly" rewarded contradicts "use as few tokens as possible". The new shape gives a mild penalty for slack and a steep penalty for overshoot — the real-world objective of *honest upper-bound prediction*.
- **`correctness` from an LLM judge** because keyword matching alone is too brittle for medium/hard questions ("vaccine" vs "vaccination", or "the moon" vs "Earth's natural satellite").
- **`keyword_verification` is kept** as a 5% sanity floor in case the judge hallucinates.
- **`budget_reasonableness` was dropped** in v0.2.0 — it rewarded memorising a fixed prior and double-dipped with `efficiency`.

See [`ARCHITECTURE.md`](../ARCHITECTURE.md) §5 for the full rationale.

---

## 5. `judge.py` — the LLM judge

The judge is the source of the `correctness` component. It's pluggable so the env still works without an HF token (degraded but functional).

**Interface:**

```python
class Judge(Protocol):
    def score_correctness(
        self,
        prompt: str,
        answer: str,
        expected_keywords: Optional[list[str]] = None,
    ) -> float: ...
```

**Implementations:**

| Class | When to use | How it works |
|---|---|---|
| `HFInferenceJudge` | Production / training | Calls `huggingface_hub.InferenceClient.chat_completion` against the Inference Providers router. Default model: `meta-llama/Llama-3.1-8B-Instruct`. Falls back to keyword scoring on per-call API failures. |
| `KeywordJudge` | Tests / CI / no-token dev | Pure-Python word-boundary keyword overlap. Zero network. |

**Selection** is driven by env vars at process start:

| Var | Values | Default | Effect |
|---|---|---|---|
| `JUDGE_BACKEND` | `huggingface` / `keyword` | `huggingface` | Picks the implementation |
| `JUDGE_MODEL` | any HF chat model | `meta-llama/Llama-3.1-8B-Instruct` | Overrides the HF model |
| `HF_TOKEN` | a token string | (unset) | Required by `HFInferenceJudge`; if missing, transparently degrades to `KeywordJudge` |

`get_judge()` is a process-level singleton (constructed lazily) so importing `token_efficiency_env` never touches the network and tests can `reset_judge()` between runs.

**Judge prompt design:**

The prompt feeds `expected_keywords` to the judge as anchor "key facts". Without these, smaller judge models (8B-class) hallucinate the right answer and grade against their hallucination — especially on math. The prompt also enforces a `Score: X.XX` output contract so `_parse_score()` can extract a number robustly even if the model adds prose.

---

## 6. `prompts.py` — the question bank

A static `PROMPT_BANK` list of 24 dicts:

```python
PROMPT_BANK = [
    {
        "prompt": "What is the capital of France? One word.",
        "complexity": "easy",
        "expected_keywords": ["paris"],
    },
    # ... 23 more ...
]
```

**Distribution:** 8 easy, 8 medium, 8 hard.

**`expected_keywords` are critical:**
- They feed both the `keyword_verification` component (5% direct) AND the LLM judge prompt (which uses them as anchor facts to ground correctness scoring).
- Use lowercase, the actual fact words (e.g. `["100", "celsius"]` for the boiling-point question), not function words.

To **add a question:**
1. Pick a complexity tier with intentionally calibrated answer length:
   - `easy` ≈ 1–5 word answers
   - `medium` ≈ 2–4 sentence answers
   - `hard` ≈ longer paragraph answers
2. Choose `expected_keywords` that any correct answer would contain.
3. Append to `PROMPT_BANK`. No other code changes needed — the curriculum samples by `complexity` automatically.

---

## 7. `models.py` — Pydantic schemas

Defines the on-the-wire data classes:

- `TokenEfficiencyAction` — single field `raw_response: str`.
- `TokenEfficiencyObservation` — see the table in §2.

`Action` and `Observation` base classes come from `openenv.core.env_server.types`. They provide `done`, `reward`, and `metadata` for free; **`metadata` is stripped on the wire** by OpenEnv's serializer, so anything the trainer needs to read must be a real declared field.

---

## 8. `client.py` — the typed WebSocket client

```python
from token_efficiency_env.client import TokenEfficiencyEnv
from token_efficiency_env.models import TokenEfficiencyAction

with TokenEfficiencyEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()
    result = env.step(TokenEfficiencyAction(raw_response="..."))
```

**This is the only path trainers should use.** It subclasses `openenv.core.EnvClient`, which uses WebSockets, which means a single env instance is held for the connection's lifetime. That's how curriculum progression and `recent_rewards` accumulate across calls.

The two private hooks the subclass provides:

- `_step_payload(action)` → `{"raw_response": action.raw_response}` (what gets sent on the wire)
- `_parse_result(payload)` → builds a `StepResult` with the rich `TokenEfficiencyObservation`

Don't bypass this and call `httpx.post("/step")` directly — see §9 for why.

---

## 9. `server/app.py` — the OpenEnv server wrapper

A 30-line FastAPI app that mounts the environment via `openenv.core.env_server.create_app(...)`.

The single most important thing in this file:

```python
app = create_app(
    TokenEfficiencyEnvironment,
    TokenEfficiencyAction,
    TokenEfficiencyObservation,
    env_name="token_efficiency_env",
    max_concurrent_envs=1,   # ← intentional, do NOT change
)
```

**Why `max_concurrent_envs=1`?** OpenEnv's HTTP layer round-robins `/reset` and `/step` requests across the pool with **no session affinity**. With a pool > 1, `POST /reset` could land on instance #3 while the matching `POST /step` lands on instance #5 — meaning the agent gets scored on a different question than the one it was asked. The WebSocket interface (used by `client.py`) is the *only* stateful path.

**For parallel rollouts** (GRPO needs them): run **N separate server processes on N ports**, each at `max_concurrent_envs=1`, and let the trainer load-balance.

---

## 10. `openenv.yaml` — deployment manifest

```yaml
name: token-efficiency-env
version: 0.2.0
description: >
  An RL environment that trains LLMs to answer correctly using fewer tokens
  ...
entry_point:        token_efficiency_env.server.token_efficiency_env_environment:TokenEfficiencyEnvironment
action_schema:      token_efficiency_env.models:TokenEfficiencyAction
observation_schema: token_efficiency_env.models:TokenEfficiencyObservation
```

OpenEnv reads this when deploying to a HuggingFace Space (Phase 5, deferred). Keep `entry_point` and the schema paths pointing at the **canonical** modules — never at the back-compat shims (`env_server.py` / `env_client.py`).

---

## 11. Environment variables

| Var | Required? | Where used | Default | Notes |
|---|---|---|---|---|
| `HF_TOKEN` | Recommended (otherwise judge degrades) | `judge.HFInferenceJudge` | (unset) | Needs **"Make calls to Inference Providers"** permission |
| `JUDGE_BACKEND` | No | `judge.get_judge()` | `huggingface` | Set to `keyword` to skip the LLM judge entirely |
| `JUDGE_MODEL` | No | `judge.HFInferenceJudge` | `meta-llama/Llama-3.1-8B-Instruct` | Any HF Inference-supported chat model |
| `ENABLE_WEB_INTERFACE` | No | `openenv-core` | `false` | Set to `true` to mount Gradio UI at `/web/` |

---

## 12. Common edits and where to make them

| You want to… | Edit | Notes |
|---|---|---|
| Add or rephrase a question | `prompts.py` | Set `complexity` and `expected_keywords` |
| Tune reward weights | `scorer.py::score_answer` | Weights sum to 1.0; cliff penalties are separate |
| Change a reward formula (e.g. self-assessment shape) | `scorer.py` (`_self_assessment_asymmetric`, `_efficiency_absolute`) | Add a regression test in `tests/test_reward_invariants.py` |
| Add a new anti-hacking cliff | `server/token_efficiency_env_environment.py::step` | Use `self._fail(error=..., reward=...)` for a clean exit |
| Swap the judge model | env var `JUDGE_MODEL`, no code change | Or hard-code `DEFAULT_JUDGE_MODEL` in `judge.py` |
| Add a new judge backend | `judge.py` — implement `Judge` Protocol, register in `get_judge()` | |
| Change the action format | `models.TokenEfficiencyAction` + parsing in `step()` | The Gradio UI auto-rebuilds its form from the schema |
| Change the curriculum thresholds | `server/token_efficiency_env_environment.py::CURRICULUM_PHASES` | |
| Tune episode token limit | `server/token_efficiency_env_environment.py::MAX_TOKEN_LIMIT` (also exported from `models.py` for the wire) | |
| Add a new observation field | `models.TokenEfficiencyObservation` + populate in `reset`/`step` + unpack in `client.py::_parse_result` | All three places — easy to forget the client! |

---

## 13. Running the server

From the repo root, with package installed in editable mode (`pip install -e ./token_efficiency_env`):

```powershell
# PowerShell — with HF judge + Gradio UI
$env:HF_TOKEN = 'hf_...'
$env:JUDGE_BACKEND = 'huggingface'
$env:ENABLE_WEB_INTERFACE = 'true'
python -m uvicorn token_efficiency_env.server.app:app --host 127.0.0.1 --port 8000
```

```bash
# bash — with offline keyword judge (no token needed)
JUDGE_BACKEND=keyword \
ENABLE_WEB_INTERFACE=true \
  python -m uvicorn token_efficiency_env.server.app:app --host 127.0.0.1 --port 8000
```

Then:

| URL | What it is |
|---|---|
| <http://localhost:8000/web/> | Gradio playground |
| <http://localhost:8000/docs> | FastAPI Swagger UI (manual `/reset`, `/step`) |
| <http://localhost:8000/health> | Liveness probe |
| <http://localhost:8000/metadata> | Env name / version |

---

## 14. Running the test suite

From the repo root:

```bash
pytest tests/                      # all tests
pytest tests/test_reward_invariants.py -v   # fast unit tests only
pytest tests/test_e2e_websocket.py -v       # integration tests (needs server)
```

The integration suite **auto-skips** if no server is up at `http://127.0.0.1:8000`, so the unit suite always runs green even on a fresh machine without HF credentials.

---

## 15. Common errors and fixes

**`ModuleNotFoundError: No module named 'openenv'`**
→ Install dependencies: `pip install -e ./token_efficiency_env` from the repo root.

**`HFInferenceJudge requires the HF_TOKEN environment variable`**
→ Either set `HF_TOKEN`, or set `JUDGE_BACKEND=keyword` to use the offline scorer.

**`Address already in use` when starting uvicorn on port 8000**
→ A previous server is still running. On Windows:
```powershell
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

**Reward is `-1.0` and `error == "bad_format"` even on what looks like a valid response**
→ The regex requires `<budget>` then `<answer>` in that order, both with closing tags, both wrapping the right content. Check for stray whitespace inside the tags or missing `</budget>`.

**Reset returns one prompt, but Step seems to score against a different prompt**
→ You're going through raw HTTP. OpenEnv's HTTP endpoints are stateless. Use `TokenEfficiencyEnv` from `client.py` (it uses WebSockets). See [`ARCHITECTURE.md`](../ARCHITECTURE.md) §9.

**Llama-3.1-8B judge gives 0.0 for an obviously correct answer**
→ Make sure the question's `expected_keywords` in `prompts.py` are realistic anchor facts. They're injected into the judge prompt as "key facts a correct answer must contain". Bad keywords → bad judging.

**`huggingface_hub` cache symlinks warning on Windows**
→ Cosmetic, ignore. To silence: `$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"`.

---

*Last updated: 2026-04 · v0.2.0*
