# TokenEfficiencyEnv — Architecture & Onboarding

> **Read this file first.** The two existing `README.md` files describe an older version of the project (7-component scorer, Anthropic judge, hard-coded `anthropic` dependency). They will be cleaned up in Phase 6. Until then, treat **this file** as the canonical description of how the project actually works today.

---

## 1. TL;DR

`TokenEfficiencyEnv` is a [Meta OpenEnv](https://github.com/meta-pytorch/OpenEnv)–compatible RL environment. Its job is to teach a small language model (target: `Qwen/Qwen2.5-3B-Instruct`) to answer questions **correctly using as few tokens as possible**, by forcing the model to first *predict* its own token budget and then live within it.

Each episode is one question. The model must output **exactly** this string:

```
<budget>N</budget><answer>your answer here</answer>
```

The env then computes a scalar reward in roughly `[-1.0, +1.0]` from a 6-component scorer and returns it through the OpenEnv HTTP/WebSocket interface. A separate trainer (Phase 7, not yet built) will use those rewards to do GRPO updates on the LLM.

**Status:** Phases 1–4 complete (env logic, judge integration, anti-hacking, end-to-end local server). Phases 5 (HF Spaces deploy), 6 (docs cleanup), and 7 (training notebook scaffold) are pending.

---

## 2. Suggested reading order for new contributors

Read these files in this order. Skip nothing in the first three; skim the rest.

| # | File | Why |
|---|---|---|
| 1 | **`ARCHITECTURE.md`** (this file) | The map. |
| 2 | `token_efficiency_env/server/token_efficiency_env_environment.py` | The actual environment. The `reset()` / `step()` methods, curriculum logic, and anti-hacking guards live here. |
| 3 | `token_efficiency_env/scorer.py` | The 6-component reward function. The whole training signal comes from here. |
| 4 | `token_efficiency_env/judge.py` | The pluggable LLM judge (HuggingFace Inference, with keyword fallback). |
| 5 | `token_efficiency_env/prompts.py` | The hand-curated bank of 24 questions (8 easy / 8 medium / 8 hard) with expected keywords. |
| 6 | `token_efficiency_env/models.py` | The `TokenEfficiencyAction` / `TokenEfficiencyObservation` Pydantic schemas — the wire contract. |
| 7 | `token_efficiency_env/server/app.py` | Tiny FastAPI wrapper that mounts the environment as an OpenEnv server. Read the comment about `max_concurrent_envs`. |
| 8 | `token_efficiency_env/client.py` | The typed Python client a trainer uses to talk to the server over WebSockets. |
| 9 | `token_efficiency_env/openenv.yaml` | OpenEnv deployment manifest (entry point + schema paths). |
| 10 | `_phase3_smoketest.py` and `_phase4_e2e.py` | Sanity-check scripts. Useful as worked examples of "what good looks like" — not part of the production code. |

The two `README.md` files are deliberately *not* in this list — they're slated for rewrite in Phase 6.

---

## 3. End-to-end data flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            ONE TRAINING EPISODE                           │
│                                                                           │
│  1. Trainer calls   env.reset()                                           │
│     │                                                                     │
│     ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ TokenEfficiencyEnvironment._sample_task(curriculum_phase)        │    │
│  │   • picks one of 24 prompts from prompts.py weighted by phase    │    │
│  │   • returns Observation { prompt, episode_token_limit, ... }     │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│     │                                                                     │
│     ▼                                                                     │
│  2. LLM generates a response, e.g.                                        │
│        "<budget>8</budget><answer>Paris</answer>"                         │
│                                                                           │
│  3. Trainer calls   env.step(TokenEfficiencyAction(raw_response=…))       │
│     │                                                                     │
│     ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ TokenEfficiencyEnvironment.step(action)                          │    │
│  │   ① Parse  <budget>N</budget><answer>X</answer>   (regex)        │    │
│  │   ② Anti-hacking cliffs (bad_format, empty, parrot, repetition,  │    │
│  │     too_long)                                                    │    │
│  │       └─ if hit, return immediately with negative reward          │    │
│  │   ③ Count tokens with the Qwen tokenizer                         │    │
│  │   ④ Call scorer.score_answer(...)                                │    │
│  │       └─ which calls judge.get_judge().score_correctness(...)    │    │
│  │           └─ HF Inference → meta-llama/Llama-3.1-8B-Instruct     │    │
│  │   ⑤ Update recent_rewards deque + curriculum phase               │    │
│  │   ⑥ Return Observation { reward, reward_components, phase, … }   │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  4. Trainer gets reward → GRPO loss → backprop → update LLM weights      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. The action contract

The model **must** emit a single string of this exact shape:

```
<budget>N</budget><answer>X</answer>
```

- `N` is a positive integer — the model's *self-predicted upper bound* on tokens it will use.
- `X` is the answer text.
- The two tags must both be present and in this order.
- Anything else is treated as malformed → `bad_format` cliff.

The action is wrapped in a Pydantic schema in `models.py`:

```python
class TokenEfficiencyAction(BaseModel):
    raw_response: str
```

Why a single string and not separate `budget: int` / `answer: str` fields? Because we want the *LLM itself* to learn the format. Splitting it into structured fields would let the trainer pre-parse and side-step the format-quality reward.

---

## 5. The 6-component reward (canonical)

Defined in `scorer.py::score_answer`. Weights:

| # | Component | Weight | What it measures |
|---|---|---|---|
| 1 | **correctness** | **0.55** | LLM judge score in `{0.0, 0.3, 0.7, 1.0}`. Falls back to keyword overlap if the judge errors. |
| 2 | **efficiency** | 0.15 | `tokens_used` vs the *complexity-ideal* token count (easy=15, medium=60, hard=130). Caps at 1.0 for `tokens_used <= ideal`; declines linearly past it. |
| 3 | **self_assessment** | 0.15 | Asymmetric. Mild penalty for slack (`budget >> tokens_used`); steep penalty for overshoot (`tokens_used > budget`). Encourages an *honest upper-bound* prediction, not just a high one. |
| 4 | **redundancy** | 0.05 | `1 - repeated_word_ratio`. Catches "Paris Paris Paris…". |
| 5 | **keyword_verification** | 0.05 | Pure-Python keyword overlap from `prompts.py`. A sanity floor in case the LLM judge hallucinates. |
| 6 | **format_quality** | 0.05 | Both tags present, budget is a clean positive int, well-ordered. |

Final reward (clamped to `[-1.0, 1.0]`):

```
reward = 0.55·correctness
       + 0.15·efficiency
       + 0.15·self_assessment
       + 0.05·redundancy
       + 0.05·keyword_verification
       + 0.05·format_quality
```

### Why these specific changes from the original 7-component design?

- **Dropped `budget_reasonableness`** (which compared the model's allocated budget to a per-complexity prior). It rewarded *memorising fixed priors*, not actual learning, and double-dipped with `efficiency`.
- **Made `efficiency` absolute** (vs an absolute ideal token count) instead of *relative to the model's own budget*. Otherwise the model could trivially max efficiency by predicting a tiny budget and undershooting.
- **Made `self_assessment` asymmetric**. The original symmetric formula rewarded the model for *using exactly its budget*, which directly contradicted `efficiency`'s "use less" signal. The new shape mildly punishes slack but heavily punishes overshoot, lining up with the real-world objective of "honest upper-bound prediction."

---

## 6. Anti-hacking cliffs

These short-circuit the 6-component formula. Defined in `token_efficiency_env_environment.py::step`.

| Trigger | `error` tag | `reward` | Why it exists |
|---|---|---|---|
| Missing/malformed `<budget>` or `<answer>` tag | `bad_format` | `-1.0` | Forces format compliance from step 1. |
| `len(answer) < 2` chars (e.g. `.`, `a`, ` `) | `empty` | `-1.0` | Closes the "score 1.0 by saying almost nothing" loophole. |
| Normalised question is a substring of the answer | `parrot` | `-0.5` | Closes the "repeat the question = high keyword overlap" loophole. |
| A single word covers >60% of the answer | `repetition` | `-0.5` | Stops degenerate "the the the …" outputs. |
| `tokens_used > 500` (`MAX_ANSWER_TOKENS`) | `too_long` | `-0.5` | Hard ceiling on response length (the advertised `episode_token_limit=200` is the *soft* target — the inefficiency is paid through the `efficiency` component before this cliff trips). |

When a cliff fires, `reward_components` is returned as `{}` and `done=True`.

---

## 7. Curriculum learning

The env keeps a rolling deque of the last 50 episode rewards. The mean of that window decides which difficulty distribution to sample from on the next `reset()`.

| Phase | easy | medium | hard | promotes when |
|---|---|---|---|---|
| 1 | 100% | 0% | 0% | `avg_reward_50 ≥ 0.40` |
| 2 | 60% | 40% | 0% | `≥ 0.50` |
| 3 | 30% | 40% | 30% | `≥ 0.60` |
| 4 | 20% | 40% | 40% | terminal |

This is implemented in `TokenEfficiencyEnvironment._select_phase()` and is purely server-side state — **the trainer doesn't need to know about it**.

---

## 8. The two LLMs (don't confuse them)

| Model | Role | Trained? | Where it runs |
|---|---|---|---|
| **`Qwen/Qwen2.5-3B-Instruct`** | The *trainee*. Reads the prompt, emits `<budget>…</budget><answer>…</answer>`. | **Yes** — its weights change during Phase 7. | On the trainer's GPU. The env *also* loads this model's tokenizer (read-only) for accurate token counting. |
| **`meta-llama/Llama-3.1-8B-Instruct`** | The *judge*. Given (prompt, answer, expected_keywords), returns a correctness score in `{0.0, 0.3, 0.7, 1.0}`. | **No** — frozen, just inference. | HuggingFace Inference Providers (free tier, requires `HF_TOKEN`). |

Override via env vars:

```
JUDGE_BACKEND=huggingface          # or "keyword" to disable HF
JUDGE_MODEL=meta-llama/Llama-3.1-8B-Instruct   # any HF-supported chat model
HF_TOKEN=hf_xxx                    # required for huggingface backend
```

If `HF_TOKEN` is missing or the API errors out, the judge falls back to pure keyword overlap automatically. Training degrades but doesn't crash.

---

## 9. Statefulness gotcha — HTTP vs WebSocket

This caught us in Phase 4 and is worth knowing up front.

OpenEnv's `create_app(...)` exposes the env over **two** transports:

- **HTTP** (`POST /reset`, `POST /step`): **stateless**. Each request may be served by a *different* `Environment` instance from a pool of size `max_concurrent_envs`. There is no session affinity. Sending `/reset` followed by `/step` over raw HTTP is **not safe** — they may hit different instances and you'll be scored on a question you weren't asked.
- **WebSocket** (`/ws`, used by `client.TokenEfficiencyEnv`): **stateful**. The connection holds one env instance for its lifetime; `reset` / `step` calls are guaranteed to share state.

**Consequences:**

1. `server/app.py` pins `max_concurrent_envs=1` so even raw-HTTP usage works correctly within a single server process. Don't change this without reading the comment.
2. For parallel-rollout GRPO, run **N separate server processes on N ports** rather than bumping `max_concurrent_envs`.
3. Always prefer the typed `TokenEfficiencyEnv` client over hand-rolled HTTP requests. The `_phase4_e2e.py` script demonstrates this.

---

## 10. Repository layout (current)

```
token-efficiency-env/
├── ARCHITECTURE.md                ← you are here
├── README.md                      ← stale, ignore until Phase 6
├── web_dashboard.py               ← stale local Gradio dashboard, ignore until Phase 6
├── dashboard.html                 ← partner of web_dashboard.py, ignore until Phase 6
├── _phase3_smoketest.py           ← reward formula + cliff regression script
├── _phase4_e2e.py                 ← end-to-end WebSocket round-trip script
└── token_efficiency_env/
    ├── README.md                  ← stale, ignore until Phase 6
    ├── pyproject.toml             ← deps: openenv-core, transformers, huggingface_hub, pydantic
    ├── openenv.yaml               ← deploy manifest pointing at the canonical entry point
    ├── __init__.py
    ├── prompts.py                 ← 24 questions × {complexity, keywords}
    ├── scorer.py                  ← 6-component reward function (canonical)
    ├── judge.py                   ← Judge protocol + HFInferenceJudge + KeywordJudge + factory
    ├── models.py                  ← TokenEfficiencyAction / TokenEfficiencyObservation
    ├── client.py                  ← typed WebSocket client (use this in trainers)
    ├── env_client.py              ← thin alias re-exporting client.py (back-compat)
    ├── env_server.py              ← thin alias re-exporting server env (back-compat)
    └── server/
        ├── app.py                 ← FastAPI wrapper (max_concurrent_envs=1, see comment)
        ├── token_efficiency_env_environment.py   ← THE ENVIRONMENT (canonical impl)
        ├── requirements.txt       ← deps used by the Docker image
        └── Dockerfile             ← used by HF Spaces deploy in Phase 5
```

---

## 11. Running locally

### Prereqs

- Python 3.10+
- A HuggingFace account + token with **"Make calls to Inference Providers"** permission. Get one at <https://huggingface.co/settings/tokens>.

### Install

```powershell
cd token-efficiency-env
pip install -e ./token_efficiency_env
```

### Run the server (with the Gradio web UI mounted at `/web/`)

PowerShell:

```powershell
$env:HF_TOKEN = 'hf_yourtokenhere'
$env:JUDGE_BACKEND = 'huggingface'
$env:ENABLE_WEB_INTERFACE = 'true'
python -m uvicorn token_efficiency_env.server.app:app --host 127.0.0.1 --port 8000
```

bash:

```bash
HF_TOKEN=hf_yourtokenhere JUDGE_BACKEND=huggingface ENABLE_WEB_INTERFACE=true \
  python -m uvicorn token_efficiency_env.server.app:app --host 127.0.0.1 --port 8000
```

### Open it

| URL | What it is |
|---|---|
| <http://localhost:8000/web/> | Gradio playground (Reset → type `<budget>…</budget><answer>…</answer>` → Step) |
| <http://localhost:8000/docs> | FastAPI Swagger UI (manual `/reset`, `/step` calls) |
| <http://localhost:8000/health> | `{"status": "healthy"}` |
| <http://localhost:8000/metadata> | Env name, version, description |

### Drive it from Python (the way the trainer will)

```python
from token_efficiency_env.client import TokenEfficiencyEnv
from token_efficiency_env.models import TokenEfficiencyAction

with TokenEfficiencyEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()
    print("Q:", obs.prompt, "(limit:", obs.episode_token_limit, "tokens)")

    result = env.step(TokenEfficiencyAction(
        raw_response="<budget>5</budget><answer>Paris</answer>"
    ))
    print("Reward:", result.reward)
    print("Components:", result.observation.reward_components)
    print("Phase:", result.observation.phase, "| Episode:", result.observation.episode)
```

### Run the regression scripts

```powershell
# Reward formulas + anti-hacking cliffs (no server required)
python _phase3_smoketest.py

# End-to-end round-trip (requires server running on :8000)
python _phase4_e2e.py
```

---

## 12. Status & roadmap

> **Note on ordering:** Phase 5 (HF Spaces deploy) is intentionally being done **last**, after Phases 6 and 7. The training notebook in Phase 7 can drive a local `http://localhost:8000` server during development, so deployment isn't blocking; doing it last also lets us mint the production `HF_TOKEN` only once.

| Phase | Goal | Status |
|---|---|---|
| 1 | Move env logic into the OpenEnv-shaped `server/token_efficiency_env_environment.py`; wire schemas | ✅ done |
| 2 | Replace Anthropic with HuggingFace Inference judge; add keyword fallback | ✅ done |
| 3 | Reward redesign (6 components, asymmetric self-assessment, absolute efficiency); new anti-hacking cliffs | ✅ done |
| 4 | Local end-to-end smoke test (HTTP + WebSocket) | ✅ done |
| 6 | Rewrite both `README.md`s; remove or modernise `web_dashboard.py`; add a `CHANGELOG` | ⏳ next |
| 7 | Training notebook scaffold (TRL `GRPOTrainer` + Qwen2.5-3B + N parallel env servers) | ⏳ pending |
| 5 | Deploy as a HuggingFace Space (Docker SDK) — **deferred to last** | ⏳ pending |

---

## 13. Glossary

- **Episode**: one `reset` → `step` cycle = one question/answer pair.
- **Budget (`N`)**: the model's *self-predicted* upper bound on its own token usage. Not a constraint imposed by the env, just a prediction the env scores.
- **Episode token limit**: a *soft* target (default 200 tokens) advertised to the trainer. Going over costs reward through the `efficiency` component but does *not* terminate the episode. The actual hard cliff is `MAX_ANSWER_TOKENS=500` → `too_long`.
- **Cliff**: a short-circuit reward path that bypasses the 6-component formula. Used to make reward-hacking unprofitable.
- **Curriculum phase**: which difficulty mix the env is currently sampling from. Server-side state, advances on rolling-50 average reward.
- **Trainee**: the LLM whose weights are being updated (Qwen2.5-3B-Instruct).
- **Judge**: the frozen LLM that scores correctness (Llama-3.1-8B-Instruct via HF Inference).
- **GRPO**: Group Relative Policy Optimization — the RL algorithm we plan to use in Phase 7. It needs multiple parallel rollouts of the same prompt, which is why we'll run multiple env server processes.
