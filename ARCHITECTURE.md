# TokenEfficiencyEnv — Architecture & Onboarding

> **Read this file first.** This is the canonical design document. The
> root `README.md` is the friendly executive summary; this file is where
> the reward maths, anti-hacking cliffs, statefulness gotcha, and file
> map live.

---

## 1. TL;DR

`TokenEfficiencyEnv` is a [Meta OpenEnv](https://github.com/meta-pytorch/OpenEnv)–compatible RL environment. Its job is to teach a small language model (target: `Qwen/Qwen2.5-3B-Instruct`) to answer questions **correctly using as few tokens as possible**, by forcing the model to first *predict* its own token budget and then live within it.

Each episode is one question. The model must output **exactly** this string:

```
<budget>N</budget><answer>your answer here</answer>
```

The env then computes a scalar reward in roughly `[-1.0, +1.0]` from a 6-component scorer and returns it through the OpenEnv HTTP/WebSocket interface. A trainer (`notebooks/train_grpo.ipynb` + `training/reward_adapter.py`) uses those rewards to do GRPO updates on the LLM.

**Status:** all implementation phases (1 – 8) are complete. The v0.3.0
release closed the hidden-CoT loophole (strict-tag parser, raw-response
token pricing, diagnostic `answer_token_count`). The HF Space is live.
The only open items are operational: a 300-step full GRPO run on Colab,
and a mid-training sampling callback. See `docs/audit/HACKATHON_ALIGNMENT_REPORT.md` §24.

---

## 2. Suggested reading order for new contributors

Read these files in this order. Skip nothing in the first three; skim the rest.

| # | File | Why |
|---|---|---|
| 1 | **`ARCHITECTURE.md`** (this file) | The map. |
| 2 | `token_efficiency_env/server/token_efficiency_env_environment.py` | The actual environment. `reset()` / `step()`, curriculum logic, anti-hacking guards, and the v0.3.0 `SHELL_RE` strict parser all live here. |
| 3 | `token_efficiency_env/scorer.py` | The 6-component reward function. The whole training signal comes from here. |
| 4 | `token_efficiency_env/judge.py` | The pluggable LLM judge (HuggingFace Inference, with keyword fallback and out-of-range rejection). |
| 5 | `token_efficiency_env/prompts.py` + `prompt_bank_loader.py` | 24 hand-curated starter prompts + optional 2.3k programmatic bank from GSM8K / TriviaQA / ARC / OpenOrca. |
| 6 | `token_efficiency_env/models.py` | The `TokenEfficiencyAction` / `TokenEfficiencyObservation` Pydantic schemas — the wire contract. Includes the v0.3.0 diagnostic `answer_token_count` field. |
| 7 | `token_efficiency_env/server/app.py` | Tiny FastAPI wrapper that mounts the environment as an OpenEnv server. Read the comment about `max_concurrent_envs`. |
| 8 | `token_efficiency_env/client.py` | The typed Python client a trainer uses to talk to the server over WebSockets. |
| 9 | `token_efficiency_env/openenv.yaml` | OpenEnv deployment manifest (entry point + schema paths). |
| 10 | `training/reward_adapter.py` | The bridge from TRL's `reward_funcs` API to `TokenEfficiencyEnvironment.step()`. Default in-process backend + optional WebSocket backend. |
| 11 | `notebooks/train_grpo.ipynb` | The Colab-ready GRPO scaffold: bootstrap → config banner → split → SFT warmup → GRPO → eval → before/after report. |
| 12 | `docs/audit/` | The reasoning trail: reward-hacking audit, fix plan, comparative analysis, hackathon self-review. Read these for the *why*. |

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
| 2 | **efficiency** | 0.15 | `tokens_used` vs the *complexity-ideal* token count (easy=27, medium=72, hard=142 — bumped in v0.3.0 to absorb the 10–12 wrapper tokens now priced). Caps at 1.0 for `tokens_used <= ideal`; declines linearly past it. `tokens_used` counts the **full raw response**, including any CoT that tried to sneak in before `<budget>`. |
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
- **v0.3.0: `tokens_used` counts the full raw response, not just the inner `<answer>`.** Combined with the strict `SHELL_RE` parser (§6) this closes the hidden-chain-of-thought loophole: the model can't pad with 500 tokens of scratch reasoning before `<budget>` and pay only for "Paris." inside `<answer>`. The `COMPLEXITY_IDEAL_TOKENS` values above were bumped by +12 in the same release to absorb the wrapper-token cost now being priced. See `CHANGELOG.md` [0.3.0] §1 and `tests/test_anti_cliff.py` for coverage.

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
├── README.md                      ← executive summary + quick-start
├── CHANGELOG.md                   ← per-release design rationale
├── pytest.ini
├── requirements-train.txt         ← trl / peft / accelerate / datasets / num2words
├── web_dashboard.py               ← local Gradio UI (delegates to the real env)
├── dashboard.html
├── docs/
│   └── audit/                     ← reward-hacking audit, fix plan, hackathon review
├── deploy/
│   ├── push_to_hf_space.py        ← HfApi-based deploy helper (no openenv CLI needed)
│   ├── SPACE_README.md            ← judge-facing README pasted onto the Space landing page
│   └── README.md                  ← operator guide
├── notebooks/
│   ├── train_grpo.ipynb           ← Colab-ready GRPO training scaffold
│   └── eval_baseline_vs_trained.py
├── training/
│   ├── config.py                  ← TrainingConfig (smoke + full presets)
│   ├── prompts_split.py           ← stratified 85/10/5 train/holdout/probe
│   ├── reward_adapter.py          ← InProcessRewardAdapter + WSRewardAdapter
│   ├── sft_format_data.py         ← 50-example SFT format warmup
│   ├── server_pool.py             ← uvicorn worker pool for the WS path
│   ├── colab_bootstrap.py         ← is_colab() + bootstrap() for the notebook
│   └── compare_report.py          ← before/after markdown + plots
├── scripts/
│   ├── demo_session.py
│   ├── smoke_prompt_bank.py
│   └── smoke_compare_report.py
├── tests/                         ← 147 passing, 7 skipped (WebSocket e2e)
└── token_efficiency_env/
    ├── README.md                  ← in-package contributor guide
    ├── pyproject.toml             ← deps: openenv-core, transformers, huggingface_hub, pydantic
    ├── openenv.yaml               ← deploy manifest pointing at the canonical entry point
    ├── __init__.py
    ├── prompts.py                 ← 24 starter questions + dispatch to full-bank loader
    ├── prompt_bank_loader.py      ← 2.3k programmatic bank (GSM8K + TriviaQA + ARC + OpenOrca)
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
        └── Dockerfile             ← used by HF Spaces deploy
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

### Run the regression tests

```powershell
# Reward formulas + anti-hacking cliffs (no server, no HF_TOKEN required)
pytest tests/ -m "not slow"

# End-to-end round-trip (requires server running on :8000)
pytest tests/test_e2e_websocket.py
```

---

## 12. Status & roadmap

All implementation phases are complete. The CoT-hardening release
(v0.3.0) closed the final critical vulnerability, and the HF Space was
deployed immediately afterwards. Remaining work is operational, not
architectural, and is tracked in
`docs/audit/HACKATHON_ALIGNMENT_REPORT.md` §24.

| Phase | Goal | Status |
|---|---|---|
| 1 | Move env logic into the OpenEnv-shaped `server/token_efficiency_env_environment.py`; wire schemas | ✅ done |
| 2 | Replace Anthropic with HuggingFace Inference judge; add keyword fallback | ✅ done |
| 3 | Reward redesign (6 components, asymmetric self-assessment, absolute efficiency); anti-hacking cliffs | ✅ done |
| 4 | Local end-to-end smoke test (HTTP + WebSocket) | ✅ done |
| 5 | Deploy as a HuggingFace Space (Docker SDK) | ✅ done |
| 6 | Rewrite READMEs, modernise dashboard, add CHANGELOG, promote smoke tests | ✅ done |
| 7 | Training notebook scaffold (TRL `GRPOTrainer` + Qwen2.5-3B + LoRA + held-out eval) | ✅ done |
| 8 | Programmatic 2.3k-prompt bank (GSM8K + TriviaQA + ARC + OpenOrca), stratified split, SFT warmup, smoke preset, Colab bootstrap | ✅ done |
| v0.3.0 | Hidden-CoT fix (strict `SHELL_RE`, raw-response token pricing, `answer_token_count` diagnostic) | ✅ done |

### Phase 7 layout (added in this phase)

```
training/
  config.py             # TrainingConfig dataclass — single source of truth
  prompts_split.py      # deterministic train/holdout split of PROMPT_BANK
  server_pool.py        # spawn / health-check / teardown N uvicorn workers
  reward_adapter.py     # InProcessRewardAdapter (default) + WSRewardAdapter
notebooks/
  train_grpo.ipynb              # main scaffold (16 cells, top-to-bottom)
  eval_baseline_vs_trained.py   # standalone before/after eval harness
requirements-train.txt          # trl + peft + accelerate + datasets
tests/test_training_scaffold.py # offline smoke tests (config / split / adapter)
```

The reward adapter has **two backends** chosen by `TrainingConfig.reward_backend`:
* `"in_process"` (default) — instantiates `TokenEfficiencyEnvironment` directly in the trainer's process. Fast, no servers, no ports. This is what GRPO actually wants because it sets the prompt itself; we just hand the env the trainer-chosen `current_task` before each `step()`.
* `"ws"` — drives a real `ServerPool` over the WebSocket client. Optional, useful as a stress test of the deployed server path. Heavier per-call.

Both expose the same `__call__(prompts, completions) -> list[float]`, so the notebook flips between them with one config flag.

---

## 13. Glossary

- **Episode**: one `reset` → `step` cycle = one question/answer pair.
- **Budget (`N`)**: the model's *self-predicted* upper bound on its own token usage. Not a constraint imposed by the env, just a prediction the env scores.
- **Episode token limit**: a *soft* target (default 200 tokens) advertised to the trainer. Going over costs reward through the `efficiency` component but does *not* terminate the episode. The actual hard cliff is `MAX_ANSWER_TOKENS=500` → `too_long`.
- **Cliff**: a short-circuit reward path that bypasses the 6-component formula. Used to make reward-hacking unprofitable.
- **Curriculum phase**: which difficulty mix the env is currently sampling from. Server-side state, advances on rolling-50 average reward.
- **Trainee**: the LLM whose weights are being updated (Qwen2.5-3B-Instruct).
- **Judge**: the frozen LLM that scores correctness (Llama-3.1-8B-Instruct via HF Inference).
- **GRPO**: Group Relative Policy Optimization — the RL algorithm used in `notebooks/train_grpo.ipynb`. It needs multiple parallel rollouts of the same prompt; when using the WS reward backend we run multiple env server processes to avoid curriculum-state contention.
