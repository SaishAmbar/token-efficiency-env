# TokenEfficiencyEnv

> An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible RL environment that trains LLMs to answer questions correctly using **fewer tokens**, by forcing the model to first *predict* its own token budget and then live within it.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3--Clause-yellow)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaishAmbar/token-efficiency-env/blob/main/notebooks/train_grpo.ipynb)

> **Don't have a local GPU?** Click the **Open In Colab** badge above to run the full GRPO training notebook on a free T4 — the first cell installs everything, prompts for your `HF_TOKEN`, and takes you straight to a runnable trainer. No local setup needed.

> **Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first** for the full design (data flow, reward maths, anti-hacking cliffs, statefulness gotcha, file map). This README is the friendly executive summary.
>
> Looking for the **reward-hacking audit trail**, the fix plan, or the hackathon self-assessment? See [`docs/audit/`](docs/audit/).

---

## What it does

Standard LLMs over-explain — a one-word answer often comes wrapped in three sentences of throat-clearing. This environment makes the model **predict its own answer length** before answering, and rewards it for being both **correct AND tight**.

Each episode the model receives a question and must reply in *exactly* this format:

```
<budget>N</budget><answer>your answer here</answer>
```

- `N` = the model's self-predicted upper bound on tokens it will use
- The answer must actually live within that budget
- The reward is computed by a 6-component scorer (see below)

A separate trainer (e.g. TRL's `GRPOTrainer`) collects rewards across thousands of episodes and updates the model's weights to make it both more accurate and more concise.

---

## Architecture at a glance

```
┌────────────────────────────────────────────────────────────────────┐
│                         ONE TRAINING EPISODE                       │
│                                                                    │
│  env.reset()  ──►  picks a question (curriculum-weighted)         │
│                    returns: { prompt, episode_token_limit, ... }   │
│                                                                    │
│  LLM generates:  "<budget>5</budget><answer>Paris</answer>"        │
│                                                                    │
│  env.step(action)  ──►  parse tags                                 │
│                          anti-hacking cliffs                       │
│                          count tokens (Qwen tokenizer)             │
│                          score correctness (Llama judge via HF)    │
│                          combine 6 reward components               │
│                          returns: observation + reward             │
│                                                                    │
│  Trainer (GRPO) uses reward to update LLM weights                  │
└────────────────────────────────────────────────────────────────────┘
```

Two LLMs in the loop (don't confuse them):

| Model | Role | Trained? |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | The **trainee** | Yes — its weights change |
| `meta-llama/Llama-3.1-8B-Instruct` | The **judge** for correctness | No — frozen, runs on HF Inference Providers |

---

## The 6-component reward

| # | Component | Weight | What it measures |
|---|---|---:|---|
| 1 | **correctness** | 0.55 | LLM judge in `{0.0, 0.3, 0.7, 1.0}`. Falls back to keyword overlap on judge errors. |
| 2 | **efficiency** | 0.15 | `tokens_used` vs an *absolute* per-complexity ideal (easy=27, medium=72, hard=142 — bumped in v0.3.0 to absorb the 10–12 wrapper tokens now being priced). Caps at 1.0; declines past the ideal. |
| 3 | **self_assessment** | 0.15 | Asymmetric — mild penalty for slack, steep penalty for overshoot. Trains *honest upper-bound* prediction. |
| 4 | **redundancy** | 0.05 | `1 - repeated_word_ratio`. Catches "Paris Paris Paris…". |
| 5 | **keyword_verification** | 0.05 | Pure-Python keyword overlap. A sanity floor against judge hallucinations. |
| 6 | **format_quality** | 0.05 | Both tags present, budget is a positive int, well-ordered. |

**Anti-hacking cliffs** (override the formula entirely):

| Trigger | `error` tag | `reward` |
|---|---|---:|
| Missing/malformed `<budget>` or `<answer>` | `bad_format` | `-1.0` |
| Answer < 2 chars (e.g. `.`, `a`) | `empty` | `-1.0` |
| Answer is the question parroted back | `parrot` | `-0.5` |
| Single word covers >60% of the answer | `repetition` | `-0.5` |
| `tokens_used > 500` (the `MAX_ANSWER_TOKENS` ceiling) | `too_long` | `-0.5` |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) §5–6 for the *why* behind each design choice.

---

## Curriculum

The env tracks the trainee's average reward over the last 50 episodes. Once the average crosses a threshold, the difficulty mix shifts:

| Phase | easy | medium | hard | promotes when |
|---|---:|---:|---:|---|
| 1 | 100% | 0% | 0% | `avg_reward_50 ≥ 0.40` |
| 2 | 60% | 40% | 0% | `≥ 0.50` |
| 3 | 30% | 40% | 30% | `≥ 0.60` |
| 4 | 20% | 40% | 40% | terminal |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/SaishAmbar/token-efficiency-env.git
cd token-efficiency-env
pip install -e ./token_efficiency_env
```

### 2. Get a HuggingFace token

Create one at <https://huggingface.co/settings/tokens> with **"Make calls to Inference Providers"** permission. Then:

```powershell
# PowerShell
$env:HF_TOKEN = 'hf_yourtokenhere'
$env:JUDGE_BACKEND = 'huggingface'
```

```bash
# bash / zsh
export HF_TOKEN=hf_yourtokenhere
export JUDGE_BACKEND=huggingface
```

> No HF token? Set `JUDGE_BACKEND=keyword` instead. The env will use offline keyword-overlap scoring — fine for development, weaker training signal.

### 3. Start the env server

```bash
ENABLE_WEB_INTERFACE=true \
  python -m uvicorn token_efficiency_env.server.app:app --host 127.0.0.1 --port 8000
```

| URL | What it is |
|---|---|
| <http://localhost:8000/web/> | Browser playground (Reset → type response → Step) |
| <http://localhost:8000/docs> | FastAPI Swagger UI |
| <http://localhost:8000/health> | Liveness probe |

### 4. Drive it from Python (the way a trainer does)

```python
from token_efficiency_env.client import TokenEfficiencyEnv
from token_efficiency_env.models import TokenEfficiencyAction

with TokenEfficiencyEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()
    print("Q:", obs.prompt)

    result = env.step(TokenEfficiencyAction(
        raw_response="<budget>5</budget><answer>Paris</answer>"
    ))
    print("Reward:", result.reward)
    print("Components:", result.observation.reward_components)
```

> **Always use `TokenEfficiencyEnv` (WebSocket), not raw HTTP.** The HTTP endpoints are stateless and curriculum/episode state lives in WebSocket sessions. See [`ARCHITECTURE.md`](ARCHITECTURE.md) §9.

---

## Tests

```bash
pytest tests/                  # fast unit suite, no server / token needed
# Integration tests in tests/test_e2e_websocket.py auto-skip if no server is up.
```

---

## Project layout

```
token-efficiency-env/
├── ARCHITECTURE.md             ← read this first (canonical design)
├── README.md                   ← you are here
├── CHANGELOG.md                ← release history + design rationale
├── pytest.ini                  ← test discovery config
├── requirements-train.txt      ← trl / peft / accelerate / datasets / num2words
├── web_dashboard.py            ← optional local Gradio UI (delegates to the real env)
├── dashboard.html
├── docs/
│   └── audit/                  ← reward-hacking audit, fix plan, hackathon self-review
├── deploy/
│   ├── push_to_hf_space.py     ← HfApi-based deploy (no openenv CLI needed)
│   ├── SPACE_README.md         ← judge-facing README served on the Space landing page
│   └── README.md               ← operator guide
├── notebooks/
│   ├── train_grpo.ipynb        ← Colab-ready GRPO training notebook (SFT warmup → GRPO → eval)
│   └── eval_baseline_vs_trained.py
├── training/
│   ├── config.py               ← TrainingConfig (smoke + full presets)
│   ├── prompts_split.py        ← stratified 85/10/5 train/holdout/probe
│   ├── reward_adapter.py       ← InProcessRewardAdapter (default) + WSRewardAdapter
│   ├── sft_format_data.py      ← 50-example SFT format warmup
│   ├── server_pool.py          ← spawn/teardown N uvicorn workers for WS backend
│   ├── colab_bootstrap.py      ← is_colab() + bootstrap() for the notebook cell 0
│   └── compare_report.py       ← before/after markdown + plots
├── scripts/
│   ├── demo_session.py         ← scripted baseline demo against local or live Space
│   ├── smoke_prompt_bank.py    ← one-command loader smoke for the 2.3k bank
│   └── smoke_compare_report.py ← regenerate the report from synthetic data (no GPU)
├── tests/                      ← 147 passing, 7 skipped (WebSocket e2e)
│   ├── test_reward_invariants.py
│   ├── test_anti_cliff.py       ← CoT-exploit rejection, parrot Jaccard, redundancy ramp
│   ├── test_prompt_bank_loader.py
│   ├── test_training_scaffold.py
│   ├── test_sft_format_data.py
│   ├── test_compare_report.py
│   ├── test_colab_bootstrap.py
│   ├── test_space_deploy_config.py
│   ├── test_demo_session.py
│   └── test_e2e_websocket.py
└── token_efficiency_env/
    ├── README.md               ← in-package contributor guide
    ├── pyproject.toml
    ├── openenv.yaml
    ├── prompts.py              ← 24 starter prompts + lazy loader dispatch
    ├── prompt_bank_loader.py   ← 2.3k programmatic bank (GSM8K + TriviaQA + ARC + OpenOrca)
    ├── scorer.py               ← 6-component reward
    ├── judge.py                ← HuggingFace + keyword judges
    ├── models.py               ← Pydantic schemas (incl. answer_token_count diagnostic)
    ├── client.py               ← WebSocket client (use this in trainers)
    └── server/
        ├── app.py
        ├── token_efficiency_env_environment.py   ← THE ENVIRONMENT
        ├── Dockerfile          ← used by HF Spaces deploy
        └── requirements.txt
```

---

## Roadmap

All implementation phases from the original plan are complete. The
remaining open items are operational (full-scale GRPO run) and
documentation (save-path guidance), tracked in
[`docs/audit/HACKATHON_ALIGNMENT_REPORT.md`](docs/audit/HACKATHON_ALIGNMENT_REPORT.md) §24.

- ✅ Phase 1 — OpenEnv-shaped environment + schemas
- ✅ Phase 2 — HuggingFace Inference judge + keyword fallback
- ✅ Phase 3 — Reward redesign (6 components, asymmetric self-assessment, absolute efficiency) + anti-hacking cliffs
- ✅ Phase 4 — Local end-to-end verification
- ✅ Phase 5 — Deploy as a HuggingFace Space (`deploy/push_to_hf_space.py` + live Space)
- ✅ Phase 6 — Docs & tests cleanup
- ✅ Phase 7 — Training notebook scaffold (TRL `GRPOTrainer` + Qwen2.5-3B + LoRA + held-out eval)
- ✅ Phase 8 — Programmatic 2.3k-prompt bank (GSM8K + TriviaQA + ARC + OpenOrca) + stratified split + SFT warmup + smoke preset + Colab bootstrap
- ✅ v0.3.0 — Hidden-CoT loophole closed (strict `SHELL_RE` parser, raw-response token pricing, `answer_token_count` diagnostic)

**Open, non-blocking:**
- ⏳ 300-step full GRPO run on Colab (50-step smoke is shipped).
- ⏳ Mid-training sampling callback (print N rollouts every K steps).
- ⏳ Judge adversarial stress-test suite.

See [`CHANGELOG.md`](CHANGELOG.md) for what landed in each release.

---

## License

BSD-3-Clause. See [`LICENSE`](LICENSE).
