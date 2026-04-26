---
title: Token Efficiency Env
emoji: ✂️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
license: bsd-3-clause
tags:
  - openenv
  - reinforcement-learning
  - llm
  - token-efficiency
  - grpo
  - meta-hackathon
---

# TokenEfficiencyEnv

> **An OpenEnv RL environment that teaches LLMs to answer *correctly* using *fewer tokens*.**
> The model receives a question, self-allocates a token budget, and is scored on a 6-component reward with honest anti-hacking cliffs.

This Space is the environment **server**. To use it, point your
[EnvClient](https://github.com/meta-pytorch/OpenEnv) at this URL — no model is hosted here; a judge (Qwen / Llama-3.1-8B via HF Inference) scores the correctness of whatever answer your agent returns.

---

## Quick links

- **Repo:** https://github.com/SaishAmbar/token-efficiency-env
- **Wire contract:** `<budget>N</budget><answer>text</answer>`
- **Interactive web UI:** [`/web`](./web) — poke the environment without writing code.
- **Health check:** [`/health`](./health)
- **OpenAPI docs:** [`/docs`](./docs)

---

## 30-second pitch

Large models are over-verbose. That's the core token-efficiency problem:
even for a question with a one-word answer, a naive `Qwen2.5-3B` will
emit 80+ tokens of preamble. Every one of those tokens costs compute,
latency, and — at scale — money.

`TokenEfficiencyEnv` is an RL environment that makes concision *learnable*:

1. The agent sees a prompt.
2. The agent responds with `<budget>N</budget><answer>text</answer>` —
   where `N` is its **own prediction** of how many tokens the answer will take.
3. The environment scores:
   - **Correctness** (40%) — LLM judge (Llama-3.1-8B-Instruct) rates factual accuracy.
   - **Efficiency** (25%) — fewer raw response tokens → higher reward, continuous.
   - **Self-assessment** (15%) — does `N` match the actual token count? Calibration matters.
   - **Redundancy** (5%) — penalises repetition within the answer.
   - **Keyword verification** (5%) — sanity check the answer contains expected terms.
   - **Format quality** (10%) — clean structure, no embedded double-spaces.
4. And we *cliff* (hard -0.5 to -1.0 reward) on 5 known exploits:
   `bad_format`, `empty`, `parrot`, `repetition`, `too_long`.

No chain-of-thought leakage possible: tokens are counted over the
**raw response**, not just the inner `<answer>`, so hidden scratch-pad
reasoning doesn't cheat the efficiency score.

---

## What an agent sees

**Action schema** (`TokenEfficiencyAction`):
```json
{ "raw_response": "<budget>5</budget><answer>Paris.</answer>" }
```

**Observation schema** (`TokenEfficiencyObservation`):
```json
{
  "reward": 0.82,
  "reward_components": {
    "correctness": 1.0,
    "efficiency": 1.0,
    "self_assessment": 1.0,
    "redundancy": 1.0,
    "keyword_verification": 1.0,
    "format_quality": 1.0
  },
  "allocated_budget": 5,
  "tokens_used": 12,
  "answer_token_count": 2,
  "complexity": "easy",
  "error": "",
  "done": true
}
```

---

## Usage — from a trainer

```python
from token_efficiency_env import TokenEfficiencyEnv, TokenEfficiencyAction

async with TokenEfficiencyEnv(base_url="https://<your-space>.hf.space") as env:
    obs = await env.reset()
    result = await env.step(TokenEfficiencyAction(
        raw_response="<budget>5</budget><answer>Paris.</answer>"
    ))
    print(result.reward, result.reward_components)
```

Or via the sync wrapper, or via direct HTTP `/reset` + `/step`. See the
[repo README](https://github.com/SaishAmbar/token-efficiency-env) for
full examples.

---

## Server environment variables

| Variable            | Default                             | Purpose |
|---------------------|-------------------------------------|---------|
| `JUDGE_MODEL`       | `meta-llama/Llama-3.1-8B-Instruct`  | HF Inference model for the correctness judge. |
| `HF_TOKEN`          | (unset)                             | Required by the HF judge; falls back to `KeywordJudge` if missing. |
| `PROMPT_BANK_MODE`  | `starter`                           | `starter` (24 curated prompts) or `full` (~2.3k programmatic from GSM8K + TriviaQA + ARC + OpenOrca). |
| `MAX_CONCURRENT_ENVS` | `1`                               | **Keep at 1** — state (curriculum, recent_rewards) must persist across requests. Scale with multiple Spaces, not multiple workers. |

Set them via Space **Settings → Variables and secrets**.

---

## Training pipeline

This Space hosts only the env server. The matching **training scaffold**
(GRPO with Qwen2.5-3B + LoRA, Colab-ready one-click run) lives in the
repo: [`notebooks/train_grpo.ipynb`](https://github.com/SaishAmbar/token-efficiency-env/blob/main/notebooks/train_grpo.ipynb).
Click the "Open in Colab" badge in the repo README to reproduce end-to-end.

---

## Built for the Meta OpenEnv hackathon (2026)

Honest scoring; reward-hacking closed at v0.3.0 (see
[`CHANGELOG.md`](https://github.com/SaishAmbar/token-efficiency-env/blob/main/CHANGELOG.md)
for the 7-of-8 vulnerability audit closure).
