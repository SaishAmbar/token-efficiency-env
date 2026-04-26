---
title: Token Efficiency Playground
emoji: ✂️
colorFrom: pink
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: bsd-3-clause
short_description: Interactive playground for TokenEfficiencyEnv rewards.
tags:
  - openenv
  - reinforcement-learning
  - llm
  - token-efficiency
  - grpo
  - meta-hackathon
  - playground
---

# Token Efficiency Playground

> A **light-theme, browser-based playground** for [`TokenEfficiencyEnv`](https://github.com/SaishAmbar/token-efficiency-env) — the OpenEnv RL environment that teaches LLMs to answer correctly using fewer tokens.

This Space is the **demo / showcase** counterpart to the headless OpenEnv
server. It runs the *exact same* scoring code path the GRPO trainer uses,
so the rewards you see here are the rewards the model would receive
during training.

---

## What you can do

### 1. Interactive Mode

Pick a question from the curriculum, type a budget and an answer, hit
**Submit & Score**. You'll see:

- The **final weighted reward** (one big peach number).
- A breakdown of all six components — Correctness, Efficiency,
  Self-Assessment, Redundancy, Keyword Verification, Format Quality —
  with their weights and individual scores.
- Diagnostics: the prompt the answer was scored against, the budget you
  declared, the raw token count of the full `<budget>...</budget><answer>...</answer>`
  envelope, the inner-answer token count, and the running average reward.

The structured **Budget** and **Answer** inputs are wrapped into the
required XML envelope client-side before being sent to the server, so
you never have to type angle brackets.

### 2. Training Simulation

Pick `N` (50–500) episodes and watch a fake "trainee" whose skill ramps
from 0 → 1 get scored by the same env. The reward chart that appears is
representative of how a real GRPO loop converges. The keyword judge is
forced for this tab so it stays fast and offline regardless of whether
`HF_TOKEN` is set.

---

## How the reward works

```
final_reward = 0.55 * correctness            (LLM judge or keyword fallback)
             + 0.15 * efficiency             (vs complexity ideal: 27/72/142)
             + 0.15 * self_assessment        (asymmetric: over-promise hurts ~4× more)
             + 0.05 * redundancy             (unique-word ratio + bigram penalty)
             + 0.05 * keyword_verification   (expected stems found in answer)
             + 0.05 * format_quality         (clean structure)
```

**Anti-hacking cliffs** (applied *before* the 6 components):
`bad_format`, `empty`, `parrot`, `repetition`, and `too_long` all return
a hard `-1.0` reward, bypassing the gentle weighted sum entirely.

---

## Judge backend

The Space tries two judges, in order:

1. **HF Inference Provider** (`Llama-3.1-8B-Instruct` by default) — used
   if the Space has an `HF_TOKEN` secret with read scope. This is the
   "real" reviewer-grade judge.
2. **KeywordJudge** — offline keyword-stem matcher, used as a fallback
   so the Space stays fully functional even on a fresh deploy with no
   secret configured.

To enable the LLM judge:

1. Go to your Space → **Settings** → **Variables and secrets**.
2. Add a secret called `HF_TOKEN` with a token that has `read` scope.
3. The Space will pick it up on next restart — no rebuild needed.

You can verify which judge is live by hitting `/api/state` from your
browser; the JSON response includes a `judge` field naming the active
class.

---

## Useful endpoints

| Endpoint | Purpose |
|---|---|
| `/` | The interactive playground UI (this is the landing page). |
| `/api/questions` | List the 24 starter prompts with their complexity and expected keywords. |
| `/api/reset` | Sample a fresh question. Pass `?question_index=N` to pick a specific one. |
| `/api/step` | Submit `{prompt, raw_response}` for scoring. The UI calls this under the hood. |
| `/api/simulate` | Run a batch sweep with the fake trainee. |
| `/api/state` | Episode count, current curriculum phase, average reward, active judge class. |

---

## Repo & docs

- **Source code:** https://github.com/SaishAmbar/token-efficiency-env
- **Architecture:** See `ARCHITECTURE.md` in the repo for the full design
  (curriculum, anti-hacking guards, reward derivation).
- **Hackathon alignment:** See `docs/audit/HACKATHON_ALIGNMENT_REPORT.md`
  for the engineering self-audit (scored 9.5/10 against Meta's OpenEnv
  hackathon brief).

---

*Built for the Meta OpenEnv hackathon. BSD-3-Clause licensed.*
