# TokenEfficiencyEnv — Complete Project Guide

> An OpenEnv RL environment that trains LLMs to answer correctly using **fewer tokens**,
> by dynamically allocating a token budget based on task complexity.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [How the Environment Works (Big Picture)](#2-how-the-environment-works-big-picture)
3. [Folder Structure](#3-folder-structure)
4. [File-by-File Explanation](#4-file-by-file-explanation)
   - [env_server.py](#41-env_serverpy)
   - [env_client.py](#42-env_clientpy)
   - [prompts.py](#43-promptspy)
   - [scorer.py](#44-scorerpy)
   - [openenv.yaml](#45-openenvyaml)
5. [The Output Format Contract](#5-the-output-format-contract)
6. [The Scoring Formula](#6-the-scoring-formula)
7. [Setup — Step by Step](#7-setup--step-by-step)
8. [Testing the Environment Locally](#8-testing-the-environment-locally)
9. [Google Colab (P3) Setup](#9-google-colab-p3-setup)
10. [Environment Variables](#10-environment-variables)
11. [Common Errors and Fixes](#11-common-errors-and-fixes)

---

## 1. What Is This Project?

Standard LLMs tend to generate verbose answers even when a short answer would be correct.
This project creates a **Reinforcement Learning (RL) training environment** where:

- The model is given a question and a **token budget**.
- The model must **self-allocate** how many tokens it will use to answer (`<budget>N</budget>`).
- The model then provides a concise answer inside `<answer>...</answer>`.
- A **scorer** judges the answer's correctness AND how efficiently the model used its budget.
- A **reward signal** is returned to the RL trainer (e.g. TRL/GRPO in Colab).

Over many training steps, the model learns to be **both accurate AND concise**.

---

## 2. How the Environment Works (Big Picture)

```
┌────────────────────────────────────────────────────────────────┐
│                        Training Loop                           │
│                                                                │
│  1. env.reset()  →  picks a CURRICULUM-AWARE question         │
│                      returns: { prompt, episode_token_limit }  │
│                                                                │
│  2. Model generates response in the format:                    │
│       <budget>60</budget><answer>Paris.</answer>               │
│                                                                │
│  3. env.step(action)  →  anti-hacking checks                  │
│                           parses budget + answer               │
│                           runs 7-COMPONENT SCORER              │
│                           logs all reward components           │
│                           returns reward (float)               │
│                                                                │
│  4. RL trainer updates model weights based on reward           │
│                                                                │
│  Curriculum auto-advances when avg reward exceeds threshold    │
│  Repeat thousands of times → model improves                    │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Folder Structure

```
token-efficiency-env/
├── .gitignore                        ← excludes __pycache__ and .env
├── README.md                         ← this file
└── token_efficiency_env/             ← main project folder
    ├── env_server.py                 ← THE ENVIRONMENT (core logic)
    ├── env_client.py                 ← client to connect to HF Space
    ├── prompts.py                    ← bank of 24 questions
    ├── scorer.py                     ← reward calculation logic
    ├── openenv.yaml                  ← OpenEnv framework config
    ├── __init__.py                   ← package init (auto-generated)
    ├── client.py                     ← OpenEnv boilerplate (auto-generated)
    ├── models.py                     ← OpenEnv boilerplate (auto-generated)
    ├── pyproject.toml                ← Python package metadata
    └── server/                       ← FastAPI server (auto-generated)
        ├── app.py
        ├── Dockerfile
        └── requirements.txt
```

> **Note:** Files marked "auto-generated" were created by `openenv init`. You do not need to edit them.

---

## 4. File-by-File Explanation

---

### 4.1 `env_server.py`

**What it is:** The heart of the project. Defines the RL environment with **curriculum learning**, **anti-reward-hacking protections**, and **structured monitoring**.

**What it imports:**

| Import | Why |
|--------|-----|
| `re` | To parse `<budget>` and `<answer>` tags using regex |
| `sys`, `os` | To add OpenEnv's source path so Python can find it |
| `time` | For step timeout protection |
| `logging` | For structured per-component reward monitoring |
| `openenv.core.Environment` | Base class for all OpenEnv environments |
| `pydantic.BaseModel` | For defining typed action/observation schemas |
| `transformers.AutoTokenizer` | To count how many tokens the model's answer uses |
| `prompts.PROMPT_BANK` | The list of questions to sample from |
| `scorer.score_answer` | The 7-component reward function |
| `random` | For question selection within curriculum phases |

---

**Curriculum Learning (4 phases):**

The environment adapts difficulty based on a rolling average reward:

| Phase | Mix | Advance When |
|-------|-----|--------------|
| Phase 1 | 100% easy | avg reward > 0.4 |
| Phase 2 | 60% easy + 40% medium | avg reward > 0.5 |
| Phase 3 | 30% easy + 40% medium + 30% hard | avg reward > 0.6 |
| Phase 4 | 20% easy + 40% medium + 40% hard | Final phase |

---

**Anti-Reward-Hacking Protections:**

| Protection | What It Does |
|------------|-------------|
| Budget clamping | Forces budget to [1, 200] range |
| Empty answer detection | Penalty for blank/whitespace answers |
| Repetition guard | Catches "Paris Paris Paris" exploits (>60% same word) |
| Answer length sanity | Hard cap at 500 tokens |
| Step timeout | 30-second max per step |

---

**`reset()`** — Starts a new episode:
1. Checks if curriculum phase should advance (based on rolling avg reward)
2. Picks a question **weighted by current curriculum phase** (not random)
3. Logs episode info (phase, complexity, prompt)
4. Returns a `TokenEfficiencyObservation` with prompt and token limit

**`step(action)`** — Processes the model's response:
1. Parses `<budget>` and `<answer>` tags (format check → -1.0 penalty if missing)
2. **Anti-hacking checks**: budget clamping, empty answer, repetition guard, length sanity
3. Counts tokens using Qwen tokenizer
4. Calls 7-component `score_answer()` with complexity and keywords
5. Tracks reward in rolling window for curriculum advancement
6. **Logs all 7 reward components** for monitoring
7. Returns `observation`, `reward`, `done=True`, and detailed `info` dict

**`state()`** — Returns current task, episode count, phase, and average reward.

---

### 4.2 `env_client.py`

**What it is:** A lightweight client class so external code (or a Colab notebook) can talk to the deployed environment on Hugging Face Spaces via HTTP.

```python
from openenv.core import EnvClient

class TokenEfficiencyClient(EnvClient):
    base_url: str = "https://your-hf-space-url.hf.space"
```

**What to change:** Once you deploy to HF Spaces, replace `"https://your-hf-space-url.hf.space"` with your actual Space URL.

**How it works:** `EnvClient` (from OpenEnv) handles all the HTTP POST calls to the `/reset` and `/step` endpoints automatically. You just set the URL.

---

### 4.3 `prompts.py`

**What it is:** A static list of 24 questions at 3 difficulty levels. Used by `env_server.py` during `reset()`.

```python
PROMPT_BANK = [
    {"prompt": "What is 15% of 200?", "complexity": "easy"},
    ...
]
```

**Breakdown:**

| Complexity | Count | Expected answer length | Example |
|------------|-------|----------------------|---------|
| `easy`     | 8     | 1–5 words            | "What is the capital of France?" |
| `medium`   | 8     | 2–4 sentences        | "Explain what gravity is." |
| `hard`     | 8     | Detailed paragraphs  | "Explain how transformers work in ML." |

**Why complexity matters:**
The `scorer.py` uses prompt length as a proxy for complexity to judge whether the model's self-allocated budget was reasonable. Harder questions deserve a higher budget.

---

### 4.4 `scorer.py`

**What it is:** The 7-component reward function. Called by `env_server.step()` after every model response.

```python
def score_answer(prompt, response, allocated_budget, tokens_used,
                 complexity="medium", expected_keywords=None) -> dict
```

**Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | str | The original question |
| `response` | str | The model's answer (extracted from `<answer>` tag) |
| `allocated_budget` | int | The budget the model chose (from `<budget>` tag) |
| `tokens_used` | int | Actual tokens in the answer (counted by tokenizer) |
| `complexity` | str | Question complexity level ("easy", "medium", "hard") |
| `expected_keywords` | list | Keyword stems to verify in the answer |

**Output:** A dict with `"reward"` (float) and `"details"` (per-component scores).

The reward is built from **7 independent components**:

---

**Component 1 — Correctness (40% weight)**

Uses Claude Haiku API as primary LLM judge. Clamped to [0.0, 1.0].
If the API call fails, `correctness = 0.0`.

---

**Component 2 — Efficiency (20% weight)**

Rewards using fewer tokens than the self-allocated budget.
Gradient penalty for exceeding budget (proportional to overshoot).

---

**Component 3 — Budget Reasonableness (10% weight)**

Now uses the actual `complexity` label from PROMPT_BANK (not word count proxy):
- Easy → ideal budget ≈ 30 tokens (15% of 200)
- Medium → ideal budget ≈ 90 tokens (45% of 200)
- Hard → ideal budget ≈ 160 tokens (80% of 200)

---

**Component 4 — Redundancy Penalty (10% weight)** *(NEW)*

Measures unique information density. Uses unique word ratio + bigram analysis.
Penalises "Paris. Paris is the capital. The capital is Paris." style answers.

---

**Component 5 — Self-Assessment Accuracy (10% weight)** *(NEW)*

How close was the model's `<budget>` to actual `tokens_used`?
Trains the model to predict its own output length accurately.

---

**Component 6 — Keyword Verification (5% weight)** *(NEW)*

Secondary correctness check that does NOT rely on LLM judge.
Uses `expected_keywords` from PROMPT_BANK to verify answer content.

---

**Component 7 — Format Quality (5% weight)** *(NEW)*

Rewards clean, well-structured responses. Penalises excessive whitespace,
too-short answers for hard questions, too-long answers for easy questions,
and answers that just restate the question.

---

**Final reward formula:**

```
reward = 0.40 × correctness + 0.20 × efficiency + 0.10 × budget_reasonableness
       + 0.10 × redundancy  + 0.10 × self_assessment
       + 0.05 × keyword_verification + 0.05 × format_quality
```

Returned as a dict: `{"reward": float, "details": {component_scores}}`.

---

### 4.5 `openenv.yaml`

**What it is:** The configuration file that tells the OpenEnv framework what this environment is and where to find its classes.

```yaml
name: token-efficiency-env
version: 0.1.0
description: >
  An RL environment that trains LLMs to answer correctly
  using fewer tokens by dynamically allocating a token budget
  based on task complexity.
entry_point: env_server:TokenEfficiencyEnv
action_schema: env_server:TokenEfficiencyAction
observation_schema: env_server:TokenEfficiencyObservation
```

| Field | Meaning |
|-------|---------|
| `name` | Display name of the environment |
| `version` | Semantic version |
| `entry_point` | `module:ClassName` — where to find the Environment class |
| `action_schema` | `module:ClassName` — Pydantic model for actions |
| `observation_schema` | `module:ClassName` — Pydantic model for observations |

The OpenEnv server reads this file to auto-wire the FastAPI endpoints (`/reset`, `/step`, `/state`).

---

## 5. The Output Format Contract

**This is the most important agreement between all 3 people.**

Every time the model responds, it MUST use exactly this format — no exceptions:

```
<budget>60</budget><answer>Gravity is the force of attraction between masses.</answer>
```

- `<budget>N</budget>` — N is an integer (1–200), chosen by the model itself
- `<answer>text</answer>` — the model's actual answer, can span multiple lines

**If the model does not follow this format:**
```python
return {
    "observation": {"error": "bad format"},
    "reward": -1.0,    # hard penalty
    "done": True,
    "info": {"error": "model did not follow format"}
}
```

The model gets penalized with `-1.0` reward for every badly formatted response.
This strongly incentivizes the model to always follow the format during training.

---

## 6. The Scoring Formula

Quick reference card:

```
┌──────────────────────────────────────────────────────────────┐
│  reward = 0.40 × correctness                                │
│         + 0.20 × efficiency                                 │
│         + 0.10 × budget_reasonableness                      │
│         + 0.10 × redundancy_penalty                         │
│         + 0.10 × self_assessment_accuracy                   │
│         + 0.05 × keyword_verification                       │
│         + 0.05 × format_quality                             │
│                                                              │
│  7 independent signals — much harder to game than 3          │
└──────────────────────────────────────────────────────────────┘
```

**Worked example:**

Model answers: `<budget>20</budget><answer>Paris.</answer>`
for the question: "What is the capital of France?" (easy, keywords: ["paris"])

- `correctness` = 1.0 (Claude confirms correct)
- `efficiency` = 1.0 − (2/20) × 0.3 = **0.97** (used 2 of 20 tokens)
- `budget_reasonableness` = 1 − |0.10 − 0.15| = **0.95** (easy ideal = 15%)
- `redundancy` = **1.0** (no repetition in single word)
- `self_assessment` = 1 − |20−2|/20 = **0.10** (budget was too generous)
- `keyword_verification` = **1.0** ("paris" found in answer)
- `format_quality` = **1.0** (clean, appropriate length for easy)
- **Final reward** = 0.40×1.0 + 0.20×0.97 + 0.10×0.95 + 0.10×1.0 + 0.10×0.10 + 0.05×1.0 + 0.05×1.0 = **0.809**

---

## 7. Setup — Step by Step

### Prerequisites

- Python 3.10+
- Git
- A GitHub account with access to `github.com/SaishAmbar/token-efficiency-env`
- An Anthropic API key (for scoring during training)

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/SaishAmbar/token-efficiency-env.git
cd token-efficiency-env
```

---

### Step 2 — Clone OpenEnv (required dependency)

```bash
cd ..
git clone https://github.com/meta-pytorch/OpenEnv.git
cd token-efficiency-env
```

After this, your folder structure should look like:

```
your-folder/
├── token-efficiency-env/    ← your project
└── OpenEnv/                 ← the framework
```

The `sys.path.insert(0, '../OpenEnv/src')` in `env_server.py` and `env_client.py`
depends on this exact layout.

---

### Step 3 — Install dependencies

```bash
pip install openenv-core trl transformers anthropic pydantic
```

Verify everything is installed:

```bash
python -c "import openenv; import transformers; import trl; print('all good')"
```

Expected output: `all good`

---

### Step 4 — Set your Anthropic API key

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

**Mac/Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

Without this key, the correctness score will default to `0.0` (the `except` block in `scorer.py`).

---

### Step 5 — Navigate to the project folder

```bash
cd token-efficiency-env/token_efficiency_env
```

---

### Step 6 — Test the environment manually

```python
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../OpenEnv/src')

from env_server import TokenEfficiencyEnv, TokenEfficiencyAction

env = TokenEfficiencyEnv()

# Start a new episode
obs = env.reset()
print("Prompt:", obs.prompt)
print("Token limit:", obs.episode_token_limit)

# Simulate a model response
result = env.step(TokenEfficiencyAction(
    raw_response="<budget>20</budget><answer>Paris.</answer>"
))

print("Reward:", result["reward"])
print("Tokens used:", result["info"]["tokens_used"])
print("Budget allocated:", result["info"]["allocated_budget"])
```

---

### Step 7 — Push changes to GitHub

After editing any file:

```bash
git add .
git commit -m "describe what you changed"
git push origin main
```

If push is rejected (remote has newer commits):
```bash
git pull origin main --allow-unrelated-histories
git push origin main
```

---

## 8. Testing the Environment Locally

Run this quick test directly from the terminal:

```bash
cd token_efficiency_env

python -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../OpenEnv/src')
from env_server import TokenEfficiencyEnv, TokenEfficiencyAction

env = TokenEfficiencyEnv()
obs = env.reset()
print('Prompt:', obs.prompt)

result = env.step(TokenEfficiencyAction(raw_response='<budget>20</budget><answer>Paris.</answer>'))
print('Reward:', result['reward'])
print('Info:', result['info'])
"
```

**Expected output (example):**
```
Prompt: What is the capital of France?
Reward: 0.931
Info: {'tokens_used': 2, 'allocated_budget': 20, 'correctness': 0.931}
```

**Test bad format (should return -1.0):**
```python
result = env.step(TokenEfficiencyAction(raw_response="This is just a plain response"))
print(result["reward"])  # → -1.0
```

---

## 9. Google Colab (P3) Setup

Open [colab.research.google.com](https://colab.research.google.com) → New Notebook → Runtime → Change runtime type → **T4 GPU** → Save.

Paste the following into separate cells in order:

---

**Cell 1 — Install packages:**
```python
!pip install openenv-core trl unsloth transformers anthropic
!git clone https://github.com/meta-pytorch/OpenEnv.git
!git clone https://github.com/SaishAmbar/token-efficiency-env.git
```

---

**Cell 2 — Load the model:**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=512,
    load_in_4bit=True
)
```

---

**Cell 3 — Set Anthropic API key:**
```python
import os
os.environ["ANTHROPIC_API_KEY"] = "paste-your-key-here"
```

---

**Cell 4 — Import the environment:**
```python
import sys
sys.path.insert(0, "/content/OpenEnv/src")
sys.path.insert(0, "/content/token-efficiency-env/token_efficiency_env")

from env_server import TokenEfficiencyEnv, TokenEfficiencyAction
env = TokenEfficiencyEnv()
```

---

**Cell 5 — Test before training (run this first):**
```python
obs = env.reset()
print("Prompt:", obs.prompt)

result = env.step(TokenEfficiencyAction(
    raw_response="<budget>20</budget><answer>Paris.</answer>"
))
print("Reward:", result["reward"])
print("Info:", result["info"])
```

> ✅ If a reward number prints → environment is working. Proceed to training.
> ❌ If an error appears → check that all cells above ran successfully first.

---

## 10. Environment Variables

| Variable | Required? | Description |
|----------|-----------|-------------|
| `ANTHROPIC_API_KEY` | Yes (for scoring) | Your Anthropic API key. Without it, correctness defaults to 0. |

**Where to get an Anthropic API key:**
Go to [console.anthropic.com](https://console.anthropic.com) → API Keys → Create Key.

---

## 11. Common Errors and Fixes

---

**Error: `No module named 'openenv'`**

Cause: The `sys.path.insert` path to `OpenEnv/src` is wrong.
Fix: Make sure `OpenEnv/` sits one folder above `token-efficiency-env/`. The relative path `../OpenEnv/src` must point to the right place.

```
your-folder/
├── token-efficiency-env/   ← you are here
└── OpenEnv/                ← must be here
```

---

**Error: `No module named 'anthropic'`**

Fix:
```bash
pip install anthropic
```

---

**Error: `model did not follow format` / reward = -1.0**

Cause: The model's response did not contain `<budget>N</budget>` and `<answer>text</answer>`.
Fix: Make sure your prompt to the model includes the format instruction. For Colab training (P3), add this system prompt:

```
Always respond in exactly this format:
<budget>N</budget><answer>your answer here</answer>
where N is the number of tokens you plan to use.
```

---

**Error: `refusing to merge unrelated histories` (git pull)**

Fix:
```bash
git pull origin main --allow-unrelated-histories
```

---

**Error: `The token '&&' is not a valid statement separator` (Windows PowerShell)**

Cause: PowerShell doesn't support `&&` like bash does.
Fix: Run commands separately with `;` instead:
```powershell
git add .; git commit -m "your message"
```

---

**Warning: `huggingface_hub cache-system uses symlinks`**

This is just a warning on Windows, not an error. The tokenizer still works.
To suppress it permanently:
```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

---

## Quick Reference — Who Does What

| Person | Files to edit | Where |
|--------|--------------|-------|
| P1 | `env_server.py`, `env_client.py` | Cursor on laptop |
| P2 | `prompts.py`, `scorer.py` | Cursor on same laptop |
| P3 | Training cells | Google Colab in browser |

**Git workflow:**
1. P1 pushes → P2 runs `git pull` → P2 edits → P2 pushes → P3 runs `git clone` in Colab

---

*Last updated: April 2026*
