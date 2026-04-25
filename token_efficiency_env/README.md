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
│  1. env.reset()  →  picks a random question from PROMPT_BANK  │
│                      returns: { prompt, episode_token_limit }  │
│                                                                │
│  2. Model generates response in the format:                    │
│       <budget>60</budget><answer>Paris.</answer>               │
│                                                                │
│  3. env.step(action)  →  parses budget + answer               │
│                           counts tokens used                   │
│                           calls score_answer()                 │
│                           returns reward (float 0 to 1)        │
│                                                                │
│  4. RL trainer updates model weights based on reward           │
│                                                                │
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

**What it is:** The heart of the project. Defines the RL environment.

**What it imports:**

| Import | Why |
|--------|-----|
| `re` | To parse `<budget>` and `<answer>` tags using regex |
| `sys`, `os` | To add OpenEnv's source path so Python can find it |
| `openenv.core.Environment` | Base class for all OpenEnv environments |
| `pydantic.BaseModel` | For defining typed action/observation schemas |
| `transformers.AutoTokenizer` | To count how many tokens the model's answer uses |
| `prompts.PROMPT_BANK` | The list of questions to sample from |
| `scorer.score_answer` | The function that returns the reward |
| `random` | To randomly pick a question each episode |

**The Tokenizer:**

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
```

This loads the Qwen tokenizer at startup. It downloads ~few MB the first time, then caches.
It is used to count exact token usage in the model's answer.

---

**`TokenEfficiencyAction` — What the model sends:**

```python
class TokenEfficiencyAction(BaseModel):
    raw_response: str
```

The entire model output (as a raw string) is wrapped in this object before being passed to `step()`.
Example value of `raw_response`:
```
<budget>60</budget><answer>Gravity is the force that attracts objects toward each other.</answer>
```

---

**`TokenEfficiencyObservation` — What the environment sends back:**

```python
class TokenEfficiencyObservation(BaseModel):
    prompt: str
    episode_token_limit: int = 200
```

After `reset()` is called, the model receives:
- `prompt` — the question it needs to answer
- `episode_token_limit` — the absolute maximum tokens allowed (always 200)

The model must choose its OWN budget (≤ 200) via the `<budget>` tag.

---

**`TokenEfficiencyEnv` — The environment class:**

```python
class TokenEfficiencyEnv(Environment):
    def __init__(self): ...
    def reset(self): ...
    def step(self, action): ...
    def state(self): ...
```

**`__init__`** — Sets up two instance variables:
- `self.current_task` — stores the current question dict (initially `None`)
- `self.episode_token_limit` — hardcoded to `200`

**`reset()`** — Starts a new episode:
1. Picks a random item from `PROMPT_BANK` (e.g. `{"prompt": "What is gravity?", "complexity": "medium"}`)
2. Stores it in `self.current_task`
3. Returns a `TokenEfficiencyObservation` with the prompt and token limit

**`step(action)`** — Processes the model's response:
1. Reads `action.raw_response` (the full model output string)
2. Uses `re.search` to extract the number inside `<budget>N</budget>`
3. Uses `re.search` to extract the text inside `<answer>text</answer>`
4. If either tag is missing → returns `reward = -1.0` (bad format penalty)
5. Counts tokens in the answer using the Qwen tokenizer
6. Calls `score_answer()` with prompt, answer, budget, and tokens_used
7. Returns a dict with `observation`, `reward`, `done=True`, and `info`

**`state()`** — Returns the current task (used for debugging/logging).

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

**What it is:** The reward function. Called by `env_server.step()` after every model response.

```python
def score_answer(prompt, response, allocated_budget, tokens_used) -> float
```

**Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | str | The original question |
| `response` | str | The model's answer (extracted from `<answer>` tag) |
| `allocated_budget` | int | The budget the model chose (from `<budget>` tag) |
| `tokens_used` | int | Actual tokens in the answer (counted by tokenizer) |

**Output:** A single float reward between 0.0 and 1.0 (approximately).

The reward is built from **3 components**:

---

**Component 1 — Correctness (50% weight)**

Uses the Anthropic Claude API (`claude-haiku-4-5`) as an LLM judge:

```python
judge = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=10,
    messages=[{"role": "user", "content": f"Question: {prompt}\nAnswer: {response}\nRate 0.0 to 1.0..."}]
)
correctness = float(judge.content[0].text.strip())
```

The judge returns a number like `0.8`. This is clamped to `[0.0, 1.0]`.
If the API call fails (e.g. no key), `correctness = 0.0`.

---

**Component 2 — Efficiency (30% weight)**

Rewards using fewer tokens than the self-allocated budget:

```python
if tokens_used <= allocated_budget:
    efficiency = 1.0 - (tokens_used / allocated_budget) * 0.3
else:
    efficiency = -0.5   # penalty for going over own budget
```

- If you used 20 out of 60 tokens → efficiency ≈ 0.90 (good)
- If you used 60 out of 60 tokens → efficiency = 0.70
- If you used 80 but allocated only 60 → efficiency = −0.5 (penalty)

---

**Component 3 — Budget Reasonableness (20% weight)**

Checks whether the model allocated an appropriate budget for the complexity:

```python
prompt_length = len(prompt.split())          # word count as complexity proxy
expected_ratio = min(prompt_length / 15, 1.0)
allocated_ratio = min(allocated_budget / 200, 1.0)
budget_reasonableness = 1.0 - abs(allocated_ratio - expected_ratio)
```

- A short easy question (4 words) → expected_ratio ≈ 0.27 → budget ≈ 54 tokens is ideal
- A long hard question (10 words) → expected_ratio ≈ 0.67 → budget ≈ 134 tokens is ideal
- Allocating 200 tokens for "What is 2^8?" would score poorly here

---

**Final reward formula:**

```
reward = 0.5 × correctness + 0.3 × efficiency + 0.2 × budget_reasonableness
```

Rounded to 4 decimal places and returned as a float.

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
┌─────────────────────────────────────────────────────┐
│  reward = 0.5 × correctness                         │
│         + 0.3 × efficiency                          │
│         + 0.2 × budget_reasonableness               │
│                                                     │
│  correctness        = Claude API score (0.0–1.0)    │
│  efficiency         = how well budget was used      │
│  budget_reasonableness = appropriate budget for Q   │
└─────────────────────────────────────────────────────┘
```

**Worked example:**

Model answers: `<budget>20</budget><answer>Paris.</answer>`
for the question: "What is the capital of France?"

- `correctness` = 1.0 (Claude confirms "Paris" is correct)
- `tokens_used` = 2, `allocated_budget` = 20
- `efficiency` = 1.0 − (2/20) × 0.3 = **0.97**
- `prompt_length` = 6 words → `expected_ratio` = 6/15 = 0.40 → ideal budget ≈ 80
- `allocated_ratio` = 20/200 = 0.10
- `budget_reasonableness` = 1 − |0.10 − 0.40| = **0.70**
- **Final reward** = 0.5×1.0 + 0.3×0.97 + 0.2×0.70 = **0.931**

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
