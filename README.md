# 🎯 TokenEfficiencyEnv

> An OpenEnv RL environment that trains LLMs to answer correctly using **fewer tokens**, by dynamically allocating a token budget based on task complexity.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License: BSD](https://img.shields.io/badge/License-BSD-yellow)](LICENSE)

---

## 🧠 What Does This Do?

Standard LLMs generate verbose answers even when a short answer would suffice. This project creates an **RL training environment** where the model learns to:

1. **Self-allocate** a token budget (`<budget>N</budget>`)
2. **Answer concisely** within that budget (`<answer>text</answer>`)
3. **Balance accuracy vs efficiency** — scored on 7 independent dimensions

Over thousands of training steps, the model becomes both **accurate AND concise**.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Training Loop                           │
│                                                                │
│  1. env.reset()  →  picks a question (curriculum-aware)       │
│                      returns: { prompt, episode_token_limit }  │
│                                                                │
│  2. Model generates response:                                  │
│       <budget>60</budget><answer>Paris.</answer>               │
│                                                                │
│  3. env.step(action)  →  parses budget + answer               │
│                           runs 7-component scorer              │
│                           applies anti-hacking checks          │
│                           returns reward (float)               │
│                                                                │
│  4. RL trainer (TRL/GRPO) updates model weights               │
│                                                                │
│  Repeat → model improves at being concise AND correct          │
└────────────────────────────────────────────────────────────────┘
```

---

## 📊 7-Component Scoring Formula

| # | Component | Weight | What It Measures |
|---|-----------|--------|-----------------|
| 1 | **Correctness** | 40% | LLM judge (Claude Haiku) rates answer 0.0–1.0 |
| 2 | **Efficiency** | 20% | Tokens used vs self-allocated budget |
| 3 | **Budget Reasonableness** | 10% | Budget matches question complexity |
| 4 | **Redundancy Penalty** | 10% | Unique word ratio — penalises repetition |
| 5 | **Self-Assessment** | 10% | How accurately budget predicts actual usage |
| 6 | **Keyword Verification** | 5% | Pattern match for known answer fragments |
| 7 | **Format Quality** | 5% | Clean, well-structured response |

```
reward = 0.40×correctness + 0.20×efficiency + 0.10×budget_reasonableness
       + 0.10×redundancy + 0.10×self_assessment
       + 0.05×keyword_verification + 0.05×format_quality
```

---

## 📈 Curriculum Learning

The environment **adapts difficulty** based on model performance:

| Phase | Questions | Advance When |
|-------|-----------|-------------|
| Phase 1 | 100% Easy | avg reward > 0.4 |
| Phase 2 | 60% Easy + 40% Medium | avg reward > 0.5 |
| Phase 3 | 30% Easy + 40% Medium + 30% Hard | avg reward > 0.6 |
| Phase 4 | 20% Easy + 40% Medium + 40% Hard | Final phase |

---

## 🛡️ Anti-Reward-Hacking Protections

- **Budget clamping**: Forces budget to [1, 200] range
- **Empty answer detection**: Penalty for blank responses
- **Repetition guard**: Catches "Paris Paris Paris Paris" style exploits
- **Answer length sanity**: Hard cap at 500 tokens
- **Step timeout**: 30-second max per step
- **7 independent reward functions**: Much harder to game than a single signal

---

## 📁 Project Structure

```
token-efficiency-env/
├── README.md                         ← this file
└── token_efficiency_env/             ← main project folder
    ├── env_server.py                 ← THE ENVIRONMENT (core logic + curriculum)
    ├── env_client.py                 ← client to connect to HF Space
    ├── prompts.py                    ← 24 questions with keywords
    ├── scorer.py                     ← 7-component reward function
    ├── openenv.yaml                  ← OpenEnv framework config
    ├── __init__.py                   ← package init
    ├── client.py                     ← OpenEnv client boilerplate
    ├── models.py                     ← OpenEnv models boilerplate
    ├── pyproject.toml                ← Python package metadata
    └── server/                       ← FastAPI server
        ├── app.py
        ├── Dockerfile
        └── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/SaishAmbar/token-efficiency-env.git
cd token-efficiency-env
```

### 2. Clone OpenEnv (dependency)
```bash
cd ..
git clone https://github.com/meta-pytorch/OpenEnv.git
cd token-efficiency-env
```

### 3. Install dependencies
```bash
pip install openenv-core trl transformers anthropic pydantic
```

### 4. Set API key
```powershell
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```
```bash
# Mac/Linux
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

### 5. Test locally
```python
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../OpenEnv/src')

from env_server import TokenEfficiencyEnv, TokenEfficiencyAction

env = TokenEfficiencyEnv()
obs = env.reset()
print("Prompt:", obs.prompt)

result = env.step(TokenEfficiencyAction(
    raw_response="<budget>20</budget><answer>Paris.</answer>"
))
print("Reward:", result["reward"])
print("Components:", result["info"]["reward_components"])
```

---

## 👥 Team Structure

| Person | Responsibility | Files |
|--------|---------------|-------|
| P1 | Environment | `env_server.py`, `env_client.py` |
| P2 | Rewards & Prompts | `scorer.py`, `prompts.py` |
| P3 | Training (Colab) | TRL + Unsloth + GRPO |

---

## 🛠️ Tech Stack

- **OpenEnv** — environment interface
- **TRL** — RL training (GRPO)
- **Unsloth** — efficient training & inference
- **Qwen2.5-3B-Instruct** — base model
- **Claude Haiku** — LLM judge for correctness

---

*Last updated: April 2026*
