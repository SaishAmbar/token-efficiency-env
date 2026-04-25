"""
TokenEfficiencyEnv — Interactive Web Dashboard (Backend)

A pretty browser UI that exercises the *real* TokenEfficiencyEnvironment
in-process, not via HTTP. There is no longer a duplicate "local scorer" —
every reward in this dashboard is computed by the same code path the
GRPO trainer uses, including the LLM judge, anti-hacking cliffs, and
curriculum sampling.

Two tabs:
  - **Interactive Mode** — pick a question, type your own response, see
    the 6-component reward breakdown. Uses the configured judge
    (HuggingFace by default; falls back to keyword if HF_TOKEN missing).
  - **Training Simulation** — batch-runs N episodes through a fake
    "trainee" whose skill ramps from 0 to 1, scored by the real env.
    Forces the keyword judge so it stays fast and offline regardless of
    how you started the server.

Run:
    # default port 8001 (so it doesn't clash with the OpenEnv server on 8000)
    python web_dashboard.py
    # or:  uvicorn web_dashboard:app --host 0.0.0.0 --port 8001

This file deliberately runs on port 8001 by default. The OpenEnv server
(in token_efficiency_env/server/app.py) lives on 8000 — they're meant to
coexist while you're poking at the env.
"""
from __future__ import annotations

import contextlib
import os
import random
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Make the `token_efficiency_env` package importable when running this file
# directly from the repo root (no `pip install -e` required for local play).
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from token_efficiency_env import judge as judge_mod  # noqa: E402
from token_efficiency_env.models import TokenEfficiencyAction  # noqa: E402
from token_efficiency_env.prompts import PROMPT_BANK  # noqa: E402
from token_efficiency_env.server.token_efficiency_env_environment import (  # noqa: E402
    TokenEfficiencyEnvironment,
)

app = FastAPI(title="TokenEfficiencyEnv Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Singleton env for the Interactive tab ───────────────────────────────
# One process-level instance so the Episode counter, recent_rewards deque,
# and curriculum phase persist across user clicks. The env lazily loads the
# Qwen tokenizer on the first step() call (one-time ~2s download).
_interactive_env: Optional[TokenEfficiencyEnvironment] = None


def _get_interactive_env() -> TokenEfficiencyEnvironment:
    global _interactive_env
    if _interactive_env is None:
        _interactive_env = TokenEfficiencyEnvironment()
    return _interactive_env


# ─── Force keyword judge for the Simulation tab ──────────────────────────
@contextlib.contextmanager
def _force_keyword_judge():
    """Temporarily swap the process-level judge singleton for KeywordJudge.

    The simulation tab fires 50-200 step() calls in a tight loop — at ~1-2s
    per HF Inference call that's an unacceptable wait, so we force the
    offline keyword scorer for the simulation only. The interactive tab is
    unaffected.
    """
    saved = judge_mod._JUDGE
    judge_mod._JUDGE = judge_mod.KeywordJudge()
    try:
        yield
    finally:
        judge_mod._JUDGE = saved


# ─── Fake trainee for the Simulation tab ─────────────────────────────────
# These are hand-written "good answers" used to fabricate plausible LLM
# output at varying skill levels. They are NOT used by the env — only by
# the simulator, to make the training-curve animation look like real RL.
GOOD_ANSWERS: dict[str, tuple[str, str]] = {
    "What is 15% of 200? Reply with one number.": ("8", "30"),
    "What is the capital of France? One word.": ("3", "Paris"),
    "What does CPU stand for?": ("8", "Central Processing Unit"),
    "What is 2 to the power of 8?": ("4", "256"),
    "What color do you get mixing red and blue?": ("4", "Purple"),
    "How many sides does a hexagon have?": ("4", "Six"),
    "What is the boiling point of water in Celsius?": ("8", "100 degrees Celsius"),
    "Who wrote Romeo and Juliet?": ("6", "William Shakespeare"),
    "Explain what gravity is.": (
        "40",
        "Gravity is a fundamental force of attraction between objects with mass. "
        "Greater mass means stronger gravitational pull.",
    ),
    "What is the difference between RAM and ROM?": (
        "60",
        "RAM is volatile memory for temporary data that is lost when powered off. "
        "ROM is non-volatile read-only memory that retains data permanently.",
    ),
    "How does a vaccine work?": (
        "60",
        "Vaccines introduce weakened or inactive pathogens to stimulate the immune "
        "system to produce antibodies, creating immunity without causing the disease.",
    ),
    "What causes seasons on Earth?": (
        "60",
        "Seasons are caused by Earth's 23.5-degree axial tilt. As Earth orbits the sun, "
        "different hemispheres receive varying amounts of direct sunlight.",
    ),
    "Explain what inflation means.": (
        "40",
        "Inflation is the sustained increase in general price levels over time, "
        "reducing the purchasing power of money.",
    ),
    "What is the difference between speed and velocity?": (
        "50",
        "Speed is a scalar measuring how fast something moves. Velocity is a vector "
        "that includes both speed and direction.",
    ),
    "How does the internet work in simple terms?": (
        "60",
        "The internet is a global network of computers that exchange data through "
        "servers using standardized protocols, routing information in packets.",
    ),
    "What is photosynthesis?": (
        "50",
        "Photosynthesis is the process by which plants convert light energy, water, "
        "and carbon dioxide into glucose and oxygen for energy.",
    ),
    "Explain how transformers work in machine learning.": (
        "120",
        "Transformers use self-attention mechanisms to process all tokens in parallel. "
        "Each layer computes attention scores between token pairs, allowing the model "
        "to capture long-range dependencies. The architecture uses multi-head attention, "
        "layer normalization, and feed-forward networks.",
    ),
    "What are the causes and effects of climate change?": (
        "120",
        "Climate change is primarily caused by greenhouse gas emissions, especially "
        "carbon dioxide from burning fossil fuels. Effects include rising global "
        "temperatures, melting ice caps, sea level rise, and extreme weather events.",
    ),
    "Explain the difference between supervised and unsupervised learning.": (
        "120",
        "Supervised learning trains on labeled data to predict outputs, used for "
        "classification and regression. Unsupervised learning finds patterns in unlabeled "
        "data through clustering and dimensionality reduction without predefined targets.",
    ),
    "How does the immune system fight a virus?": (
        "120",
        "When a pathogen enters the body, innate immune cells respond first. Then "
        "adaptive immunity activates: T-cells destroy infected cells while B-cells "
        "produce antibodies that neutralize the pathogen and provide future immunity.",
    ),
    "Explain what quantum entanglement is.": (
        "120",
        "Quantum entanglement is a phenomenon where two particles become correlated so "
        "that measuring the state of one instantly determines the state of the other, "
        "regardless of distance. This non-local connection challenges classical physics.",
    ),
    "What is the significance of the Turing Test?": (
        "120",
        "The Turing Test, proposed by Alan Turing, evaluates machine intelligence by "
        "testing if a human judge can distinguish between a machine and human in "
        "conversation. It remains a foundational benchmark for artificial intelligence.",
    ),
    "How do neural networks learn from data?": (
        "120",
        "Neural networks learn by adjusting connection weights through backpropagation. "
        "During training, the network makes predictions, calculates error via a loss "
        "function, then computes gradients to update weights, gradually minimizing error.",
    ),
    "Explain the theory of relativity in simple terms.": (
        "120",
        "Einstein's theory has two parts. Special relativity says space and time are "
        "linked and nothing travels faster than light. General relativity explains "
        "gravity as the curvature of spacetime caused by mass and energy.",
    ),
}


def _simulated_response(prompt: str, skill: float) -> str:
    """Fabricate an LLM response of varying quality based on `skill` in [0,1].

    skill < 0.15 → no XML tags at all       (bad_format cliff)
    skill < 0.30 → correct but verbose, huge budget
    skill < 0.50 → correct, mediocre budget calibration
    skill < 0.70 → correct, decent budget calibration
    skill >= 0.70 → correct, well-calibrated
    """
    good = GOOD_ANSWERS.get(prompt)
    if not good:
        return "<budget>50</budget><answer>I'm not sure about this question.</answer>"
    good_budget, good_answer = good
    if skill < 0.15:
        return "The answer to this question is something I need to think about carefully."
    if skill < 0.30:
        return (
            f"<budget>190</budget><answer>Well, that is a very interesting question. "
            f"Let me think about this carefully. {good_answer}. I hope that helps you "
            f"understand the topic better and provides the information you wanted.</answer>"
        )
    if skill < 0.50:
        bad_budget = str(int(int(good_budget) * random.uniform(1.5, 3.0)))
        return f"<budget>{bad_budget}</budget><answer>{good_answer}. This is an important topic.</answer>"
    if skill < 0.70:
        ok_budget = str(int(int(good_budget) * random.uniform(1.0, 1.5)))
        return f"<budget>{ok_budget}</budget><answer>{good_answer}.</answer>"
    return f"<budget>{good_budget}</budget><answer>{good_answer}.</answer>"


# ─── Request models ──────────────────────────────────────────────────────
class StepRequest(BaseModel):
    raw_response: str


class SimulateRequest(BaseModel):
    num_episodes: int = 60


# ─── Helpers ─────────────────────────────────────────────────────────────
def _step_to_dict(env: TokenEfficiencyEnvironment, obs) -> dict:
    """Map a TokenEfficiencyObservation to the JSON shape the frontend expects.

    Frontend keys vs env field names (kept stable for back-compat):
        budget       <- allocated_budget
        avg_reward   <- avg_reward_50
        details      <- reward_components
    """
    avg = sum(env.recent_rewards) / len(env.recent_rewards) if env.recent_rewards else 0.0
    return {
        "reward": float(obs.reward) if obs.reward is not None else 0.0,
        "details": dict(obs.reward_components),
        "tokens_used": obs.tokens_used,
        "budget": obs.allocated_budget,
        "answer": obs.answer,
        "avg_reward": round(avg, 4),
        "episode": obs.episode,
        "phase": obs.phase,
        "complexity": obs.complexity,
        "error": obs.error or None,
    }


def _reset_to_dict(env: TokenEfficiencyEnvironment, obs) -> dict:
    task = env.current_task or {}
    return {
        "episode": obs.episode,
        "phase": obs.phase,
        "complexity": task.get("complexity", "easy"),
        "prompt": obs.prompt,
        "keywords": task.get("expected_keywords", []),
        "token_limit": obs.episode_token_limit,
    }


# ─── API endpoints ───────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/questions")
async def get_questions():
    return [
        {
            "prompt": p["prompt"],
            "complexity": p["complexity"],
            "keywords": p.get("expected_keywords", []),
        }
        for p in PROMPT_BANK
    ]


@app.post("/api/reset")
async def api_reset(question_index: Optional[int] = None):
    env = _get_interactive_env()
    if question_index is not None and 0 <= question_index < len(PROMPT_BANK):
        # User picked a specific question — temporarily override the
        # curriculum sampler so reset() picks our question while still doing
        # all its normal phase/episode bookkeeping.
        chosen = PROMPT_BANK[question_index]
        original = env._select_question
        env._select_question = lambda: chosen  # type: ignore[method-assign]
        try:
            obs = env.reset()
        finally:
            env._select_question = original  # type: ignore[method-assign]
    else:
        obs = env.reset()
    return _reset_to_dict(env, obs)


@app.post("/api/step")
async def api_step(req: StepRequest):
    env = _get_interactive_env()
    if env.current_task is None:
        return JSONResponse(
            {"error": "Call /api/reset first", "reward": 0.0, "details": {}},
            status_code=400,
        )
    obs = env.step(TokenEfficiencyAction(raw_response=req.raw_response))
    return _step_to_dict(env, obs)


@app.post("/api/simulate")
async def api_simulate(req: SimulateRequest):
    """Run N episodes with a fake trainee whose skill ramps 0 → 1.

    Forces the keyword judge so it stays fast even when the dashboard was
    started with HF credentials.
    """
    n = max(1, min(int(req.num_episodes), 500))
    results: list[dict] = []
    with _force_keyword_judge():
        sim_env = TokenEfficiencyEnvironment()
        for i in range(n):
            skill = min(1.0, (i / n) * 1.1 + random.uniform(-0.05, 0.05))
            reset_obs = sim_env.reset()
            prompt = reset_obs.prompt
            raw = _simulated_response(prompt, skill)
            obs = sim_env.step(TokenEfficiencyAction(raw_response=raw))
            avg = (
                sum(sim_env.recent_rewards) / len(sim_env.recent_rewards)
                if sim_env.recent_rewards else 0.0
            )
            results.append({
                "episode": obs.episode,
                "reward": float(obs.reward) if obs.reward is not None else 0.0,
                "avg_reward": round(avg, 4),
                "phase": obs.phase,
                "complexity": sim_env.current_task.get("complexity", "easy"),
                "prompt": prompt[:60],
                "answer": obs.answer[:80],
                "budget": obs.allocated_budget,
                "tokens_used": obs.tokens_used,
                "skill": round(skill, 3),
                "details": dict(obs.reward_components),
                "error": obs.error or None,
            })
    return results


@app.get("/api/state")
async def api_state():
    env = _get_interactive_env()
    avg = (
        sum(env.recent_rewards) / len(env.recent_rewards)
        if env.recent_rewards else 0.0
    )
    return {
        "episode": env.episode_count,
        "phase": env._phase_name(),
        "phase_index": env.current_phase,
        "avg_reward": round(avg, 4),
        "total_rewards": len(env.recent_rewards),
        "judge": type(judge_mod.get_judge()).__name__,
    }


def main():  # pragma: no cover — convenience CLI entry point
    import uvicorn
    port = int(os.environ.get("DASHBOARD_PORT", "8001"))
    print(f"\n  Dashboard at http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
