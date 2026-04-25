"""
TokenEfficiencyEnv — Interactive Web Dashboard (Backend)
"""
import sys, os, random, re, json, math
from pathlib import Path
from collections import Counter
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "token_efficiency_env"))
from prompts import PROMPT_BANK

app = FastAPI(title="TokenEfficiencyEnv Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Constants ──────────────────────────────────────────────────────
MAX_TOKEN_LIMIT = 200
COMPLEXITY_BUDGET = {"easy": 0.15, "medium": 0.45, "hard": 0.80}
CURRICULUM_PHASES = [
    {"name": "Phase 1: Easy Only", "weights": {"easy": 1.0, "medium": 0.0, "hard": 0.0}, "threshold": 0.4},
    {"name": "Phase 2: Easy + Medium", "weights": {"easy": 0.6, "medium": 0.4, "hard": 0.0}, "threshold": 0.5},
    {"name": "Phase 3: Mixed", "weights": {"easy": 0.3, "medium": 0.4, "hard": 0.3}, "threshold": 0.6},
    {"name": "Phase 4: Full Difficulty", "weights": {"easy": 0.2, "medium": 0.4, "hard": 0.4}, "threshold": None},
]

# ─── Local Scorer (no API key needed) ───────────────────────────────
def count_tokens(text):
    return max(1, int(len(text.split()) * 1.3))

def local_score(prompt, response, budget, tokens_used, complexity, keywords):
    details = {}
    # 1. Correctness via keywords (40%)
    if keywords:
        matches = sum(1 for kw in keywords if kw.lower() in response.lower())
        correctness = matches / len(keywords)
    else:
        correctness = 0.5
    details["correctness"] = round(correctness, 4)

    # 2. Efficiency (20%)
    if budget <= 0:
        efficiency = 0.0
    elif tokens_used <= budget:
        efficiency = 1.0 - (tokens_used / budget) * 0.3
    else:
        efficiency = max(-0.5, -0.5 * ((tokens_used - budget) / budget))
    details["efficiency"] = round(efficiency, 4)

    # 3. Budget Reasonableness (10%)
    expected = COMPLEXITY_BUDGET.get(complexity, 0.45)
    allocated = min(budget / 200, 1.0)
    budget_r = max(0.0, 1.0 - abs(allocated - expected))
    details["budget_reasonableness"] = round(budget_r, 4)

    # 4. Redundancy (10%)
    words = response.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if len(words) >= 2:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bp = min(Counter(bigrams).most_common(1)[0][1] / max(len(bigrams),1), 1.0)
        else:
            bp = 0.0
        redundancy = unique_ratio * 0.7 + (1.0 - bp) * 0.3
    else:
        redundancy = 0.0
    details["redundancy"] = round(redundancy, 4)

    # 5. Self-Assessment (10%)
    if budget > 0 and tokens_used > 0:
        self_a = max(0.0, 1.0 - abs(budget - tokens_used) / budget)
    else:
        self_a = 0.0
    details["self_assessment"] = round(self_a, 4)

    # 6. Keyword Verification (5%)
    if keywords:
        kw_score = sum(1 for kw in keywords if kw.lower() in response.lower()) / len(keywords)
    else:
        kw_score = 1.0
    details["keyword_verification"] = round(kw_score, 4)

    # 7. Format Quality (5%)
    fmt = 1.0
    if "  " in response or response != response.strip():
        fmt -= 0.2
    if complexity in ("medium","hard") and len(words) < 5:
        fmt -= 0.3
    if complexity == "easy" and len(words) > 30:
        fmt -= 0.3
    pw = set(prompt.lower().split())
    aw = set(response.lower().split())
    if len(aw) > 0 and len(pw & aw) / len(aw) > 0.7:
        fmt -= 0.3
    fmt = max(0.0, min(1.0, fmt))
    details["format_quality"] = round(fmt, 4)

    reward = round(0.40*correctness + 0.20*efficiency + 0.10*budget_r + 0.10*redundancy + 0.10*self_a + 0.05*kw_score + 0.05*fmt, 4)
    details["final_reward"] = reward
    return {"reward": reward, "details": details}

# ─── Environment State ──────────────────────────────────────────────
class EnvState:
    def __init__(self):
        self.reset_all()
    def reset_all(self):
        self.episode = 0
        self.phase = 0
        self.recent_rewards = []
        self.current_task = None
        self.history = []
        self.pools = {
            "easy": [p for p in PROMPT_BANK if p["complexity"] == "easy"],
            "medium": [p for p in PROMPT_BANK if p["complexity"] == "medium"],
            "hard": [p for p in PROMPT_BANK if p["complexity"] == "hard"],
        }

env = EnvState()

def select_question(phase_idx):
    weights = CURRICULUM_PHASES[phase_idx]["weights"]
    candidates = []
    for comp, w in weights.items():
        if w > 0:
            candidates.extend(random.choices(env.pools[comp], k=max(1, int(w*10))))
    return random.choice(candidates) if candidates else random.choice(PROMPT_BANK)

def maybe_advance():
    if env.phase >= len(CURRICULUM_PHASES) - 1:
        return
    if len(env.recent_rewards) < 10:
        return
    avg = sum(env.recent_rewards[-50:]) / min(len(env.recent_rewards), 50)
    thr = CURRICULUM_PHASES[env.phase]["threshold"]
    if thr and avg >= thr:
        env.phase += 1

# ─── Simulated LLM Responses ───────────────────────────────────────
GOOD_ANSWERS = {
    "What is 15% of 200?": ("15", "30."),
    "What is the capital of France?": ("20", "Paris."),
    "What does CPU stand for?": ("20", "Central Processing Unit."),
    "What is 2 to the power of 8?": ("15", "256."),
    "What color do you get mixing red and blue?": ("15", "Purple."),
    "How many sides does a hexagon have?": ("15", "Six sides."),
    "What is the boiling point of water in Celsius?": ("15", "100 degrees Celsius."),
    "Who wrote Romeo and Juliet?": ("20", "William Shakespeare."),
    "Explain what gravity is.": ("80", "Gravity is a fundamental force of attraction between objects with mass. Greater mass means stronger gravitational pull."),
    "What is the difference between RAM and ROM?": ("80", "RAM is volatile memory for temporary data that is lost when powered off. ROM is non-volatile read-only memory that retains data permanently."),
    "How does a vaccine work?": ("80", "Vaccines introduce weakened or inactive pathogens to stimulate the immune system to produce antibodies, creating immunity without causing the disease."),
    "What causes seasons on Earth?": ("80", "Seasons are caused by Earth's 23.5-degree axial tilt. As Earth orbits the sun, different hemispheres receive varying amounts of direct sunlight."),
    "Explain what inflation means.": ("80", "Inflation is the sustained increase in general price levels over time, reducing the purchasing power of money."),
    "What is the difference between speed and velocity?": ("80", "Speed is a scalar measuring how fast something moves. Velocity is a vector that includes both speed and direction."),
    "How does the internet work in simple terms?": ("80", "The internet is a global network of computers that exchange data through servers using standardized protocols, routing information in packets."),
    "What is photosynthesis?": ("80", "Photosynthesis is the process by which plants convert light energy, water, and carbon dioxide into glucose and oxygen for energy."),
    "Explain how transformers work in machine learning.": ("150", "Transformers use self-attention mechanisms to process all tokens in parallel rather than sequentially. Each layer computes attention scores between token pairs, allowing the model to capture long-range dependencies. The architecture uses multi-head attention, layer normalization, and feed-forward networks."),
    "What are the causes and effects of climate change?": ("150", "Climate change is primarily caused by greenhouse gas emissions, especially carbon dioxide from burning fossil fuels. Effects include rising global temperatures, melting ice caps, sea level rise, and extreme weather events."),
    "Explain the difference between supervised and unsupervised learning.": ("150", "Supervised learning trains on labeled data to predict outputs, used for classification and regression. Unsupervised learning finds patterns in unlabeled data through clustering and dimensionality reduction without predefined targets."),
    "How does the immune system fight a virus?": ("150", "When a pathogen enters the body, innate immune cells respond first. Then adaptive immunity activates: T-cells destroy infected cells while B-cells produce antibodies that neutralize the pathogen and provide future immunity."),
    "Explain what quantum entanglement is.": ("150", "Quantum entanglement is a phenomenon where two particles become correlated so that measuring the state of one instantly determines the state of the other, regardless of distance. This non-local connection challenges classical physics."),
    "What is the significance of the Turing Test?": ("150", "The Turing Test, proposed by Alan Turing, evaluates machine intelligence by testing if a human judge can distinguish between a machine and human in conversation. It remains a foundational benchmark for artificial intelligence."),
    "How do neural networks learn from data?": ("150", "Neural networks learn by adjusting connection weights through backpropagation. During training, the network makes predictions, calculates error via a loss function, then computes gradients to update weights, gradually minimizing prediction error."),
    "Explain the theory of relativity in simple terms.": ("150", "Einstein's theory has two parts. Special relativity says space and time are linked and nothing travels faster than light. General relativity explains gravity as the curvature of spacetime caused by mass and energy."),
}

def simulated_response(task, skill):
    prompt = task["prompt"]
    good = GOOD_ANSWERS.get(prompt)
    if not good:
        return f"<budget>50</budget><answer>I'm not sure about this question.</answer>"
    good_budget, good_answer = good
    r = random.random()
    if skill < 0.15:
        return f"The answer to this question is something I need to think about carefully."
    elif skill < 0.3:
        return f"<budget>190</budget><answer>Well, that is a very interesting question. Let me think about this carefully. {good_answer} I hope that helps you understand the topic better and provides the information you were looking for.</answer>"
    elif skill < 0.5:
        bad_budget = str(int(float(good_budget) * random.uniform(1.5, 3.0)))
        return f"<budget>{bad_budget}</budget><answer>{good_answer} This is an important topic to understand.</answer>"
    elif skill < 0.7:
        ok_budget = str(int(float(good_budget) * random.uniform(1.0, 1.5)))
        return f"<budget>{ok_budget}</budget><answer>{good_answer}</answer>"
    else:
        return f"<budget>{good_budget}</budget><answer>{good_answer}</answer>"

# ─── Request Models ─────────────────────────────────────────────────
class StepRequest(BaseModel):
    raw_response: str

class SimulateRequest(BaseModel):
    num_episodes: int = 50

# ─── API Endpoints ──────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/api/questions")
async def get_questions():
    return [{"prompt": p["prompt"], "complexity": p["complexity"], "keywords": p.get("expected_keywords", [])} for p in PROMPT_BANK]

@app.post("/api/reset")
async def reset(question_index: Optional[int] = None):
    maybe_advance()
    if question_index is not None and 0 <= question_index < len(PROMPT_BANK):
        env.current_task = PROMPT_BANK[question_index]
    else:
        env.current_task = select_question(env.phase)
    env.episode += 1
    return {
        "episode": env.episode, "phase": CURRICULUM_PHASES[env.phase]["name"],
        "complexity": env.current_task["complexity"], "prompt": env.current_task["prompt"],
        "keywords": env.current_task.get("expected_keywords", []), "token_limit": MAX_TOKEN_LIMIT,
    }

@app.post("/api/step")
async def step(req: StepRequest):
    if not env.current_task:
        return JSONResponse({"error": "Call /api/reset first"}, 400)
    raw = req.raw_response
    budget_m = re.search(r"<budget>(\d+)</budget>", raw)
    answer_m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if not budget_m or not answer_m:
        env.recent_rewards.append(-1.0)
        return {"reward": -1.0, "error": "Bad format — missing <budget> or <answer> tags", "details": {}, "anti_hack": "bad_format"}
    budget = max(1, min(200, int(budget_m.group(1))))
    answer = answer_m.group(1).strip()
    if not answer:
        env.recent_rewards.append(-1.0)
        return {"reward": -1.0, "error": "Empty answer", "details": {}, "anti_hack": "empty"}
    words = answer.lower().split()
    if len(words) > 3:
        wc = Counter(words)
        top_w, top_c = wc.most_common(1)[0]
        if top_c / len(words) > 0.6:
            env.recent_rewards.append(-0.5)
            return {"reward": -0.5, "error": f"Repetitive: '{top_w}' repeated {top_c}/{len(words)}", "details": {}, "anti_hack": "repetition"}
    tokens_used = count_tokens(answer)
    if tokens_used > 500:
        env.recent_rewards.append(-0.5)
        return {"reward": -0.5, "error": f"Too long: {tokens_used} tokens", "details": {}, "anti_hack": "too_long"}
    result = local_score(env.current_task["prompt"], answer, budget, tokens_used, env.current_task["complexity"], env.current_task.get("expected_keywords", []))
    env.recent_rewards.append(result["reward"])
    if len(env.recent_rewards) > 100:
        env.recent_rewards = env.recent_rewards[-50:]
    avg = sum(env.recent_rewards[-50:]) / min(len(env.recent_rewards), 50)
    return {
        "reward": result["reward"], "details": result["details"], "tokens_used": tokens_used,
        "budget": budget, "answer": answer, "avg_reward": round(avg, 4),
        "episode": env.episode, "phase": CURRICULUM_PHASES[env.phase]["name"],
    }

@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    env.reset_all()
    results = []
    n = req.num_episodes
    for i in range(n):
        # Skill progresses naturally with training (like real RL)
        skill = min(1.0, (i / n) * 1.1 + random.uniform(-0.05, 0.05))
        maybe_advance()
        task = select_question(env.phase)
        env.current_task = task
        env.episode += 1
        raw = simulated_response(task, skill)
        budget_m = re.search(r"<budget>(\d+)</budget>", raw)
        answer_m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        if not budget_m or not answer_m:
            reward = -1.0
            details = {}
            tokens_used = 0
            budget = 0
            answer = raw
        else:
            budget = max(1, min(200, int(budget_m.group(1))))
            answer = answer_m.group(1).strip()
            tokens_used = count_tokens(answer)
            r = local_score(task["prompt"], answer, budget, tokens_used, task["complexity"], task.get("expected_keywords", []))
            reward = r["reward"]
            details = r["details"]
        env.recent_rewards.append(reward)
        if len(env.recent_rewards) > 100:
            env.recent_rewards = env.recent_rewards[-50:]
        avg = sum(env.recent_rewards[-50:]) / min(len(env.recent_rewards), 50)
        results.append({
            "episode": env.episode, "reward": reward, "avg_reward": round(avg, 4),
            "phase": CURRICULUM_PHASES[env.phase]["name"], "complexity": task["complexity"],
            "prompt": task["prompt"][:50], "answer": answer[:80], "budget": budget,
            "tokens_used": tokens_used, "skill": round(skill, 3), "details": details,
        })
    return results

@app.get("/api/state")
async def get_state():
    avg = 0.0
    if env.recent_rewards:
        avg = sum(env.recent_rewards[-50:]) / min(len(env.recent_rewards), 50)
    return {
        "episode": env.episode, "phase": CURRICULUM_PHASES[env.phase]["name"],
        "phase_index": env.phase, "avg_reward": round(avg, 4),
        "total_rewards": len(env.recent_rewards),
    }

if __name__ == "__main__":
    import uvicorn
    print("\n  Dashboard running at http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
