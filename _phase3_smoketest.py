"""Phase 3 smoke test — exercises the new reward formulas + anti-hacking guards.

Run with:
    $env:HF_TOKEN = '<token>'
    python _phase3_smoketest.py

Tests:
  1. Perfect concise correct answer        → expect ~0.96
  2. Wrong answer                          → expect ~0.15-0.30
  3. Hack: missing <budget> tag            → expect -1.0  (bad_format)
  4. Hack: single-char answer "."          → expect -1.0  (empty)
  5. Hack: parrot the question             → expect -0.5  (parrot)
  6. Verbose-but-correct legit answer      → parrot guard MUST NOT fire
  7. Overshoot budget                      → self_assessment heavily penalised
"""
import sys
sys.path.insert(0, '.')

from token_efficiency_env.models import TokenEfficiencyAction
from token_efficiency_env.server.token_efficiency_env_environment import (
    TokenEfficiencyEnvironment,
)
import token_efficiency_env.server.token_efficiency_env_environment as ee


def fresh_env(task_idx: int) -> TokenEfficiencyEnvironment:
    env = TokenEfficiencyEnvironment()
    env.current_task = ee.PROMPT_BANK[task_idx]
    env.episode_count = 1
    return env


def show(label: str, obs):
    err = obs.error if obs.error else "(none)"
    print(f"  reward={obs.reward}  error={err}")
    if obs.reward_components:
        for k, v in obs.reward_components.items():
            print(f"    {k:24s} {v}")


def hr(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# Index 0 = boiling point, index 1 = capital of France
hr("TEST 1: Perfect concise correct answer (target ~0.96)")
env = fresh_env(1)
print(f"Q: {env.current_task['prompt']}")
obs = env.step(TokenEfficiencyAction(raw_response="<budget>20</budget><answer>Paris.</answer>"))
show("perfect", obs)

hr("TEST 2: Wrong answer (target ~0.15-0.30)")
env = fresh_env(1)
obs = env.step(TokenEfficiencyAction(raw_response="<budget>20</budget><answer>Tokyo.</answer>"))
show("wrong", obs)

hr("TEST 3: Hack — missing <budget> tag (target -1.0, error=bad_format)")
env = fresh_env(1)
obs = env.step(TokenEfficiencyAction(raw_response="<answer>Paris.</answer>"))
show("no_budget", obs)

hr("TEST 4: Hack — single-char answer (target -1.0, error=empty)")
env = fresh_env(1)
obs = env.step(TokenEfficiencyAction(raw_response="<budget>5</budget><answer>.</answer>"))
show("dot_only", obs)

hr("TEST 5: Hack — parrot the question (target -0.5, error=parrot)")
env = fresh_env(1)
obs = env.step(TokenEfficiencyAction(
    raw_response="<budget>20</budget><answer>What is the capital of France?</answer>"
))
show("parrot", obs)

hr("TEST 6: Verbose-but-correct legit answer (parrot guard MUST NOT fire)")
env = fresh_env(0)
print(f"Q: {env.current_task['prompt']}")
obs = env.step(TokenEfficiencyAction(
    raw_response="<budget>40</budget><answer>The boiling point of water is 100 degrees Celsius.</answer>"
))
show("verbose_legit", obs)

hr("TEST 7: Overshoot budget (self_assessment should heavily penalise)")
env = fresh_env(1)
verbose = (
    "The capital city of the country known as France is the beautiful city "
    "of Paris located in the Ile-de-France region in the north of the country."
)
obs = env.step(TokenEfficiencyAction(
    raw_response=f"<budget>5</budget><answer>{verbose}</answer>"
))
show("overshoot", obs)

print()
print("=" * 70)
print("Done.")
