import re
import sys
import os
sys.path.insert(0, os.path.abspath('../OpenEnv/src'))

from openenv.core import Environment
from pydantic import BaseModel
from transformers import AutoTokenizer
from prompts import PROMPT_BANK
from scorer import score_answer
import random

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

class TokenEfficiencyAction(BaseModel):
    raw_response: str

class TokenEfficiencyObservation(BaseModel):
    prompt: str
    episode_token_limit: int = 200

class TokenEfficiencyEnv(Environment):

    def __init__(self):
        self.current_task = None
        self.episode_token_limit = 200

    def reset(self):
        self.current_task = random.choice(PROMPT_BANK)
        return TokenEfficiencyObservation(
            prompt=self.current_task["prompt"],
            episode_token_limit=self.episode_token_limit
        )

    def step(self, action: TokenEfficiencyAction):
        raw = action.raw_response

        # parse budget tag
        budget_match = re.search(r"<budget>(\d+)</budget>", raw)
        answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)

        if not budget_match or not answer_match:
            return {
                "observation": {"error": "bad format"},
                "reward": -1.0,
                "done": True,
                "info": {"error": "model did not follow format"}
            }

        allocated_budget = int(budget_match.group(1))
        answer = answer_match.group(1).strip()
        tokens_used = len(tokenizer.encode(answer))

        reward = score_answer(
            prompt=self.current_task["prompt"],
            response=answer,
            allocated_budget=allocated_budget,
            tokens_used=tokens_used
        )

        return {
            "observation": {
                "answer": answer,
                "tokens_used": tokens_used,
                "allocated_budget": allocated_budget
            },
            "reward": reward,
            "done": True,
            "info": {
                "tokens_used": tokens_used,
                "allocated_budget": allocated_budget,
                "correctness": reward
            }
        }

    def state(self):
        return {"current_task": self.current_task}