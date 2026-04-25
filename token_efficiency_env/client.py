"""TokenEfficiencyEnv client.

Async WebSocket client for the TokenEfficiencyEnv environment server. Each
client instance gets its own dedicated environment session on the server
(curriculum state is per-session and does not bleed across trainers).

Usage (async):

    >>> async with TokenEfficiencyEnv(base_url="http://localhost:8000") as env:
    ...     result = await env.reset()
    ...     print(result.observation.prompt)
    ...
    ...     result = await env.step(
    ...         TokenEfficiencyAction(raw_response="<budget>20</budget><answer>Paris.</answer>")
    ...     )
    ...     print(result.reward, result.observation.reward_components)

Usage (sync wrapper):

    >>> with TokenEfficiencyEnv(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset()
    ...     result = env.step(TokenEfficiencyAction(raw_response="..."))

Connecting to a Hugging Face Space:

    >>> env = await TokenEfficiencyEnv.from_env("your-username/token-efficiency-env")
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TokenEfficiencyAction, TokenEfficiencyObservation


class TokenEfficiencyEnv(
    EnvClient[TokenEfficiencyAction, TokenEfficiencyObservation, State]
):
    """Client for the TokenEfficiencyEnv environment."""

    def _step_payload(self, action: TokenEfficiencyAction) -> Dict[str, Any]:
        return {"raw_response": action.raw_response}

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[TokenEfficiencyObservation]:
        obs_data = payload.get("observation", {}) or {}
        reward = payload.get("reward")
        done = payload.get("done", False)

        observation = TokenEfficiencyObservation(
            prompt=obs_data.get("prompt", ""),
            episode_token_limit=obs_data.get("episode_token_limit", 200),
            answer=obs_data.get("answer", ""),
            allocated_budget=obs_data.get("allocated_budget", 0),
            tokens_used=obs_data.get("tokens_used", 0),
            complexity=obs_data.get("complexity", ""),
            phase=obs_data.get("phase", ""),
            episode=obs_data.get("episode", 0),
            avg_reward_50=obs_data.get("avg_reward_50", 0.0),
            reward_components=obs_data.get("reward_components", {}) or {},
            error=obs_data.get("error", ""),
            done=done,
            reward=reward,
        )

        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
