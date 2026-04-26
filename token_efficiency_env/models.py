"""
Data models for the TokenEfficiencyEnv environment.

The environment trains an LLM to answer questions correctly using fewer tokens
by self-allocating a token budget per question. The model's response must be
formatted as: ``<budget>N</budget><answer>text</answer>``.

Action:
    raw_response: The full XML-tagged response string the model produced.

Observation:
    prompt:               The question the model should answer.
    episode_token_limit:  Hard upper bound on the budget the model may allocate.
    answer:               Parsed answer text (populated after step()).
    allocated_budget:     Budget the model self-allocated (clamped to [1, 200]).
    tokens_used:          Token count of the full raw response (v0.3.0 Layer A).
    complexity:           "easy" | "medium" | "hard".
    phase:                Curriculum phase name.
    episode:              Episode index.
    avg_reward_50:        Rolling average reward over the last 50 episodes.
    reward_components:    Per-component score breakdown from the 6-component scorer.
    error:                Non-empty short tag when an anti-hacking guard tripped:
                          "bad_format", "empty", "parrot", "repetition", "too_long".
    answer_token_count:   Token count of the inner answer only. Diagnostic, does
                          NOT feed back into the reward. Use this for dashboards
                          that need to display inner-vs-total budget breakdown.
"""

from typing import Dict, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field

# ─── Constants exposed to the trainer / clients ─────────────────────────
MAX_TOKEN_LIMIT = 200


class TokenEfficiencyAction(Action):
    """Action sent by the trainer: the model's full XML-tagged response."""

    raw_response: str = Field(
        ...,
        description=(
            "The model's full response, expected to follow the format "
            "'<budget>N</budget><answer>text</answer>'."
        ),
    )


class TokenEfficiencyObservation(Observation):
    """Observation returned to the trainer after reset() and step().

    Note: ``done``, ``reward``, and ``metadata`` are inherited from the base
    ``Observation`` class. ``metadata`` is stripped on the wire by OpenEnv's
    serializer, so anything the trainer needs to read must be a real field.
    """

    # Always present (after reset and step)
    prompt: str = Field(
        default="",
        description="The question the agent should answer.",
    )
    episode_token_limit: int = Field(
        default=MAX_TOKEN_LIMIT,
        description="Hard upper bound on the budget the model may allocate.",
    )

    # Populated after step()
    answer: str = Field(
        default="",
        description="Parsed answer text extracted from <answer>...</answer>.",
    )
    allocated_budget: int = Field(
        default=0,
        description="Budget the model self-allocated (clamped to [1, 200]).",
    )
    tokens_used: int = Field(
        default=0,
        description=(
            "Token count of the full raw response (v0.3.0 Layer A of the "
            "hidden-CoT fix — includes wrapper tokens and any reasoning "
            "the model tried to leak outside the tags). For the "
            "inner-answer-only count see ``answer_token_count``."
        ),
    )
    complexity: str = Field(
        default="",
        description='Question complexity: "easy", "medium", or "hard".',
    )
    phase: str = Field(
        default="",
        description="Current curriculum phase name.",
    )
    episode: int = Field(
        default=0,
        description="Episode index for this session.",
    )
    avg_reward_50: float = Field(
        default=0.0,
        description="Rolling average reward over the last 50 episodes.",
    )
    reward_components: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component scores from the 6-component scorer.",
    )
    error: str = Field(
        default="",
        description=(
            "Non-empty short tag when an anti-hacking guard tripped: "
            '"bad_format", "empty", "parrot", "repetition", or "too_long".'
        ),
    )
    answer_token_count: Optional[int] = Field(
        default=None,
        description="Token count of inner answer only (diagnostic — §1 Layer B)",
    )
