"""
DEPRECATED — kept only for backward compatibility.

The real environment now lives in
``token_efficiency_env.server.token_efficiency_env_environment``. This module
re-exports it under the legacy ``TokenEfficiencyEnv`` name, plus the legacy
constants and curriculum config, so any old code or notebook that does
``from env_server import TokenEfficiencyEnv`` keeps working.

New code should import from the package directly:

    from token_efficiency_env.server.token_efficiency_env_environment import (
        TokenEfficiencyEnvironment,
    )
    from token_efficiency_env.models import (
        TokenEfficiencyAction,
        TokenEfficiencyObservation,
    )
"""

from __future__ import annotations

import warnings

from .models import (
    MAX_TOKEN_LIMIT,
    TokenEfficiencyAction,
    TokenEfficiencyObservation,
)
from .server.token_efficiency_env_environment import (
    CURRICULUM_PHASES,
    MAX_ANSWER_TOKENS,
    MAX_BUDGET,
    MIN_BUDGET,
    REPETITION_THRESHOLD,
    REWARD_WINDOW,
    TokenEfficiencyEnvironment as TokenEfficiencyEnv,
)

warnings.warn(
    "token_efficiency_env.env_server is deprecated. Import "
    "TokenEfficiencyEnvironment from "
    "token_efficiency_env.server.token_efficiency_env_environment instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "TokenEfficiencyEnv",
    "TokenEfficiencyAction",
    "TokenEfficiencyObservation",
    "MAX_TOKEN_LIMIT",
    "MIN_BUDGET",
    "MAX_BUDGET",
    "MAX_ANSWER_TOKENS",
    "REPETITION_THRESHOLD",
    "REWARD_WINDOW",
    "CURRICULUM_PHASES",
]
