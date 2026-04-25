"""Training scaffold for TokenEfficiencyEnv.

Lives outside the ``token_efficiency_env`` package on purpose: the env package
stays a pure environment with no ML-framework dependencies, while this
package owns the trainer-side glue (config, prompt split, env-server pool,
reward adapter that bridges the env into TRL's reward_funcs API).

Phase 7 entrypoint: ``notebooks/train_grpo.ipynb`` consumes everything here.
"""

from .config import TrainingConfig
from .prompts_split import (
    HOLDOUT_INDICES,
    TRAIN_INDICES,
    holdout_prompts,
    train_prompts,
)
from .reward_adapter import (
    InProcessRewardAdapter,
    RewardLog,
    WSRewardAdapter,
    build_reward_func,
)
from .server_pool import ServerPool

__all__ = [
    "TrainingConfig",
    "HOLDOUT_INDICES",
    "TRAIN_INDICES",
    "train_prompts",
    "holdout_prompts",
    "InProcessRewardAdapter",
    "WSRewardAdapter",
    "RewardLog",
    "build_reward_func",
    "ServerPool",
]
