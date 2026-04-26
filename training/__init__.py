"""Training scaffold for TokenEfficiencyEnv.

Lives outside the ``token_efficiency_env`` package on purpose: the env package
stays a pure environment with no ML-framework dependencies, while this
package owns the trainer-side glue (config, prompt split, env-server pool,
reward adapter that bridges the env into TRL's reward_funcs API).

Phase 7 entrypoint: ``notebooks/train_grpo.ipynb`` consumes everything here.
"""

from .colab_bootstrap import bootstrap, is_colab
from .compare_report import (
    build_comparison,
    plot_before_after,
    plot_reward_curve,
    render_markdown_report,
)
from .config import TrainingConfig
from .prompts_split import (
    HOLDOUT_INDICES,
    PROBE_INDICES,
    TRAIN_INDICES,
    describe_split,
    dump_split,
    holdout_prompts,
    probe_prompts,
    reset_split_cache,
    stratified_split,
    train_prompts,
)
from .reward_adapter import (
    InProcessRewardAdapter,
    RewardLog,
    WSRewardAdapter,
    build_reward_func,
)
from .server_pool import ServerPool
from .sft_format_data import SFTExample, build_format_teaching_examples

__all__ = [
    "TrainingConfig",
    "bootstrap",
    "build_comparison",
    "is_colab",
    "plot_before_after",
    "plot_reward_curve",
    "render_markdown_report",
    "HOLDOUT_INDICES",
    "PROBE_INDICES",
    "TRAIN_INDICES",
    "describe_split",
    "dump_split",
    "stratified_split",
    "reset_split_cache",
    "train_prompts",
    "holdout_prompts",
    "probe_prompts",
    "InProcessRewardAdapter",
    "WSRewardAdapter",
    "RewardLog",
    "build_reward_func",
    "ServerPool",
    "SFTExample",
    "build_format_teaching_examples",
]
