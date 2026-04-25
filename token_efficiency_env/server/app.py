# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Token Efficiency Env Environment.

This module creates an HTTP server that exposes the TokenEfficiencyEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import TokenEfficiencyAction, TokenEfficiencyObservation
    from .token_efficiency_env_environment import TokenEfficiencyEnvironment
except ModuleNotFoundError:
    from models import TokenEfficiencyAction, TokenEfficiencyObservation
    from server.token_efficiency_env_environment import TokenEfficiencyEnvironment


# Create the app with web interface and README integration
app = create_app(
    TokenEfficiencyEnvironment,
    TokenEfficiencyAction,
    TokenEfficiencyObservation,
    env_name="token_efficiency_env",
    # IMPORTANT: keep this at 1.
    # OpenEnv's HTTP layer round-robins requests across this pool, with NO
    # session affinity, so /reset can land on instance #3 and the matching
    # /step on instance #5 — meaning the agent gets scored on a different
    # question than the one it was asked. We rely on the curriculum +
    # recent_rewards deque persisting across requests, so we need exactly
    # one shared instance per server process.
    #
    # For parallel-rollout GRPO, run N separate server processes on N ports,
    # each at max_concurrent_envs=1, and let the trainer load-balance.
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m token_efficiency_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn token_efficiency_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
