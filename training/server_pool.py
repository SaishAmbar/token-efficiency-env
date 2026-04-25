"""Spawn / health-check / shutdown a pool of TokenEfficiencyEnv uvicorn servers.

This is the OPTIONAL Mode B for ``reward_adapter.py`` — only used when
``TrainingConfig.reward_backend == "ws"``. The default training path uses
``InProcessRewardAdapter`` and never touches this file.

Why have it at all?
-------------------
Two reasons:

    1. **Production-realism smoke test.** Running 8 real uvicorn workers
       against a real WebSocket client exercises the same code path that
       the deployed HF Space will eventually hit. Catching regressions here
       is much cheaper than discovering them after deploy.
    2. **Stress test.** Confirms ``max_concurrent_envs=1`` per-process
       isolation actually holds when N trainees hammer the env in parallel.

Lifecycle
---------
    pool = ServerPool(num_envs=8, base_port=8100)
    try:
        pool.start()                    # spawns N uvicorns, blocks on health
        urls = pool.urls                # ["http://localhost:8100", ...]
        # ... train ...
    finally:
        pool.shutdown()                 # SIGTERM, then SIGKILL after grace
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import httpx

logger = logging.getLogger("token_efficiency_env.server_pool")

REPO_ROOT = Path(__file__).resolve().parent.parent
HEALTH_PATH = "/health"
SPAWN_TIMEOUT_S = 60
SHUTDOWN_GRACE_S = 5


def _port_is_free(port: int) -> bool:
    """True iff *nothing* is listening on localhost:port right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect(("127.0.0.1", port))
            return False
        except (ConnectionRefusedError, socket.timeout, OSError):
            return True


class ServerPool:
    """A pool of TokenEfficiencyEnv uvicorn workers, one per port.

    Each worker is its own OS process. The constructor only validates
    arguments — call ``start()`` to actually spawn.
    """

    def __init__(
        self,
        num_envs: int = 1,
        base_port: int = 8100,
        judge_backend: str = "huggingface",
        log_level: str = "WARNING",
        extra_env: Optional[dict] = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >=1, got {num_envs}")
        self.num_envs = num_envs
        self.base_port = base_port
        self.judge_backend = judge_backend
        self.log_level = log_level
        self.extra_env = extra_env or {}
        self.processes: List[subprocess.Popen] = []
        self.ports: List[int] = list(range(base_port, base_port + num_envs))

    @property
    def urls(self) -> List[str]:
        """Base URLs in port order. Stable across the pool's lifetime."""
        return [f"http://localhost:{p}" for p in self.ports]

    # ─── Lifecycle ─────────────────────────────────────────────────
    def start(self) -> "ServerPool":
        """Spawn N workers and block until each one's /health returns 200.

        Idempotent if you call start() twice on the same instance — the
        second call no-ops if all processes are already alive.
        """
        if self.processes:
            logger.info("ServerPool.start() called twice; skipping re-spawn.")
            return self

        # Pre-flight: ports must be free, otherwise we'd accidentally talk
        # to some leftover process from a previous run.
        for port in self.ports:
            if not _port_is_free(port):
                raise RuntimeError(
                    f"Port {port} is already in use. Free it first, e.g.\n"
                    f"  $conn = Get-NetTCPConnection -LocalPort {port} "
                    f"-ErrorAction SilentlyContinue; "
                    f"if ($conn) {{ Stop-Process -Id $conn.OwningProcess -Force }}"
                )

        env = os.environ.copy()
        env["JUDGE_BACKEND"] = self.judge_backend
        env["TOKEN_EFFICIENCY_LOG_LEVEL"] = self.log_level
        env.update({k: str(v) for k, v in self.extra_env.items()})

        for port in self.ports:
            cmd = [
                sys.executable, "-m", "uvicorn",
                "token_efficiency_env.server.app:app",
                "--host", "127.0.0.1",
                "--port", str(port),
                "--log-level", self.log_level.lower(),
            ]
            logger.info("Spawning env server on :%d", port)
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            self.processes.append(proc)

        try:
            self._wait_for_health()
        except Exception:
            # Don't leak zombie workers on startup failure.
            self.shutdown()
            raise

        logger.info("ServerPool ready: %s", self.urls)
        return self

    def _wait_for_health(self) -> None:
        """Block until every URL responds 200 on /health, or raise."""
        deadline = time.monotonic() + SPAWN_TIMEOUT_S
        pending = list(self.urls)
        with httpx.Client(timeout=2.0) as client:
            while pending and time.monotonic() < deadline:
                still_pending = []
                for url in pending:
                    # If the worker died during boot, fail loud rather than
                    # waiting the full 60 seconds.
                    proc = self.processes[self.urls.index(url)]
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"Worker for {url} exited during startup with "
                            f"code {proc.returncode}; check uvicorn logs."
                        )
                    try:
                        r = client.get(url + HEALTH_PATH)
                        if r.status_code == 200:
                            continue
                    except (httpx.RequestError, httpx.HTTPError):
                        pass
                    still_pending.append(url)
                pending = still_pending
                if pending:
                    time.sleep(0.5)
        if pending:
            raise TimeoutError(
                f"ServerPool: these workers never became healthy within "
                f"{SPAWN_TIMEOUT_S}s: {pending}"
            )

    def shutdown(self) -> None:
        """SIGTERM every worker, then SIGKILL any survivors after the grace."""
        for proc in self.processes:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + SHUTDOWN_GRACE_S
        for proc in self.processes:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Worker pid=%d did not exit gracefully; killing.", proc.pid
                )
                proc.kill()
                proc.wait()
        self.processes.clear()
        logger.info("ServerPool shut down.")

    # ─── Context-manager sugar ─────────────────────────────────────
    def __enter__(self) -> "ServerPool":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.shutdown()
