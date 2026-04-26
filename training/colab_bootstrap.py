"""Colab bootstrap helper — makes ``train_grpo.ipynb`` one-click runnable.

The notebook's very first cell calls :func:`bootstrap`, which:

* detects whether we're running on Google Colab,
* (on Colab only) ``git clone`` s this repo into ``/content`` and ``cd`` s in,
* (on Colab only) installs ``requirements-train.txt`` plus ``unsloth``,
* prompts the user for an ``HF_TOKEN`` via ``getpass`` if one isn't already
  in the environment — token is kept in ``os.environ`` for the session only,
  never written to disk.

Outside Colab (local Jupyter, CI, tests) :func:`bootstrap` is a pure
no-op so the existing editable-install workflow keeps working.

The logic lives here (not inline in the notebook cell) so that:

* tests can import ``training.colab_bootstrap`` and call ``is_colab()``
  without side effects,
* the notebook cell stays ~5 lines and remains easy to diff,
* we avoid duplicating this bootstrap in every demo notebook we add later.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_REPO_URL = "https://github.com/SaishAmbar/token-efficiency-env.git"
DEFAULT_REPO_DIR = "token-efficiency-env"
DEFAULT_WORK_ROOT = "/content"


def is_colab() -> bool:
    """Return True when imported inside a Google Colab kernel.

    Uses the documented ``google.colab`` module sentinel; no network calls,
    no side effects, safe to call from tests.
    """
    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def _run(cmd: list[str], *, cwd: Optional[str] = None) -> None:
    """Thin wrapper around ``subprocess.run`` that streams output live."""
    logger.info("bootstrap: $ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_repo(
    repo_url: str, repo_dir: str, work_root: str
) -> Path:
    """Clone ``repo_url`` under ``work_root`` if not already present, return its path."""
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / repo_dir
    if not dest.exists():
        _run(["git", "clone", "--depth", "1", repo_url, str(dest)])
    else:
        logger.info("bootstrap: repo already cloned at %s, skipping", dest)
    return dest


def _pip_install(args: list[str]) -> None:
    _run([sys.executable, "-m", "pip", "install", "--quiet", *args])


def _prompt_hf_token() -> None:
    """Set ``HF_TOKEN`` in os.environ via getpass if not already present.

    Uses ``getpass.getpass`` so the token is never echoed and never stored
    in notebook cell output.
    """
    if os.environ.get("HF_TOKEN"):
        logger.info("bootstrap: HF_TOKEN already set (judge will use real LLM).")
        return
    try:
        from getpass import getpass
    except ImportError:  # pragma: no cover — stdlib always available
        return
    print(
        "HF_TOKEN not set. Paste your HuggingFace token now "
        "(or press Enter to fall back to keyword-only scoring)."
    )
    token = getpass("HF_TOKEN: ").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ.setdefault("JUDGE_BACKEND", "huggingface")
        print("HF_TOKEN set for this session (not stored on disk).")
    else:
        os.environ.setdefault("JUDGE_BACKEND", "keyword")
        print(
            "No token provided; falling back to JUDGE_BACKEND=keyword. "
            "Training will still run but correctness signal is weaker."
        )


def bootstrap(
    repo_url: str = DEFAULT_REPO_URL,
    repo_dir: str = DEFAULT_REPO_DIR,
    work_root: str = DEFAULT_WORK_ROOT,
    install_unsloth: bool = True,
    prompt_token: bool = True,
) -> Path:
    """Prepare the current kernel for running ``train_grpo.ipynb``.

    * On Colab: clones the repo, ``cd`` s in, installs deps (+ Unsloth), and
      (optionally) prompts for ``HF_TOKEN``. Returns the cloned repo path.
    * Outside Colab: returns the current working directory unchanged.

    Parameters
    ----------
    repo_url : str
        Git URL to clone on Colab. Override if you've forked the repo.
    repo_dir : str
        Directory name under ``work_root``. Defaults to ``token-efficiency-env``.
    work_root : str
        Where to place the clone on Colab. Defaults to ``/content`` (Colab's
        home).
    install_unsloth : bool
        If True, install ``unsloth`` on Colab. Set False for CPU-only
        debugging on Colab (rare).
    prompt_token : bool
        If True and ``HF_TOKEN`` is unset, prompt for it via ``getpass``.
    """
    if not is_colab():
        logger.info("bootstrap: not on Colab, nothing to do.")
        return Path.cwd()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    repo_path = _ensure_repo(repo_url, repo_dir, work_root)
    os.chdir(repo_path)

    # ``pip install -r`` + editable install of the env package. The
    # env package's pyproject.toml lives under ``token_efficiency_env/``.
    _pip_install(["-e", "./token_efficiency_env"])
    _pip_install(["-r", "requirements-train.txt"])

    if install_unsloth:
        # Unsloth has its own wheel index / extras. Latest stable tag
        # handles torch + xformers compat internally on Colab T4/A100.
        _pip_install(["unsloth"])

    # Make ``training`` + ``token_efficiency_env`` importable without a
    # kernel restart. (pip install -e already did this; belt + braces.)
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    if prompt_token:
        _prompt_hf_token()

    print(f"\nBootstrap ready. Working directory: {repo_path}")
    return repo_path


__all__ = ["bootstrap", "is_colab"]
