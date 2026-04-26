"""Deploy the *Token Efficiency Playground* (web_dashboard.py) to a HF Space.

This is a sibling script to ``push_to_hf_space.py``, which deploys the
**OpenEnv environment server** (port 8000, headless protocol API). The
two ship to *different* Spaces and serve different audiences:

* ``push_to_hf_space.py``           → token-efficiency-env  (judge-facing
                                       protocol server, OpenEnv ``/web``)
* ``push_dashboard_to_hf_space.py`` → token-efficiency-playground
                                       (the redesigned light-theme UI
                                       under ``web_dashboard.py``)

Why a separate Space (and a separate script):

1. **Different SDKs at the manifest level.** The env Space's
   ``app_port`` is 8000; the dashboard Space is 7860 (HF default). The
   Dockerfiles diverge too — the env uses ``openenv-base``, while the
   dashboard uses a slim Python image for faster cold-starts.

2. **Different security posture.** The dashboard is *purely* a demo;
   it does not need to expose the OpenEnv WebSocket protocol, which
   keeps its surface area smaller.

3. **Reviewers can pick which one to demo.** The env Space is the
   protocol-correct deliverable. The Playground Space is the "wow"
   visual demo. Both are useful; neither replaces the other.

Usage::

    # Dry run — describe the upload without touching HF.
    python deploy/push_dashboard_to_hf_space.py --dry-run

    # Default: <whoami>/token-efficiency-playground, public, free hardware.
    python deploy/push_dashboard_to_hf_space.py

    # Explicit repo id.
    python deploy/push_dashboard_to_hf_space.py --repo-id myname/my-playground

    # Private + paid hardware.
    python deploy/push_dashboard_to_hf_space.py --private --hardware cpu-upgrade

Prerequisites:

* ``HF_TOKEN`` env var **with write scope**, OR be logged in via
  ``huggingface-cli login`` with a write token. Read-only tokens cannot
  create Spaces.
* ``pip install huggingface_hub``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger("push_dashboard_to_hf_space")


# ─── Layout ──────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD_FILE = _REPO_ROOT / "web_dashboard.py"
_DASHBOARD_HTML = _REPO_ROOT / "dashboard.html"
_PACKAGE_DIR = _REPO_ROOT / "token_efficiency_env"
_TEMPLATE_DIR = Path(__file__).resolve().parent / "dashboard"

DEFAULT_REPO_NAME = "token-efficiency-playground"

# Files we never want shipped to the Space, regardless of source.
# Same conservative default as push_to_hf_space.py — keeps the Space
# repo small and reviewer-friendly.
DEFAULT_IGNORE_PATTERNS: Sequence[str] = (
    ".*",                  # hidden files (.git, .env, .venv, .idea, ...)
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    "*.ipynb_checkpoints",
    "tests",
    "test_*.py",
    "*_test.py",
)


@dataclass
class DashboardPushPlan:
    """What a dashboard push intends to do; rendered under ``--dry-run``."""

    repo_id: str
    private: bool
    hardware: Optional[str]
    files_to_upload: List[Path] = field(default_factory=list)


# ─── Filesystem helpers (pure, no network) ───────────────────────────
def _should_ignore(rel_path: Path, patterns: Sequence[str]) -> bool:
    parts = rel_path.parts
    name = rel_path.name
    posix = rel_path.as_posix()
    for pat in patterns:
        if fnmatch(name, pat) or fnmatch(posix, pat):
            return True
        if any(fnmatch(part, pat) for part in parts):
            return True
    return False


def _iter_dir_files(root: Path, patterns: Sequence[str]) -> List[Path]:
    """Enumerate uploadable files under ``root``, returned relative to it."""
    out: List[Path] = []
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if _should_ignore(rel, patterns):
            continue
        out.append(rel)
    return out


def _stage_for_upload(staging_root: Path, patterns: Sequence[str]) -> Path:
    """Build the upload tree the Space will see at the repo root.

    Layout produced::

        staging/
          Dockerfile                 (from deploy/dashboard/Dockerfile)
          requirements.txt           (from deploy/dashboard/requirements.txt)
          README.md                  (from deploy/dashboard/SPACE_README.md)
          web_dashboard.py
          dashboard.html
          token_efficiency_env/...
    """
    staging = staging_root / "space"
    staging.mkdir(parents=True, exist_ok=True)

    if not _TEMPLATE_DIR.exists():
        raise FileNotFoundError(
            f"Template directory missing: {_TEMPLATE_DIR}. "
            "Did you delete deploy/dashboard/?"
        )

    # 1. Dockerfile + requirements.txt — ship verbatim.
    for name in ("Dockerfile", "requirements.txt"):
        src = _TEMPLATE_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Required template missing: {src}")
        shutil.copy2(src, staging / name)

    # 2. SPACE_README.md → README.md so HF picks up the YAML frontmatter
    #    on the Space landing page. This is what the user lands on.
    space_readme = _TEMPLATE_DIR / "SPACE_README.md"
    if not space_readme.exists():
        raise FileNotFoundError(f"SPACE_README.md missing: {space_readme}")
    shutil.copy2(space_readme, staging / "README.md")

    # 3. Application code at the repo root.
    if not _DASHBOARD_FILE.exists():
        raise FileNotFoundError(
            f"web_dashboard.py not found at {_DASHBOARD_FILE}. "
            "Run this script from a checkout that has the dashboard."
        )
    if not _DASHBOARD_HTML.exists():
        raise FileNotFoundError(
            f"dashboard.html not found at {_DASHBOARD_HTML}."
        )
    shutil.copy2(_DASHBOARD_FILE, staging / "web_dashboard.py")
    shutil.copy2(_DASHBOARD_HTML, staging / "dashboard.html")

    # 4. token_efficiency_env package — ships as a subdirectory so
    #    web_dashboard.py's ``from token_efficiency_env import …`` works
    #    inside the container without a pip install.
    if not _PACKAGE_DIR.exists():
        raise FileNotFoundError(
            f"Package directory missing: {_PACKAGE_DIR}."
        )
    pkg_dest = staging / "token_efficiency_env"
    pkg_dest.mkdir(exist_ok=True)
    for rel in _iter_dir_files(_PACKAGE_DIR, patterns):
        dst = pkg_dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_PACKAGE_DIR / rel, dst)

    return staging


def _try_whoami() -> Optional[str]:
    """Best-effort HF identity. Returns None when we can't tell."""
    try:
        from huggingface_hub import whoami  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        info = whoami()
    except Exception:
        return None
    if isinstance(info, dict):
        return info.get("name") or info.get("username") or info.get("fullname")
    return getattr(info, "name", None) or getattr(info, "username", None)


def _ensure_hf_login() -> str:
    """Return the HF username; error clearly if we're not authenticated."""
    try:
        from huggingface_hub import whoami  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is not installed. Run:\n"
            "    pip install huggingface_hub"
        ) from e
    try:
        info = whoami()
    except Exception as e:
        raise RuntimeError(
            "Not authenticated with Hugging Face. Either set HF_TOKEN "
            "(write scope) in your environment or run "
            "`huggingface-cli login` first."
        ) from e
    username = (
        info.get("name") if isinstance(info, dict) else getattr(info, "name", None)
    )
    if not username:
        raise RuntimeError(
            "Could not extract username from HF whoami response. Re-login."
        )
    return str(username)


def _build_plan(
    *,
    repo_id: Optional[str],
    private: bool,
    hardware: Optional[str],
) -> DashboardPushPlan:
    """Pure planner — does not touch the network."""
    # Default repo id: <whoami>/token-efficiency-playground; fall back to a
    # placeholder so dry-runs work without HF auth.
    if repo_id is None:
        username = _try_whoami() or "<YOUR-USERNAME>"
        repo_id = f"{username}/{DEFAULT_REPO_NAME}"

    # Compute the file list using a temporary staging dir to be precise about
    # what would actually be uploaded (rather than trying to predict it).
    with tempfile.TemporaryDirectory() as tmp:
        staging = _stage_for_upload(Path(tmp), DEFAULT_IGNORE_PATTERNS)
        files = sorted(
            p.relative_to(staging) for p in staging.rglob("*") if p.is_file()
        )

    return DashboardPushPlan(
        repo_id=repo_id,
        private=private,
        hardware=hardware,
        files_to_upload=files,
    )


def _describe_plan(plan: DashboardPushPlan) -> str:
    lines = [
        "=" * 68,
        "  HF Space deploy plan — Token Efficiency Playground",
        "=" * 68,
        f"  repo id       : {plan.repo_id}",
        f"  private       : {plan.private}",
        f"  hardware      : {plan.hardware or '(HF default — cpu-basic, free)'}",
        f"  files to push : {len(plan.files_to_upload)}",
    ]
    for rel in plan.files_to_upload[:25]:
        lines.append(f"      - {rel.as_posix()}")
    if len(plan.files_to_upload) > 25:
        lines.append(f"      ... and {len(plan.files_to_upload) - 25} more")
    lines.append("=" * 68)
    return "\n".join(lines)


def _do_push(plan: DashboardPushPlan) -> str:
    """Create or reuse the Space, then upload the staged tree."""
    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    _ensure_hf_login()
    api = HfApi()

    create_kwargs: dict = {
        "repo_id": plan.repo_id,
        "repo_type": "space",
        "space_sdk": "docker",
        "private": plan.private,
        "exist_ok": True,
    }
    if plan.hardware:
        create_kwargs["space_hardware"] = plan.hardware

    logger.info("Creating or reusing Space: %s", plan.repo_id)
    api.create_repo(**create_kwargs)

    with tempfile.TemporaryDirectory() as tmp:
        staging = _stage_for_upload(Path(tmp), DEFAULT_IGNORE_PATTERNS)
        n_files = sum(1 for _ in staging.rglob("*") if _.is_file())
        logger.info("Uploading %d files from staging dir: %s", n_files, staging)
        api.upload_folder(
            folder_path=str(staging),
            repo_id=plan.repo_id,
            repo_type="space",
            commit_message="Deploy Token Efficiency Playground",
        )

    return f"https://huggingface.co/spaces/{plan.repo_id}"


# ─── CLI ─────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="push_dashboard_to_hf_space.py",
        description="Deploy the TokenEfficiencyEnv playground UI to an HF Space.",
    )
    parser.add_argument(
        "--repo-id", default=None,
        help="Target repo id 'username/space-name'. "
             f"Defaults to <whoami>/{DEFAULT_REPO_NAME}.",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create the Space as private.",
    )
    parser.add_argument(
        "--hardware", default=None,
        help="HF hardware tier (e.g. 'cpu-basic', 'cpu-upgrade', 't4-medium').",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the deploy plan and exit without touching HF.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose logging.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    try:
        plan = _build_plan(
            repo_id=args.repo_id,
            private=args.private,
            hardware=args.hardware,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Preflight failed: %s", e)
        return 2

    print(_describe_plan(plan))

    if args.dry_run:
        print("\nDry run — no files uploaded.")
        return 0

    try:
        url = _do_push(plan)
    except RuntimeError as e:
        logger.error("Push failed: %s", e)
        return 3
    except Exception as e:  # pragma: no cover — surface HF errors loudly
        logger.exception("Unexpected error during push: %s", e)
        return 4

    print(f"\nSpace is live at: {url}")
    print("First build typically takes 4–7 minutes on cpu-basic.")
    print("Once it's running, set an HF_TOKEN secret in Space Settings to")
    print("enable the LLM judge (otherwise the keyword fallback is used).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
