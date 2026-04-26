"""Deploy ``token_efficiency_env`` to a Hugging Face Space.

Why this script exists instead of just calling ``openenv push``:

1. **We want a judge-friendly README on the Space.** The package's own
   ``token_efficiency_env/README.md`` is the *contributor* guide — useful
   but not what a hackathon reviewer wants to see on the Space landing
   page. ``openenv push`` would auto-inject a generic frontmatter into
   it; we'd rather ship our own curated ``deploy/SPACE_README.md`` with
   proper title, emoji, pitch, and usage.

2. **We don't want to require ``openenv`` CLI installed.** This script
   uses ``huggingface_hub.HfApi`` directly, so anyone with
   ``pip install huggingface_hub`` and an HF token can deploy — no need
   to pin a separate CLI version or deal with its staging machinery.

3. **--dry-run support.** We can validate everything we'd upload
   *without* actually creating a Space, which is the only safe way to
   test a deploy pipeline in CI.

Usage::

    # Dry run — validate structure, show the would-be repo id, upload nothing.
    python deploy/push_to_hf_space.py --dry-run

    # Deploy to <your-hf-username>/token-efficiency-env (default repo id from openenv.yaml).
    python deploy/push_to_hf_space.py

    # Deploy to a specific repo.
    python deploy/push_to_hf_space.py --repo-id acme/tokeneff-env

    # Deploy as private.
    python deploy/push_to_hf_space.py --private

    # Override the base Docker image (HF pulls this automatically).
    python deploy/push_to_hf_space.py --base-image ghcr.io/meta-pytorch/openenv-base:latest

Prerequisites:

* ``HF_TOKEN`` environment variable, OR be logged in via
  ``huggingface-cli login``. The script checks and fails fast.
* ``pip install huggingface_hub pyyaml``.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger("push_to_hf_space")


_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_DIR = _REPO_ROOT / "token_efficiency_env"
_SPACE_README_PATH = Path(__file__).resolve().parent / "SPACE_README.md"

# Files we NEVER want uploaded to an HF Space, even if someone adds them
# to the env directory later. Kept as literal path fragments so we can
# match with ``fnmatch``.
#
# The Space only needs: env source, Dockerfile, pyproject/uv.lock,
# openenv.yaml, README. Everything else (tests, CI caches, Jupyter
# checkpoints) adds bloat + confuses judges browsing the repo.
DEFAULT_IGNORE_PATTERNS: Sequence[str] = (
    ".*",                 # hidden files (.git, .venv, .env, ...)
    "__pycache__",        # pyc caches at any depth
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    "*.ipynb_checkpoints",
    "tests",              # env unit tests live with the env in dev, not on Space
    "test_*.py",
    "*_test.py",
)


# ─── Data classes ─────────────────────────────────────────────────────
@dataclass
class PushPlan:
    """What a push invocation intends to do; also useful to display under ``--dry-run``."""

    env_dir: Path
    space_readme: Path
    repo_id: str
    private: bool
    base_image: Optional[str]
    hardware: Optional[str]
    files_to_upload: List[Path]


# ─── Helpers ──────────────────────────────────────────────────────────
def _load_env_name() -> str:
    """Read the env name from ``token_efficiency_env/openenv.yaml``."""
    import yaml  # Local import so the module is importable without pyyaml.

    manifest_path = _ENV_DIR / "openenv.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"openenv.yaml not found at {manifest_path}. "
            "Is this repo a valid OpenEnv environment?"
        )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest.get("name"):
        raise ValueError("openenv.yaml must contain a top-level 'name' field.")
    return str(manifest["name"])


def _should_ignore(rel_path: Path, patterns: Sequence[str]) -> bool:
    """Return True when *any* part of ``rel_path`` matches an ignore pattern."""
    parts = rel_path.parts
    name = rel_path.name
    posix = rel_path.as_posix()
    for pat in patterns:
        if fnmatch(name, pat) or fnmatch(posix, pat):
            return True
        # Directory-name matches: if any path component matches the pattern,
        # everything under it is ignored.
        if any(fnmatch(part, pat) for part in parts):
            return True
    return False


def _iter_upload_files(env_dir: Path, patterns: Sequence[str]) -> List[Path]:
    """Enumerate files that would be uploaded, relative to ``env_dir``."""
    out: List[Path] = []
    for p in sorted(env_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(env_dir)
        if _should_ignore(rel, patterns):
            continue
        out.append(rel)
    return out


def _stage_for_upload(
    env_dir: Path,
    space_readme: Path,
    staging_root: Path,
    patterns: Sequence[str],
) -> Path:
    """Copy env files to a staging dir, swap README for the Space-facing one.

    Returns the staging directory path (which will sit inside ``staging_root``).
    """
    staging = staging_root / "env"
    staging.mkdir(parents=True, exist_ok=True)

    for rel in _iter_upload_files(env_dir, patterns):
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env_dir / rel, dest)

    # Overwrite README.md with SPACE_README.md so the Space landing page is
    # the judge-facing version, not the contributor guide.
    readme_text = space_readme.read_text(encoding="utf-8")
    (staging / "README.md").write_text(readme_text, encoding="utf-8")

    # Dockerfile must live at the repo root on HF Spaces. The env keeps it
    # under ``server/`` for local builds; copy a mirror to root.
    src_dockerfile = staging / "server" / "Dockerfile"
    dst_dockerfile = staging / "Dockerfile"
    if src_dockerfile.exists() and not dst_dockerfile.exists():
        shutil.copy2(src_dockerfile, dst_dockerfile)

    # OpenEnv's ``create_app()`` only mounts the ``/web`` interactive UI
    # when ``ENABLE_WEB_INTERFACE=true`` is in the environment. Our
    # ``SPACE_README.md`` frontmatter sets ``base_path: /web`` — so
    # without this env var the Space's landing page 404s.
    # Inject the flag into the staged Dockerfile (NOT the one committed
    # to your repo, which should stay off-by-default for local safety).
    if dst_dockerfile.exists():
        text = dst_dockerfile.read_text(encoding="utf-8")
        if "ENABLE_WEB_INTERFACE" not in text:
            # Insert right before the final CMD line so it applies to
            # the runtime container, not the builder.
            lines = text.splitlines()
            inserted = False
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].lstrip().startswith("CMD "):
                    lines.insert(i, "ENV ENABLE_WEB_INTERFACE=true")
                    inserted = True
                    break
            if not inserted:
                lines.append("ENV ENABLE_WEB_INTERFACE=true")
            dst_dockerfile.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return staging


def _build_plan(
    *,
    repo_id: Optional[str],
    private: bool,
    base_image: Optional[str],
    hardware: Optional[str],
    env_dir: Path = _ENV_DIR,
    space_readme: Path = _SPACE_README_PATH,
) -> PushPlan:
    """Assemble a :class:`PushPlan` from config + filesystem state.

    Pure: does NOT touch the network. Used by both ``--dry-run`` and the
    real push path, and is the primary test surface.
    """
    if not env_dir.exists():
        raise FileNotFoundError(
            f"Env directory not found: {env_dir}. "
            "This script expects to be run from the repo root."
        )
    if not space_readme.exists():
        raise FileNotFoundError(f"Space README not found at {space_readme}.")

    env_name = _load_env_name()

    # Default repo-id: <whoever-we-are>/<env-name>. When not supplied and we
    # can't authenticate (e.g. dry-run on CI), fall back to a placeholder.
    if repo_id is None:
        username = _try_whoami() or "<YOUR-USERNAME>"
        repo_id = f"{username}/{env_name}"

    files = _iter_upload_files(env_dir, DEFAULT_IGNORE_PATTERNS)

    return PushPlan(
        env_dir=env_dir,
        space_readme=space_readme,
        repo_id=repo_id,
        private=private,
        base_image=base_image,
        hardware=hardware,
        files_to_upload=files,
    )


def _try_whoami() -> Optional[str]:
    """Best-effort HF identity lookup. Returns None if we can't tell."""
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


def _describe_plan(plan: PushPlan) -> str:
    """Pretty-print the plan for dry-run output."""
    lines = [
        "=" * 68,
        "  HF Space deploy plan",
        "=" * 68,
        f"  env dir       : {plan.env_dir}",
        f"  Space README  : {plan.space_readme.name}",
        f"  repo id       : {plan.repo_id}",
        f"  private       : {plan.private}",
        f"  base image    : {plan.base_image or '(from Dockerfile FROM)'}",
        f"  hardware      : {plan.hardware or '(HF default — cpu-basic)'}",
        f"  files to push : {len(plan.files_to_upload)}",
    ]
    for rel in plan.files_to_upload[:20]:
        lines.append(f"      - {rel.as_posix()}")
    if len(plan.files_to_upload) > 20:
        lines.append(f"      ... and {len(plan.files_to_upload) - 20} more")
    lines.append("=" * 68)
    return "\n".join(lines)


def _ensure_hf_login() -> str:
    """Return the HF username, erroring clearly if we're not logged in."""
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
            "Not authenticated with Hugging Face. Either set HF_TOKEN in your "
            "environment or run `huggingface-cli login` first."
        ) from e
    username = (
        info.get("name") if isinstance(info, dict) else getattr(info, "name", None)
    )
    if not username:
        raise RuntimeError(
            "Could not extract username from HF whoami response. Re-login."
        )
    return str(username)


def _do_push(plan: PushPlan) -> str:
    """Perform the actual HF Space create + upload. Returns the Space URL."""
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
        staging = _stage_for_upload(
            plan.env_dir,
            plan.space_readme,
            Path(tmp),
            DEFAULT_IGNORE_PATTERNS,
        )
        logger.info("Uploading %d files from staging dir: %s",
                    sum(1 for _ in staging.rglob("*") if _.is_file()),
                    staging)
        api.upload_folder(
            folder_path=str(staging),
            repo_id=plan.repo_id,
            repo_type="space",
            commit_message="Deploy TokenEfficiencyEnv to Space",
        )

    return f"https://huggingface.co/spaces/{plan.repo_id}"


# ─── CLI ──────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="push_to_hf_space.py",
        description="Deploy token_efficiency_env to a Hugging Face Space.",
    )
    parser.add_argument(
        "--repo-id", default=None,
        help="Target repo id 'username/space-name'. Defaults to <whoami>/<name-from-openenv.yaml>.",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create the Space as private.",
    )
    parser.add_argument(
        "--base-image", default=None,
        help="Override Dockerfile FROM (useful for pinning a specific openenv-base).",
    )
    parser.add_argument(
        "--hardware", default=None,
        help="HF hardware tier, e.g. 'cpu-basic' (free) or 't4-medium' (paid).",
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
            base_image=args.base_image,
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
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
