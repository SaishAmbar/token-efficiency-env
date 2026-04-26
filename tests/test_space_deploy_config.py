"""Tests for the HF Space deploy scaffolding in ``deploy/``.

These tests validate *configuration* only — they never touch the network
or any HF API. The goal is to catch "we broke deploy by editing a file
nobody was paying attention to" regressions.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_DIR = _REPO_ROOT / "token_efficiency_env"
_DEPLOY_DIR = _REPO_ROOT / "deploy"


# Make ``deploy/push_to_hf_space.py`` importable under a stable name
# without turning ``deploy/`` into a full package.
def _import_push_module():
    path = _DEPLOY_DIR / "push_to_hf_space.py"
    spec = importlib.util.spec_from_file_location("push_to_hf_space", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["push_to_hf_space"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Files on disk exist ──────────────────────────────────────────────
def test_space_readme_exists():
    assert (_DEPLOY_DIR / "SPACE_README.md").is_file()


def test_deploy_human_readme_exists():
    assert (_DEPLOY_DIR / "README.md").is_file()


def test_push_script_exists():
    assert (_DEPLOY_DIR / "push_to_hf_space.py").is_file()


def test_env_openenv_yaml_exists_and_has_name():
    yaml = pytest.importorskip("yaml")
    manifest_path = _ENV_DIR / "openenv.yaml"
    assert manifest_path.is_file(), f"missing {manifest_path}"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "openenv.yaml must be a YAML mapping"
    assert data.get("name"), "openenv.yaml must set top-level 'name'"


def test_env_has_dockerfile():
    # openenv push accepts either ./Dockerfile or ./server/Dockerfile.
    root = _ENV_DIR / "Dockerfile"
    server = _ENV_DIR / "server" / "Dockerfile"
    assert root.is_file() or server.is_file(), \
        "need either env-root Dockerfile or server/Dockerfile"


def test_server_dockerfile_exposes_port_8000():
    # Every OpenEnv Space listens on 8000; catch regressions.
    dockerfile = (_ENV_DIR / "server" / "Dockerfile").read_text(encoding="utf-8")
    assert "8000" in dockerfile, "server/Dockerfile should reference port 8000"


# ─── SPACE_README.md frontmatter validation ───────────────────────────
def _parse_frontmatter(text: str) -> dict:
    """Extract and parse YAML frontmatter from a markdown file."""
    yaml = pytest.importorskip("yaml")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SPACE_README.md must start with YAML frontmatter"
    data = yaml.safe_load(match.group(1))
    assert isinstance(data, dict), "frontmatter must be a YAML mapping"
    return data


def test_space_readme_has_frontmatter():
    text = (_DEPLOY_DIR / "SPACE_README.md").read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    for required in ("title", "sdk", "app_port"):
        assert required in fm, f"frontmatter missing required field: {required}"


def test_space_readme_is_docker_sdk_on_port_8000():
    text = (_DEPLOY_DIR / "SPACE_README.md").read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    assert fm["sdk"] == "docker", "Spaces for OpenEnv envs must use the Docker SDK"
    assert int(fm["app_port"]) == 8000, "app_port must match the Dockerfile CMD port"


def test_space_readme_has_openenv_tag():
    text = (_DEPLOY_DIR / "SPACE_README.md").read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    tags = fm.get("tags") or []
    assert "openenv" in tags, "Space should be tagged 'openenv' so it appears in the hub collection"


def test_space_readme_sets_base_path_web():
    """``base_path: /web`` is what makes the OpenEnv web UI render as the
    Space's landing page. Without it, HF shows a bare "Docker running" page."""
    text = (_DEPLOY_DIR / "SPACE_README.md").read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    assert fm.get("base_path") == "/web"


# ─── push_to_hf_space.py behaviour ────────────────────────────────────
def test_push_script_argparse_has_expected_flags():
    push = _import_push_module()
    parser = push.build_parser()
    # Collect option strings flat.
    option_strings = {
        s for act in parser._actions for s in act.option_strings  # noqa: SLF001
    }
    for flag in ("--repo-id", "--private", "--base-image",
                 "--hardware", "--dry-run", "--verbose"):
        assert flag in option_strings, f"push script missing CLI flag {flag}"


def test_should_ignore_catches_caches_and_tests():
    push = _import_push_module()
    patterns = push.DEFAULT_IGNORE_PATTERNS
    for rel in (
        Path(".git/config"),
        Path("__pycache__/x.pyc"),
        Path("some/deep/path/__pycache__/y.pyc"),
        Path("tests/test_something.py"),
        Path("foo/.pytest_cache/v/cache/nodeids"),
    ):
        assert push._should_ignore(rel, patterns), \
            f"expected {rel} to be ignored"


def test_should_ignore_keeps_real_env_files():
    push = _import_push_module()
    patterns = push.DEFAULT_IGNORE_PATTERNS
    for rel in (
        Path("models.py"),
        Path("scorer.py"),
        Path("server/app.py"),
        Path("server/Dockerfile"),
        Path("openenv.yaml"),
        Path("pyproject.toml"),
        Path("README.md"),
    ):
        assert not push._should_ignore(rel, patterns), \
            f"expected {rel} to NOT be ignored"


def test_build_plan_populates_files_and_repo_id():
    push = _import_push_module()
    plan = push._build_plan(
        repo_id="acme/tokeneff",
        private=False,
        base_image=None,
        hardware=None,
    )
    assert plan.repo_id == "acme/tokeneff"
    assert plan.env_dir.name == "token_efficiency_env"
    # Sanity: real env files are present, junk files are not.
    posix_paths = {p.as_posix() for p in plan.files_to_upload}
    assert "models.py" in posix_paths
    assert "openenv.yaml" in posix_paths
    assert "server/app.py" in posix_paths
    assert not any(".pytest_cache" in p for p in posix_paths)
    assert not any("__pycache__" in p for p in posix_paths)


def test_stage_for_upload_swaps_readme(tmp_path):
    push = _import_push_module()
    staging = push._stage_for_upload(
        _ENV_DIR,
        _DEPLOY_DIR / "SPACE_README.md",
        tmp_path,
        push.DEFAULT_IGNORE_PATTERNS,
    )
    staged_readme = staging / "README.md"
    assert staged_readme.exists()
    # The staged README must be the Space-facing one, not the contributor guide.
    staged_text = staged_readme.read_text(encoding="utf-8")
    assert staged_text.startswith("---"), \
        "staged README should start with the HF frontmatter"
    assert "Token Efficiency Env" in staged_text
    assert "In-Package Contributor Guide" not in staged_text


def test_stage_for_upload_mirrors_dockerfile_to_root(tmp_path):
    push = _import_push_module()
    staging = push._stage_for_upload(
        _ENV_DIR,
        _DEPLOY_DIR / "SPACE_README.md",
        tmp_path,
        push.DEFAULT_IGNORE_PATTERNS,
    )
    # HF Spaces auto-detect a Dockerfile at the repo root, not under server/.
    assert (staging / "Dockerfile").exists(), \
        "staged dir must have a root-level Dockerfile for HF Spaces"
    assert (staging / "server" / "Dockerfile").exists(), \
        "original server/Dockerfile must be preserved"


def test_stage_for_upload_enables_web_interface(tmp_path):
    """The staged root Dockerfile MUST set ENABLE_WEB_INTERFACE=true, otherwise
    the Space's /web landing page 404s (our SPACE_README sets base_path: /web).
    The *original* server/Dockerfile must stay untouched so local builds
    remain off-by-default."""
    push = _import_push_module()
    staging = push._stage_for_upload(
        _ENV_DIR,
        _DEPLOY_DIR / "SPACE_README.md",
        tmp_path,
        push.DEFAULT_IGNORE_PATTERNS,
    )
    root_dockerfile = (staging / "Dockerfile").read_text(encoding="utf-8")
    assert "ENV ENABLE_WEB_INTERFACE=true" in root_dockerfile, \
        "staged Dockerfile must enable the /web UI for HF Spaces"

    # The injection must sit BEFORE the top-level CMD line, otherwise Docker
    # evaluates ENV after the process has already started (no-op).
    # NOTE: ``HEALTHCHECK --interval=... CMD curl ...`` also contains the
    # word CMD on an indented continuation line — we only care about the
    # *top-level* CMD instruction (no leading whitespace).
    lines = root_dockerfile.splitlines()
    env_idx = next(
        i for i, ln in enumerate(lines) if "ENABLE_WEB_INTERFACE=true" in ln
    )
    cmd_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("CMD ")
    )
    assert env_idx < cmd_idx, \
        "ENV ENABLE_WEB_INTERFACE must precede the top-level CMD"

    # The source-of-truth server/Dockerfile inside the repo must NOT be
    # mutated — only the staged copy gets the flag.
    repo_dockerfile = (_ENV_DIR / "server" / "Dockerfile").read_text(encoding="utf-8")
    assert "ENV ENABLE_WEB_INTERFACE=true" not in repo_dockerfile, \
        "we must not mutate the repo's server/Dockerfile; only the staged one"


def test_main_dry_run_exits_zero(capsys, monkeypatch):
    """A dry run must succeed even without HF credentials."""
    push = _import_push_module()
    # Clear any HF_TOKEN that might be in the environment so we prove we're
    # not accidentally calling whoami.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc = push.main(["--dry-run", "--repo-id", "demo/tokeneff"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "HF Space deploy plan" in out
    assert "demo/tokeneff" in out
    assert "Dry run — no files uploaded" in out
