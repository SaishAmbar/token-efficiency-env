"""Unit tests for ``training.colab_bootstrap``.

We only exercise the *pure* pieces (``is_colab`` + the no-op local path) so
this runs in CI on Windows/macOS/Linux without pulling in ``google.colab``
or shelling out to git. The heavy bits (repo clone, pip install) are only
hit when ``is_colab()`` is True, which is never true in our test env.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from training import colab_bootstrap


def test_is_colab_false_in_tests() -> None:
    # google.colab is never importable in our CI, so this must be False.
    assert colab_bootstrap.is_colab() is False


def test_bootstrap_is_noop_outside_colab(tmp_path: Path) -> None:
    """On a local kernel, ``bootstrap`` must return cwd unchanged and touch nothing."""
    original_cwd = Path.cwd()
    original_env = os.environ.get("HF_TOKEN")
    try:
        result = colab_bootstrap.bootstrap(
            repo_url="https://example.invalid/repo.git",
            repo_dir="irrelevant",
            work_root=str(tmp_path),
            install_unsloth=False,
            prompt_token=False,
        )
        assert result == original_cwd
        assert Path.cwd() == original_cwd
        # No repo should have been cloned to the temp root.
        assert list(tmp_path.iterdir()) == []
    finally:
        if original_env is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = original_env


def test_bootstrap_reexports_from_training_package() -> None:
    """The public API is re-exported from ``training`` for notebook ergonomics."""
    import training

    assert hasattr(training, "bootstrap")
    assert hasattr(training, "is_colab")
    assert training.bootstrap is colab_bootstrap.bootstrap
    assert training.is_colab is colab_bootstrap.is_colab


def test_prompt_token_shortcircuits_when_already_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If HF_TOKEN is already set, ``_prompt_hf_token`` must not call getpass."""
    monkeypatch.setenv("HF_TOKEN", "dummy")
    # If the short-circuit fails, this would blow up (getpass raises).
    monkeypatch.setattr(
        "getpass.getpass",
        lambda prompt="": pytest.fail("getpass called despite HF_TOKEN set"),
    )
    colab_bootstrap._prompt_hf_token()
    assert os.environ["HF_TOKEN"] == "dummy"


def test_prompt_token_accepts_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no token is set and user pastes one, it should land in os.environ."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("JUDGE_BACKEND", raising=False)
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "hf_abc123")
    colab_bootstrap._prompt_hf_token()
    assert os.environ["HF_TOKEN"] == "hf_abc123"
    assert os.environ.get("JUDGE_BACKEND") == "huggingface"


def test_prompt_token_falls_back_to_keyword_when_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank input → no token, judge falls back to keyword backend."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("JUDGE_BACKEND", raising=False)
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "   ")
    colab_bootstrap._prompt_hf_token()
    assert "HF_TOKEN" not in os.environ
    assert os.environ.get("JUDGE_BACKEND") == "keyword"
