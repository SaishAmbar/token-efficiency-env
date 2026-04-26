"""Tests for scripts/demo_session.py — the Tier-2 familiarisation script.

We test two things:

1. Every starter prompt in STARTER_PROMPTS has a CANNED_ANSWERS entry
   that actually matches. If someone edits the prompt bank we catch the
   demo script going stale.
2. The canned answer is a good one — it contains enough of the
   prompt's ``expected_keywords`` that the keyword-judge rates
   correctness=1.0. This is the whole point of the smart trial; if it
   silently stops earning full correctness the demo loses its punch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_demo():
    path = _REPO_ROOT / "scripts" / "demo_session.py"
    spec = importlib.util.spec_from_file_location("demo_session", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["demo_session"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_canned_answer_covers_every_starter_prompt():
    demo = _import_demo()
    from token_efficiency_env.prompts import STARTER_PROMPTS

    missing = [
        entry["prompt"]
        for entry in STARTER_PROMPTS
        if demo._canned_answer(entry["prompt"]) is None
    ]
    assert not missing, (
        "No canned answer for these starter prompts (edit "
        "scripts/demo_session.py CANNED_ANSWERS):\n  - "
        + "\n  - ".join(missing)
    )


def test_canned_answer_satisfies_keyword_judge():
    """Every canned answer should actually score correctness=1.0 via the
    keyword judge — otherwise the "smart trial" demonstration is a lie.

    This mirrors what the real env does on step(): substring-match each
    expected_keyword (or any alias in a list-of-lists entry) against the
    lowercased answer.
    """
    demo = _import_demo()
    from token_efficiency_env.prompts import STARTER_PROMPTS

    failures = []
    for entry in STARTER_PROMPTS:
        answer = demo._canned_answer(entry["prompt"])
        if answer is None:
            continue  # covered by the test above
        lower = answer.lower()

        # Each expected_keywords entry is either a str OR a list of
        # aliases (list-of-lists schema introduced in §7). For the
        # simple-str case the keyword must appear in the answer; for
        # the list-of-aliases case ANY alias satisfies the entry.
        for kw_entry in entry.get("expected_keywords", []):
            if isinstance(kw_entry, list):
                if not any(alias.lower() in lower for alias in kw_entry):
                    failures.append(
                        f"{entry['prompt']!r}: answer {answer!r} "
                        f"matches none of {kw_entry}"
                    )
            else:
                if kw_entry.lower() not in lower:
                    failures.append(
                        f"{entry['prompt']!r}: answer {answer!r} "
                        f"missing keyword {kw_entry!r}"
                    )

    assert not failures, "canned answers fail keyword-judge:\n  - " + "\n  - ".join(failures)


def test_demo_main_rejects_bad_url(monkeypatch, capsys):
    """If the server isn't reachable, main() should fail clean (exit != 0)
    with a readable error, not crash with a stack trace."""
    demo = _import_demo()
    # Use a port we know is not serving us.
    rc = demo.main(["--base-url", "http://127.0.0.1:59999"])
    assert rc != 0, "should return non-zero exit on connection failure"
    out = capsys.readouterr().out
    assert "ERROR" in out
