"""Pytest fixtures and path setup for the TokenEfficiencyEnv test suite.

The package lives in ``token_efficiency_env/`` (one level up). We prepend the
repo root to ``sys.path`` so ``from token_efficiency_env...`` imports work
without requiring an editable install. This mirrors how the smoke-test
scripts and the OpenEnv server discover the package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("JUDGE_BACKEND", "keyword")
