"""One-shot smoke test for the programmatic prompt bank loader.

Runs the loader against a small slice of real HuggingFace data and emits
``training/prompt_bank_report.json`` with:

* per-source counts,
* per-tier (easy / medium / hard) distribution,
* a sample of 5 entries per source,
* the git SHA + timestamp of the run (for audit).

Why this exists
---------------
The loader is covered by unit tests that use a stub dataset, but those
tests don't prove the real HF paths (column names, tokenizer download,
answer parsing) work. This script exists so that before training we can
run one command and get a JSON receipt that the bank is healthy.

Usage (local, cheap — GSM8K only, 50 rows)::

    python scripts/smoke_prompt_bank.py --sources gsm8k --max-prompts 50

Usage (Colab, full 2.3k bank)::

    python scripts/smoke_prompt_bank.py

Exit code 0 on success, 1 on any validation failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# Make ``token_efficiency_env`` + ``training`` importable when this script is
# run from a checkout that hasn't been ``pip install``-ed (the common case for
# hackathon reproducers who just ``git clone``).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        choices=["gsm8k", "trivia_qa", "arc", "openorca"],
        help="Subset of sources to load (default: all four).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Cap total prompts (after dedup).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("training/prompt_bank_report.json"),
        help="Where to write the audit report.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Import lazily so `--help` works even without `datasets` installed.
    from token_efficiency_env.prompt_bank_loader import load_programmatic_bank

    t0 = time.time()
    print(f"[smoke] loading bank (sources={args.sources}, cap={args.max_prompts})...")
    bank = load_programmatic_bank(
        seed=args.seed,
        max_prompts=args.max_prompts,
        sources=args.sources,
    )
    elapsed = time.time() - t0
    print(f"[smoke] loaded {len(bank)} entries in {elapsed:.1f}s")

    # ── Schema validation: every entry must match the shared contract ──
    errors: list[str] = []
    for i, entry in enumerate(bank[:500]):  # cap scan at 500 for speed
        if not isinstance(entry.get("prompt"), str) or not entry["prompt"]:
            errors.append(f"[{i}] missing/empty prompt")
        if entry.get("complexity") not in ("easy", "medium", "hard"):
            errors.append(f"[{i}] bad complexity={entry.get('complexity')!r}")
        kws = entry.get("expected_keywords")
        if not isinstance(kws, list):
            errors.append(f"[{i}] expected_keywords not a list")
            continue
        for j, kw in enumerate(kws):
            if not isinstance(kw, (str, list)):
                errors.append(f"[{i}].keywords[{j}] bad type={type(kw).__name__}")

    if errors:
        print("[smoke] SCHEMA ERRORS:")
        for e in errors[:10]:
            print(" -", e)
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
        return 1

    tier_counts = Counter(e["complexity"] for e in bank)
    print(
        f"[smoke] tier distribution: "
        f"easy={tier_counts['easy']}, "
        f"medium={tier_counts['medium']}, "
        f"hard={tier_counts['hard']}"
    )

    # Sample 3 entries for a human-readable preview.
    samples = bank[:3]
    for i, s in enumerate(samples):
        print(f"  sample[{i}] [{s['complexity']}] {s['prompt'][:80]}...")
        print(f"             keywords={s['expected_keywords']}")

    report = {
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "seed": args.seed,
        "sources": args.sources or ["gsm8k", "trivia_qa", "arc", "openorca"],
        "total_prompts": len(bank),
        "elapsed_seconds": round(elapsed, 2),
        "tier_counts": dict(tier_counts),
        "samples": samples,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[smoke] report written to {args.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
