"""Stratified train / holdout / probe split of the prompt bank.

Phase 8 (§6 of ``test_check_docs/VULNERABILITY_FIX_PLAN.md``):

The v0.2 scaffold used hard-coded indices (last 2 of each tier) on a
24-prompt hand-curated bank. That made the split trivially reproducible
but can't scale to the ~2.3k prompts that land with the programmatic
loader in Stage 6.A.

The new splitter:

* Works on **any** prompt bank that exposes ``[{"prompt": ..., "complexity": ...}, ...]``.
* Is **stratified** per complexity tier so holdout and probe always
  represent the same mix as train.
* Is **deterministic** under ``SPLIT_SEED`` — review-able shuffles, not
  wall-clock randomness.
* Persists the resulting indices to ``training/prompts_split.json`` when
  ``dump_split()`` is called, so reviewers can audit exactly which
  prompts the model never saw.

Fractions default to ``0.85 / 0.10 / 0.05`` (train / holdout / probe), but
for tiny tiers (<20 prompts) the allocator guarantees **at least one
prompt in each of holdout and probe** so the eval/probe sets are never
empty. This keeps the 24-prompt starter bank usable end-to-end while also
being correct for the larger programmatic bank.

Back-compat shims ``TRAIN_INDICES`` / ``HOLDOUT_INDICES`` /
``train_prompts()`` / ``holdout_prompts()`` are computed from the new
splitter rather than hard-coded, so nothing downstream has to change
immediately — but the exact numbers shift from 18/6 to ~18/3/3 on the
24-prompt bank.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from token_efficiency_env.prompts import PROMPT_BANK, get_prompt_bank

# ─── Default fractions ───────────────────────────────────────────────────
SPLIT_SEED: int = 42
TRAIN_FRACTION: float = 0.85
HOLDOUT_FRACTION: float = 0.10
PROBE_FRACTION: float = 0.05

# When a tier has fewer than this many prompts, allocator guarantees a
# minimum of 1 prompt in both ``holdout`` and ``probe`` (subtracted from
# the train share). Above this threshold the raw fractions decide.
SMALL_TIER_THRESHOLD: int = 20


# ─── Core allocator ──────────────────────────────────────────────────────
def _allocate_counts(
    n_tier: int,
    train_frac: float = TRAIN_FRACTION,
    holdout_frac: float = HOLDOUT_FRACTION,
    probe_frac: float = PROBE_FRACTION,
) -> tuple[int, int, int]:
    """Return ``(n_train, n_holdout, n_probe)`` for a tier of size ``n_tier``.

    Guarantees:
      * ``n_train + n_holdout + n_probe == n_tier``
      * ``n_holdout >= 1`` and ``n_probe >= 1`` whenever ``n_tier >= 3``
        (so tiny tiers still end up with at least one eval / one probe
        prompt). For ``n_tier == 2`` we allocate 1 holdout, 1 probe, 0
        train — a reviewer should rather see the error than train on a
        degenerate split.
      * For ``n_tier >= SMALL_TIER_THRESHOLD`` (20), counts follow the
        raw fractions with any rounding remainder going to ``train``.
    """
    if n_tier <= 0:
        return 0, 0, 0
    if n_tier == 1:
        # Single-prompt tier: put it in train, no eval signal is worse
        # than no training signal.
        return 1, 0, 0
    if n_tier == 2:
        return 0, 1, 1

    if n_tier < SMALL_TIER_THRESHOLD:
        # Small-tier safety: at least 1 in holdout AND probe.
        n_holdout = max(1, round(n_tier * holdout_frac))
        n_probe = max(1, round(n_tier * probe_frac))
        # If the two eval buckets together exceeded the tier, shrink them.
        while n_holdout + n_probe >= n_tier:
            if n_probe > 1:
                n_probe -= 1
            elif n_holdout > 1:
                n_holdout -= 1
            else:
                break
        n_train = n_tier - n_holdout - n_probe
        return n_train, n_holdout, n_probe

    n_holdout = round(n_tier * holdout_frac)
    n_probe = round(n_tier * probe_frac)
    n_train = n_tier - n_holdout - n_probe
    return n_train, n_holdout, n_probe


def stratified_split(
    prompts: Sequence[Mapping],
    seed: int = SPLIT_SEED,
    train_frac: float = TRAIN_FRACTION,
    holdout_frac: float = HOLDOUT_FRACTION,
    probe_frac: float = PROBE_FRACTION,
) -> Dict[str, List[int]]:
    """Stratify ``prompts`` by ``complexity`` and split into train/holdout/probe.

    Returns a dict ``{"train": [...], "holdout": [...], "probe": [...]}``
    with absolute indices into ``prompts``. The three lists are disjoint
    and together cover every index exactly once.

    Stratification is per ``complexity`` field: within each tier we
    shuffle deterministically (``random.Random(seed).shuffle(...)``) and
    slice according to ``_allocate_counts``. Mixing tiers back together
    happens only at return time — reviewers reading the JSON dump can see
    the per-tier makeup by cross-referencing indices.
    """
    if not 0.999 <= (train_frac + holdout_frac + probe_frac) <= 1.001:
        raise ValueError(
            f"fractions must sum to ~1.0, got "
            f"{train_frac}+{holdout_frac}+{probe_frac}="
            f"{train_frac + holdout_frac + probe_frac}"
        )

    by_tier: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        tier = p.get("complexity", "unknown") or "unknown"
        by_tier[tier].append(i)

    rng = random.Random(seed)
    train: List[int] = []
    holdout: List[int] = []
    probe: List[int] = []

    # Sort tier keys so iteration order is seed-stable regardless of
    # dict-insertion order differences across Python versions.
    for tier in sorted(by_tier):
        indices = list(by_tier[tier])
        rng.shuffle(indices)
        n_train, n_holdout, n_probe = _allocate_counts(
            len(indices), train_frac, holdout_frac, probe_frac
        )
        train.extend(indices[:n_train])
        holdout.extend(indices[n_train:n_train + n_holdout])
        probe.extend(indices[n_train + n_holdout:n_train + n_holdout + n_probe])

    # Stable order in the returned lists — nice for JSON diffs.
    return {
        "train": sorted(train),
        "holdout": sorted(holdout),
        "probe": sorted(probe),
    }


# ─── Eagerly compute the default (starter-bank) split ───────────────────
# Kept as module-level constants for back-compat with existing tests,
# dashboards, and the v0.3.x public API. Mode-aware variants below
# (re)compute the split on demand for the programmatic bank.
_DEFAULT_SPLIT = stratified_split(PROMPT_BANK)
TRAIN_INDICES: List[int] = _DEFAULT_SPLIT["train"]
HOLDOUT_INDICES: List[int] = _DEFAULT_SPLIT["holdout"]
PROBE_INDICES: List[int] = _DEFAULT_SPLIT["probe"]


# Small cache so repeated notebook cell reruns don't re-download the 2.3k
# bank. Keyed on ``mode`` because the starter bank never changes shape.
_MODE_CACHE: Dict[str, Tuple[Sequence[Mapping], Dict[str, List[int]]]] = {}


def _resolve_for_mode(
    mode: Optional[str],
) -> Tuple[Sequence[Mapping], Dict[str, List[int]]]:
    """Return ``(bank, split_dict)`` for a given ``prompt_bank_mode``.

    * ``mode=None`` → use the pre-computed starter-bank split (zero cost,
      preserves legacy ``TRAIN_INDICES`` semantics).
    * ``mode="starter"`` → explicit starter bank.
    * ``mode="full"`` → lazy-load the programmatic bank via
      ``get_prompt_bank("full")`` and compute a fresh stratified split.
    """
    if mode is None:
        return PROMPT_BANK, _DEFAULT_SPLIT

    key = mode.lower().strip()
    if key in _MODE_CACHE:
        return _MODE_CACHE[key]

    bank = list(get_prompt_bank(key))
    split = stratified_split(bank)
    _MODE_CACHE[(mode or "").lower().strip()] = (bank, split)
    return bank, split


def reset_split_cache() -> None:
    """Clear the mode-indexed bank/split cache. Intended for tests."""
    _MODE_CACHE.clear()


def train_prompts(mode: Optional[str] = None) -> List[Dict]:
    """Training subset (deep-copied). ``mode`` selects starter vs full bank."""
    bank, split = _resolve_for_mode(mode)
    return [dict(bank[i]) for i in split["train"]]


def holdout_prompts(mode: Optional[str] = None) -> List[Dict]:
    """Held-out subset (deep-copied)."""
    bank, split = _resolve_for_mode(mode)
    return [dict(bank[i]) for i in split["holdout"]]


def probe_prompts(mode: Optional[str] = None) -> List[Dict]:
    """Periodic on-policy probe subset (deep-copied).

    Used by the notebook to run a quick eval every ``cfg.eval_steps``
    during training. Distinct from ``holdout_prompts()``, which is
    reserved for the single end-of-training comparison.
    """
    bank, split = _resolve_for_mode(mode)
    return [dict(bank[i]) for i in split["probe"]]


# ─── Reporting / persistence ─────────────────────────────────────────────
def describe_split(mode: Optional[str] = None) -> str:
    """Human-readable split summary, for the notebook to print up front.

    On the starter bank the summary lists every holdout/probe prompt.
    On the full bank (~2.3k) we'd drown the notebook in output, so we
    print only counts + a 5-prompt sample per bucket.
    """
    bank, split = _resolve_for_mode(mode)
    train_idx, holdout_idx, probe_idx = split["train"], split["holdout"], split["probe"]
    mode_label = mode or "starter"

    lines = [
        f"Stratified split (seed={SPLIT_SEED}, mode={mode_label}): "
        f"train={len(train_idx)}  holdout={len(holdout_idx)}  "
        f"probe={len(probe_idx)}",
    ]

    verbose = len(bank) <= 50  # full listing for starter, sample for full
    if verbose:
        lines.append("")
        lines.append("Held-out prompts (NOT seen during training):")
        for i in holdout_idx:
            p = bank[i]
            lines.append(f"  [{i:4d}] {p['complexity']:<6} | {p['prompt']}")
        if probe_idx:
            lines.append("")
            lines.append("Probe prompts (periodic on-policy eval during training):")
            for i in probe_idx:
                p = bank[i]
                lines.append(f"  [{i:4d}] {p['complexity']:<6} | {p['prompt']}")
    else:
        lines.append("")
        lines.append("Holdout sample (5 of {} total):".format(len(holdout_idx)))
        for i in holdout_idx[:5]:
            p = bank[i]
            preview = p["prompt"][:90].replace("\n", " ")
            lines.append(f"  [{i:4d}] {p['complexity']:<6} | {preview}")
        if probe_idx:
            lines.append("")
            lines.append("Probe sample (5 of {} total):".format(len(probe_idx)))
            for i in probe_idx[:5]:
                p = bank[i]
                preview = p["prompt"][:90].replace("\n", " ")
                lines.append(f"  [{i:4d}] {p['complexity']:<6} | {preview}")

    return "\n".join(lines)


def dump_split(
    path: str | Path = "training/prompts_split.json",
    *,
    indent: int = 2,
) -> Path:
    """Persist the current split to disk for audit / CI diffing.

    Written as stable-sorted JSON so ``git diff`` on the file stays
    readable when the bank grows or the seed is bumped.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SPLIT_SEED,
        "fractions": {
            "train": TRAIN_FRACTION,
            "holdout": HOLDOUT_FRACTION,
            "probe": PROBE_FRACTION,
        },
        "total_prompts": len(PROMPT_BANK),
        "train": list(TRAIN_INDICES),
        "holdout": list(HOLDOUT_INDICES),
        "probe": list(PROBE_INDICES),
    }
    out.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    return out


# ─── Sanity checks (run at import time) ──────────────────────────────────
def _assert_split_invariants() -> None:
    n = len(PROMPT_BANK)
    if n < 3:
        raise RuntimeError(
            f"PROMPT_BANK has {n} entries; split requires at least 3 "
            "(one per complexity tier)."
        )

    all_indices = set(TRAIN_INDICES) | set(HOLDOUT_INDICES) | set(PROBE_INDICES)
    if len(all_indices) != n:
        raise RuntimeError(
            f"split does not cover PROMPT_BANK: "
            f"covered={len(all_indices)} != n={n}"
        )
    if (
        set(TRAIN_INDICES) & set(HOLDOUT_INDICES)
        or set(TRAIN_INDICES) & set(PROBE_INDICES)
        or set(HOLDOUT_INDICES) & set(PROBE_INDICES)
    ):
        raise RuntimeError("train/holdout/probe overlap; split is broken.")

    # Each eval bucket must cover every tier present in the bank, so the
    # holdout and probe metrics are stratified rather than accidentally
    # skewed toward one complexity class.
    tiers_in_bank = {p.get("complexity") for p in PROMPT_BANK}
    tiers_in_bank.discard(None)
    for name, idxs in (("holdout", HOLDOUT_INDICES), ("probe", PROBE_INDICES)):
        if not idxs:
            continue
        covered = {PROMPT_BANK[i].get("complexity") for i in idxs}
        missing = tiers_in_bank - covered
        if missing:
            raise RuntimeError(
                f"stratified split invariant violated: {name} is missing "
                f"tier(s) {sorted(missing)!r}. Shuffle seed or tier sizes "
                "produced an empty bucket somewhere."
            )


_assert_split_invariants()
