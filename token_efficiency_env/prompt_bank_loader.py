"""Programmatic prompt-bank loader (Phase 8 / Stage 6.A).

Separated from ``prompts.py`` on purpose: the latter must stay
import-clean for CI / tests / the env server. This module is only pulled
in when a caller opts into the full bank via ``PROMPT_BANK_MODE=full`` (or
calls ``load_programmatic_bank`` directly) and therefore freely imports
``datasets``, ``transformers``, and ``num2words``.

Source mix (per plan §6.A):

    | Source                   | Subset        | Tier target         | ~Count |
    |--------------------------|---------------|---------------------|--------|
    | GSM8K (train)            | first 400     | easy / medium       |   400  |
    | TriviaQA (rc.nocontext)  | random 400    | easy                |   400  |
    | ARC-Challenge (train)    | all           | medium / hard       |  ~1.1k |
    | OpenOrca (1M slice)      | length ≤ 200  | medium / hard       |   500  |
    |                                                            Total ~2.3k |

Every entry comes out in the shared schema used by ``STARTER_PROMPTS`` so
the rest of the pipeline (scorer, adapter, environment) stays ignorant of
the source::

    {
        "prompt": str,
        "complexity": "easy" | "medium" | "hard",
        "expected_keywords": list[str | list[str]],
    }

Design notes
------------

* **Tier binning** uses the Qwen2.5-3B tokenizer on the reference answer
  (same tokenizer the env uses for ``tokens_used``), so complexity stays
  calibrated to ``COMPLEXITY_IDEAL_TOKENS``.
* **Keyword extraction** is the plan's "3 lowest-IDF non-stopword content
  tokens" heuristic rather than RAKE — fewer dependencies, roughly the
  same signal, and the keyword component is only 5% of the reward.
* **Numeric aliasing** uses ``num2words`` so a reference answer like
  ``"30"`` also accepts ``"thirty"`` (§7 schema).
* **Determinism** every sample / shuffle is seeded.
* **Dedup** by normalised prompt text at the very end.

This module is NOT exercised on the default CI path. ``tests/test_prompt_
bank_loader.py`` runs a small offline slice using an injected stub, and a
``@pytest.mark.slow`` integration test covers the real HF path.
"""

from __future__ import annotations

import logging
import random
import re
import string
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# A light stopword list — deliberately short, optimised for precision
# against question / answer pairs (not recall-first NLP).
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at
    be because been before being below between both but by
    could did do does doing down during each few for from further
    had has have having he her here hers him his how
    i if in into is it its itself just
    me more most my myself
    no nor not now of off on once only or other our ours out over own
    same she should so some such
    than that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would
    you your yours yourself yourselves
    """.split()
)


# ─── Tokenizer singleton (the Qwen one the env already uses) ────────────
_TOKENIZER = None


def _get_qwen_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    return _TOKENIZER


# ─── Small helpers ───────────────────────────────────────────────────────
def _bucket_by_expected_length(reference_answer: str) -> str:
    """Map a reference answer to easy/medium/hard based on Qwen token count.

    Bands match ``COMPLEXITY_IDEAL_TOKENS`` (v0.3.0: 27 / 72 / 142) so
    the existing efficiency reward stays sensible on the full bank.
    """
    try:
        n = len(_get_qwen_tokenizer().encode(reference_answer))
    except Exception:  # pragma: no cover — tokenizer load failure path
        # Rough fallback: 1 word ≈ 1.3 tokens for English.
        n = int(len(reference_answer.split()) * 1.3)
    if n <= 15:
        return "easy"
    if n <= 80:
        return "medium"
    return "hard"


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _content_tokens(text: str) -> List[str]:
    """Lowercased word tokens with stopwords + pure-punctuation dropped."""
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return [w for w in words if w not in _STOPWORDS and not w.isdigit() or w.isdigit()]


def _extract_keywords(reference_answer: str, k: int = 3) -> List[str]:
    """Return up to ``k`` keyword stems from a reference answer.

    Heuristic: the 3 longest non-stopword tokens (proxy for low-IDF in
    practice — long words are rarer in natural text). Not a replacement
    for hand-curation, but good enough for a 5%-weighted reward signal.
    """
    seen: set[str] = set()
    candidates: List[str] = []
    order: Dict[str, int] = {}
    for i, tok in enumerate(_content_tokens(reference_answer)):
        t = tok.strip(string.punctuation)
        if not t or t in seen:
            continue
        seen.add(t)
        candidates.append(t)
        order[t] = i
    # Prefer longer tokens (stand-ins for lower IDF) and break ties by
    # order of appearance. We record the order on the fly (O(n)) instead of
    # calling ``.index()`` inside the key (O(n²) + crashes if the stripped
    # token isn't byte-identical to its unstripped source — e.g. tokens
    # with surrounding apostrophes like "'hello'" strip to "hello").
    candidates.sort(key=lambda w: (-len(w), order[w]))
    return candidates[:k]


def _add_numeric_aliases(keywords: Sequence[str]) -> List:
    """For any keyword that's purely numeric, emit ``[digit, word_form]``.

    Uses ``num2words`` if available; silently skips the alias if not
    (the digit form alone is still a valid flat keyword).
    """
    try:
        from num2words import num2words  # type: ignore[import-not-found]
    except ImportError:
        num2words = None

    out: List = []
    for kw in keywords:
        if kw.isdigit() and num2words is not None:
            try:
                word_form = num2words(int(kw))
            except Exception:  # pragma: no cover — num2words edge case
                word_form = None
            if word_form:
                out.append([kw, word_form])
                continue
        out.append(kw)
    return out


# ─── Source loaders — each returns an iterable of PromptEntries ─────────
def _load_gsm8k(rng: random.Random, cache_dir: Optional[str], n: int = 400) -> List[dict]:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)
    # GSM8K answers end with "#### <number>". Keep the number as the
    # reference answer; that's what scorer cares about for keywording.
    out: List[dict] = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        question: str = row["question"]
        reference: str = row["answer"].split("####")[-1].strip()
        if not reference:
            continue
        out.append(_build_entry(question, reference))
    return out


def _load_trivia_qa(rng: random.Random, cache_dir: Optional[str], n: int = 400) -> List[dict]:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", cache_dir=cache_dir)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    out: List[dict] = []
    for idx in idxs:
        if len(out) >= n:
            break
        row = ds[idx]
        question: str = row["question"]
        # ``answer`` is a dict with ``value`` (gold) and ``aliases``.
        reference: str = row["answer"]["value"]
        if not reference:
            continue
        out.append(_build_entry(question, reference))
    return out


def _load_arc_challenge(rng: random.Random, cache_dir: Optional[str]) -> List[dict]:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("ai2_arc", "ARC-Challenge", split="train", cache_dir=cache_dir)
    out: List[dict] = []
    for row in ds:
        question: str = row["question"]
        # The gold label points at one of ``choices.text``.
        try:
            gold_idx = row["choices"]["label"].index(row["answerKey"])
            reference: str = row["choices"]["text"][gold_idx]
        except (KeyError, ValueError):
            continue
        if not reference:
            continue
        out.append(_build_entry(question, reference))
    return out


def _load_open_orca(rng: random.Random, cache_dir: Optional[str], n: int = 500) -> List[dict]:
    from datasets import load_dataset  # type: ignore[import-not-found]

    # Stream so we don't pull the whole 1M-row set just to filter.
    ds = load_dataset(
        "Open-Orca/OpenOrca",
        split="train",
        cache_dir=cache_dir,
        streaming=True,
    )
    out: List[dict] = []
    tokenizer = _get_qwen_tokenizer()
    for row in ds:
        if len(out) >= n:
            break
        question: str = row.get("question") or ""
        reference: str = row.get("response") or ""
        if not question or not reference:
            continue
        if len(tokenizer.encode(reference)) > 200:
            continue
        out.append(_build_entry(question, reference))
    return out


def _build_entry(question: str, reference_answer: str) -> dict:
    """Assemble a single PromptEntry from a (question, reference) pair."""
    kws = _extract_keywords(reference_answer, k=3)
    return {
        "prompt": question.strip(),
        "complexity": _bucket_by_expected_length(reference_answer),
        "expected_keywords": _add_numeric_aliases(kws),
    }


# ─── Public entrypoint ───────────────────────────────────────────────────
def load_programmatic_bank(
    seed: int = 42,
    max_prompts: Optional[int] = None,
    cache_dir: Optional[str] = None,
    *,
    sources: Optional[Sequence[str]] = None,
) -> List[dict]:
    """Fetch the full ~2.3k prompt bank.

    ``sources`` is an iterable of source names — by default all four are
    used. Overriding it is mostly a test hook.

    Raises ``ImportError`` if ``datasets`` or the tokenizer aren't
    available. Prefer the ``PROMPT_BANK_MODE=starter`` fallback in that
    case; ``prompts.get_prompt_bank()`` does that automatically.
    """
    try:
        import datasets  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The full prompt bank requires the `datasets` package. "
            "Run `pip install -r requirements-train.txt` or set "
            "PROMPT_BANK_MODE=starter."
        ) from exc

    rng = random.Random(seed)
    sources = tuple(sources) if sources else ("gsm8k", "trivia_qa", "arc", "openorca")
    logger.info("Loading programmatic prompt bank from: %s", sources)

    collected: List[dict] = []
    if "gsm8k" in sources:
        collected.extend(_load_gsm8k(rng, cache_dir))
    if "trivia_qa" in sources:
        collected.extend(_load_trivia_qa(rng, cache_dir))
    if "arc" in sources:
        collected.extend(_load_arc_challenge(rng, cache_dir))
    if "openorca" in sources:
        collected.extend(_load_open_orca(rng, cache_dir))

    # Dedup on normalised prompt text — different datasets sometimes reuse
    # the same canonical trivia question.
    seen: set[str] = set()
    deduped: List[dict] = []
    for entry in collected:
        key = _normalise(entry["prompt"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)

    rng.shuffle(deduped)
    if max_prompts is not None:
        deduped = deduped[: max_prompts]

    logger.info(
        "Programmatic bank ready: %d prompts (tier counts: %s)",
        len(deduped),
        _tier_counts(deduped),
    )
    return deduped


def _tier_counts(entries: Iterable[dict]) -> Tuple[int, int, int]:
    """``(easy, medium, hard)`` counts for quick logging."""
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for e in entries:
        tier = e.get("complexity")
        if tier in counts:
            counts[tier] += 1
    return counts["easy"], counts["medium"], counts["hard"]
