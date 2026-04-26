"""
LLM judge backends for the *correctness* component of the reward.

Two backends are available, selected at process start by the ``JUDGE_BACKEND``
environment variable:

  - ``huggingface`` (default) — uses ``huggingface_hub.InferenceClient`` against
    the Inference Providers router. Requires ``HF_TOKEN``. Override the model
    with ``JUDGE_MODEL`` (default: ``meta-llama/Llama-3.1-8B-Instruct``).

  - ``keyword`` — pure-Python fallback that scores via word-boundary keyword
    matching against ``expected_keywords`` from ``prompts.py``. No network,
    no auth — handy for tests and CI.

The HF judge auto-falls-back to the keyword scorer for the *single failing
call* whenever the API errors (rate limit, transient 5xx, timeout, etc.).
That keeps GRPO training from getting a misleading 0.0 correctness signal
when the network blips.

Design notes:
  - Both clients are constructed lazily on first use, NOT at import time, so
    importing ``token_efficiency_env`` never blocks on a network call.
  - ``get_judge()`` is a process-level singleton. Each TokenEfficiencyEnv
    session shares one judge — that's intentional; the judge is stateless
    apart from a failure counter.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional, Protocol

logger = logging.getLogger("TokenEfficiencyEnv.judge")

DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


class Judge(Protocol):
    """Protocol: scores answer correctness on the closed interval [0.0, 1.0]."""

    def score_correctness(
        self,
        prompt: str,
        answer: str,
        expected_keywords: Optional[list[str]] = None,
    ) -> float: ...


# ─── Shared keyword scoring (used by KeywordJudge, HF fallback, and scorer.py) ───
def matches_keyword(answer_lower: str, keyword: str) -> bool:
    """Word-boundary keyword check with prefix-stem support.

    Two policies, picked by keyword shape:
      * Pure number (``"100"``, ``"256"``, ``"30"``) → strict full-word match,
        so ``"100"`` does NOT spuriously match ``"1000"``.
      * Anything else → prefix-stem match, so a stem like ``"plant"`` matches
        ``plant`` / ``plants`` / ``planted`` / ``plantation`` (and similarly
        ``antibod`` → ``antibody`` / ``antibodies``, ``purchas`` →
        ``purchasing``, ``intelligen`` → ``intelligence`` / ``intelligent``).
        This matches the way ``prompts.py`` was clearly authored.

    Caller passes the lower-cased answer; this function does not lower-case
    again on every call (the loop in the caller does it once).
    """
    kw = keyword.lower()
    if kw.isdigit():
        pattern = rf"\b{re.escape(kw)}\b"
    else:
        pattern = rf"\b{re.escape(kw)}\w*\b"
    return bool(re.search(pattern, answer_lower))


def _keyword_score(answer: str, keywords) -> float:
    """Fraction of expected keywords found in the answer, in [0.0, 1.0].

    Returns 0.5 (neutral) when no keywords are supplied — no ground truth.

    Supports two keyword schemas (§7 of VULNERABILITY_FIX_PLAN.md):
      * flat ``str``             — one keyword, legacy form.
      * ``list[str]`` (inner)    — alias group; matching ANY form counts
        the group as one satisfied keyword. Used for numeric aliases like
        ``["30", "thirty"]`` so digit-form and word-form answers both score.

    Both shapes can coexist in the same ``expected_keywords`` list.
    """
    if not keywords:
        return 0.5
    a = answer.lower()
    matches = 0
    for entry in keywords:
        forms = [entry] if isinstance(entry, str) else entry
        if any(matches_keyword(a, f) for f in forms):
            matches += 1
    return matches / len(keywords)


# ─── KeywordJudge — offline fallback ────────────────────────────────────
class KeywordJudge:
    """Scores correctness by word-boundary keyword overlap. No API calls."""

    name = "KeywordJudge"

    def score_correctness(
        self,
        prompt: str,
        answer: str,
        expected_keywords: Optional[list[str]] = None,
    ) -> float:
        return _keyword_score(answer, expected_keywords or [])


# ─── HFInferenceJudge — LLM judge via HF Inference Providers ────────────
# Prompt design notes:
#   - We pass expected_keywords as "key facts" to anchor the grader. Without
#     these, smaller judge models (8B-class) hallucinate the correct answer
#     and grade against their hallucination, especially for arithmetic.
#   - The rubric uses concrete bands so the model emits a number, not prose.
#   - "Score:" prefix forces the model to put the number at the end where
#     a regex search on the LAST match is reliable.
_JUDGE_PROMPT_WITH_KEYS = (
    "You are an impartial grader scoring the correctness of an answer.\n\n"
    "Question: {prompt}\n"
    "Key facts a correct answer must contain: {keys}\n"
    "Answer to grade: {answer}\n\n"
    "Scoring rubric:\n"
    "  1.0 = answer is correct and contains the key facts\n"
    "  0.7 = answer is mostly correct but missing or imprecise on some facts\n"
    "  0.4 = answer is partially relevant but largely wrong or vague\n"
    "  0.0 = answer is wrong, off-topic, or empty\n\n"
    "Reply with exactly: Score: X.XX"
)
_JUDGE_PROMPT_NO_KEYS = (
    "You are an impartial grader scoring the correctness of an answer.\n\n"
    "Question: {prompt}\n"
    "Answer to grade: {answer}\n\n"
    "Scoring rubric:\n"
    "  1.0 = answer is correct and complete\n"
    "  0.7 = answer is mostly correct but slightly imprecise or incomplete\n"
    "  0.4 = answer is partially relevant but largely wrong or vague\n"
    "  0.0 = answer is wrong, off-topic, or empty\n\n"
    "Reply with exactly: Score: X.XX"
)


def _parse_score(text: str) -> Optional[float]:
    """Pull a [0,1] score out of the judge's reply.

    Strategy: look for ``Score: <number>`` first (matches our prompt format).
    Fall back to the LAST decimal number with a leading ``0.`` or ``1.`` (the
    grade is almost always at the end of the reply, never the start where the
    model might restate facts from the question).
    """
    if not text:
        return None
    m = re.search(r"[Ss]core\s*[:=]\s*(\d+\.?\d*)", text)
    if m:
        return float(m.group(1))
    candidates = re.findall(r"\b([01](?:\.\d+)?)\b", text)
    if candidates:
        return float(candidates[-1])
    m = re.search(r"\d+\.?\d*", text)
    if m:
        return float(m.group())
    return None


class HFInferenceJudge:
    """LLM judge backed by ``huggingface_hub.InferenceClient``.

    Falls back to keyword scoring on any API failure for the failing call only.
    """

    name = "HFInferenceJudge"

    def __init__(
        self,
        model: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self.model = model or os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise RuntimeError(
                "HFInferenceJudge requires the HF_TOKEN environment variable. "
                "Set JUDGE_BACKEND=keyword to use the offline fallback instead."
            )
        self._client = None  # lazy
        self._failures = 0
        self._calls = 0

    @property
    def failure_rate(self) -> float:
        if self._calls == 0:
            return 0.0
        return self._failures / self._calls

    def _get_client(self):
        if self._client is None:
            # Imported lazily so ``import token_efficiency_env`` never touches
            # the network and never requires huggingface_hub to be installed
            # when the keyword backend is selected.
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(model=self.model, token=self.token)
        return self._client

    def score_correctness(
        self,
        prompt: str,
        answer: str,
        expected_keywords: Optional[list[str]] = None,
    ) -> float:
        self._calls += 1
        try:
            keys = expected_keywords or []
            if keys:
                # §7 schema: each entry may be a flat str or a list of alias
                # forms. Flatten with slash-separated alternatives so the
                # judge sees e.g. "30/thirty" as "one key fact, either form".
                keys_display = [
                    k if isinstance(k, str) else "/".join(k) for k in keys
                ]
                content = _JUDGE_PROMPT_WITH_KEYS.format(
                    prompt=prompt,
                    keys=", ".join(keys_display),
                    answer=answer,
                )
            else:
                content = _JUDGE_PROMPT_NO_KEYS.format(prompt=prompt, answer=answer)
            response = self._get_client().chat_completion(
                messages=[{"role": "user", "content": content}],
                max_tokens=24,
                temperature=0.0,
            )
            text = (response.choices[0].message.content or "").strip()
            value = _parse_score(text)
            if value is None:
                raise ValueError(f"could not parse a number from judge reply: {text!r}")
            return max(0.0, min(1.0, value))
        except Exception as exc:
            self._failures += 1
            logger.warning(
                "HF judge call failed (%d/%d so far): %s. "
                "Falling back to keyword score for this call.",
                self._failures, self._calls, exc,
            )
            return _keyword_score(answer, expected_keywords or [])


# ─── Factory / process-level singleton ──────────────────────────────────
_JUDGE: Optional[Judge] = None


def get_judge() -> Judge:
    """Return the configured judge, constructing it on first call.

    Selection is driven by the ``JUDGE_BACKEND`` env var:
      - ``huggingface`` / ``hf`` (default): ``HFInferenceJudge``. If
        ``HF_TOKEN`` is missing, transparently degrades to ``KeywordJudge``
        so the env still boots (useful for the HF Space first start before
        secrets are set).
      - ``keyword`` / ``offline`` / ``local``: ``KeywordJudge`` always.
    """
    global _JUDGE
    if _JUDGE is not None:
        return _JUDGE

    raw = os.environ.get("JUDGE_BACKEND", "huggingface").lower().strip()
    # Accept common aliases so a typo like JUDGE_BACKEND=hf or =offline does
    # not silently break every /step call hours into an interactive session.
    if raw in {"keyword", "offline", "local"}:
        backend = "keyword"
    elif raw in {"huggingface", "hf", "inference", "remote"}:
        backend = "huggingface"
    else:
        raise ValueError(
            f"Unknown JUDGE_BACKEND={raw!r}; expected one of "
            f"'huggingface' / 'hf' / 'inference' / 'remote' or "
            f"'keyword' / 'offline' / 'local'."
        )

    if backend == "keyword":
        _JUDGE = KeywordJudge()
        logger.info("Judge backend: KeywordJudge (offline)")
    else:  # huggingface
        try:
            _JUDGE = HFInferenceJudge()
            logger.info(
                "Judge backend: HFInferenceJudge | model=%s",
                _JUDGE.model,  # type: ignore[union-attr]
            )
        except RuntimeError as exc:
            logger.warning(
                "HF judge unavailable (%s); falling back to KeywordJudge.", exc
            )
            _JUDGE = KeywordJudge()
    return _JUDGE


def reset_judge() -> None:
    """Clear the cached judge instance. Intended for tests only."""
    global _JUDGE
    _JUDGE = None
