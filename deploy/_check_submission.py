"""One-shot check that the URLs we're about to submit are publicly reachable.

Verifies (without auth) that:
  * Both HF Spaces resolve via the public API.
  * The GitHub repo + the notebook on `main` resolve over HTTPS.

Used as a final pre-submission gate so we don't paste a URL into the form
that turns out to be private, behind a redirect, or 404.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


URLS = [
    ("HF Space — Env", "https://huggingface.co/api/spaces/SaishAmbar/token-efficiency-env"),
    ("HF Space — Playground", "https://huggingface.co/api/spaces/SaishAmbar/token-efficiency-playground"),
    ("GitHub — repo", "https://github.com/SaishAmbar/token-efficiency-env"),
    ("GitHub — notebook", "https://github.com/SaishAmbar/token-efficiency-env/blob/main/notebooks/train_grpo.ipynb"),
    ("Colab — notebook", "https://colab.research.google.com/github/SaishAmbar/token-efficiency-env/blob/main/notebooks/train_grpo.ipynb"),
]


def _probe(url: str) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "submission-check"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
            return resp.status, body[:200]
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return 0, f"error: {e}"


def main() -> int:
    fail = 0
    for label, url in URLS:
        status, body = _probe(url)
        ok = "OK" if 200 <= status < 400 else "FAIL"
        if not ok == "OK":
            fail += 1
        # For HF Space API responses, also pluck the runtime stage
        extra = ""
        if "huggingface.co/api/spaces" in url and ok == "OK":
            try:
                data = json.loads(body) if body.startswith("{") else {}
                stage = (data.get("runtime") or {}).get("stage", "?")
                private = data.get("private", False)
                extra = f" stage={stage} private={private}"
            except Exception:
                pass
        print(f"  [{ok:4s}] {label:24s} status={status}{extra}  {url}")
    print()
    if fail:
        print(f"{fail} URL(s) failed pre-submission check.")
        return 2
    print("All URLs reachable and publicly accessible. Safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
