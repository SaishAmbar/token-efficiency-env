"""Tiny helper to poll an HF Space until it reaches RUNNING.

Used by the deploy flow as a hands-free "did it build?" check.
Kept under deploy/ so it ships with the rest of the deployment tooling
(NOT shipped to the Space itself; the push script's ignore patterns
strip _* prefixed files).

Usage:
    python deploy/_wait_for_space.py SaishAmbar/token-efficiency-playground
"""
from __future__ import annotations

import sys
import time
import urllib.request
import urllib.error
import json


def _get_json(url: str, timeout: float = 15.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = ""
        return e.code, {"_error_body": body}
    except Exception as e:
        return 0, {"_error": str(e)}


def main(repo_id: str) -> int:
    runtime_url = f"https://huggingface.co/api/spaces/{repo_id}/runtime"
    user, name = repo_id.split("/", 1)
    hosted_url = f"https://{user.lower()}-{name.lower()}.hf.space/api/state"

    print(f"Polling Space build for {repo_id}")
    print(f"  runtime API : {runtime_url}")
    print(f"  hosted app  : {hosted_url}")
    print()

    deadline = time.time() + 9 * 60
    last_stage = None
    while time.time() < deadline:
        status, data = _get_json(runtime_url)
        stage = data.get("stage") or data.get("status") or f"http-{status}"
        hardware = (data.get("hardware") or {}).get("current") or "?"
        if stage != last_stage:
            print(f"  stage={stage} hardware={hardware}", flush=True)
            last_stage = stage

        if stage == "RUNNING":
            h_status, h_data = _get_json(hosted_url)
            print(f"  /api/state -> HTTP {h_status} body={str(h_data)[:160]}", flush=True)
            if h_status == 200:
                print()
                print(f"READY  https://huggingface.co/spaces/{repo_id}")
                return 0
        time.sleep(20)

    print()
    print("Timed out waiting; first build sometimes takes longer.")
    print("Open the Space in your browser and check the Logs tab.")
    return 2


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python deploy/_wait_for_space.py <username/space-name>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
