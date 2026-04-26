# Deploying TokenEfficiencyEnv to a Hugging Face Space

This folder holds the Space-deployment plumbing, kept **separate** from the
environment source so the env package stays small and pip-installable
without pulling in deploy-only deps.

| File                     | What it is |
|--------------------------|------------|
| `SPACE_README.md`        | The judge-facing README that lands on the HF Space. Has the HF YAML frontmatter (`sdk: docker`, `app_port: 8000`, etc.). Overrides the contributor-facing `token_efficiency_env/README.md` during deploy. |
| `push_to_hf_space.py`    | Python helper that stages + uploads the env directory to HF Spaces. Swaps the README for the Space-facing one. Works without the `openenv` CLI installed. |
| `README.md` (this file)  | Operator's guide for using the deploy script. |

---

## One-time setup

1. **Install deploy dependencies** (most people already have these):
   ```bash
   pip install huggingface_hub pyyaml
   ```

2. **Authenticate with Hugging Face.** Either:
   ```bash
   huggingface-cli login
   ```
   or set the `HF_TOKEN` environment variable:
   ```bash
   export HF_TOKEN="hf_..."                 # Linux / macOS
   $env:HF_TOKEN = "hf_..."                  # Windows PowerShell
   ```
   The token needs **write** access to Spaces (go to
   https://huggingface.co/settings/tokens and check the "Write" scope).

---

## Dry run — validate before spending any HF quota

```bash
python deploy/push_to_hf_space.py --dry-run
```

What this prints:
- The target repo id it would deploy to (`<whoami>/token-efficiency-env` by default).
- Every file that would be uploaded (ignoring `__pycache__`, `tests/`, hidden files, etc.).
- The Space README, base image, and hardware tier.

Safe to run any time; zero network calls to HF's mutating APIs.

---

## Actually push

```bash
# Default: create the Space as <your-username>/token-efficiency-env on free cpu-basic hardware.
python deploy/push_to_hf_space.py
```

**Common flags:**

```bash
# Push to a specific repo (org Space, or a different name).
python deploy/push_to_hf_space.py --repo-id my-org/tokeneff-env

# Start private; flip to public via the Space settings UI later.
python deploy/push_to_hf_space.py --private

# Request T4 hardware (paid; required only if you deploy a hosted Llama-3 judge).
python deploy/push_to_hf_space.py --hardware t4-medium

# Pin a specific openenv-base image (the Dockerfile's FROM).
python deploy/push_to_hf_space.py --base-image ghcr.io/meta-pytorch/openenv-base:v0.2.2
```

The script prints the live Space URL at the end. First build takes
~4–6 min while HF pulls the `openenv-base` image and compiles deps.

---

## Post-deploy: wire up the judge

The Space boots with **`KeywordJudge`** by default (no external calls) so
it's always operational. To enable the real **LLM judge** (Llama-3.1-8B via
HF Inference):

1. Go to your Space → **Settings → Variables and secrets**.
2. Add a **secret** named `HF_TOKEN` with your token value.
3. Optionally add a **variable** `JUDGE_MODEL` (default: `meta-llama/Llama-3.1-8B-Instruct`).
4. The Space will automatically restart and pick up the new config.

Verify the judge is active:
```bash
curl https://<you>-token-efficiency-env.hf.space/health
# {"status": "healthy", "judge": "HFInferenceJudge", ...}
```

---

## Post-deploy: check the web UI

Every OpenEnv Space ships with a `/web` interactive UI. Visit:
```
https://<you>-token-efficiency-env.hf.space/web
```

You can paste any `<budget>N</budget><answer>...</answer>` string, hit
**Step**, and see the full reward breakdown live. This is the easiest
way to show a judge what your env does without them needing to clone
anything.

---

## Troubleshooting

- **`Not authenticated with Hugging Face`** → run `huggingface-cli login`
  or `export HF_TOKEN=...`.
- **`Required file missing: ...`** → probably running from the wrong
  directory. The script expects to be run from the repo root.
- **Space builds but `/ws` hangs** → check the Space logs for
  `transformers` download failures. The first boot downloads the Qwen
  tokenizer (~20 MB); give it ~2 min.
- **Build fails with "no space left on device"** → HF Space free tier
  has a 50 GB disk limit. Our Dockerfile builds to ~2 GB so this should
  never trigger, but if it does, delete stale revisions from the Space's
  git history.

---

## Related docs

- **Root `README.md`** — training pipeline, architecture, quick start.
- **`ARCHITECTURE.md`** — reward function, cliffs, curriculum design.
- **`CHANGELOG.md`** — v0.3.0 reward-honesty release, Phase 8 scaffolding.
- **OpenEnv deployment guide** — `OpenEnv/tutorial/02-deployment.md`
  (vendored in this repo).
