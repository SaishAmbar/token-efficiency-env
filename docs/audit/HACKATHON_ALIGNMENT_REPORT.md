# Hackathon Alignment Report — TokenEfficiencyEnv

> **Scope:** point-by-point audit of the current `token-efficiency-env-latest/`
> tree against the two hackathon guidance documents in this folder
> (`pdf_extracted.txt` = 60-question FAQ, `docx_extracted.txt` = 22-section
> self-serve guide). **Read-only analysis** — no code was changed to
> produce this report.
>
> **Source commit audited:** `3efe2d7` on `main` (Sun Apr 26 2026, 10:12 IST).
> **Deployed artifact:** HuggingFace Space (live, web UI + API verified).
> **Test status:** 147 tests pass, 7 skipped (WebSocket E2E only, expected).
>
> **Headline score: 8.5 / 10** — strongly aligned with the guidance, one
> critical unpatched vulnerability (hidden-CoT loophole, see §5.1), a few
> moderate gaps around mid-training monitoring and QLoRA save path, and
> the 300-step full training hasn't completed yet (50-step smoke did).

---

## 0. One-paragraph executive summary

TokenEfficiencyEnv is a **textbook-correct RLVR environment** on top of the
recommended OpenEnv + TRL + Unsloth stack. The task is verifiable, the
reward is layered (6 components + 5 hard cliffs), the curriculum is
implemented, and the Space is deployed. The weakest links are **(a)** the
hidden-chain-of-thought loophole documented in `VULNERABILITY_FIX_PLAN.md`
§1 — still unpatched; a model could pad with 500 think-tokens outside the
tags and score 0.97 (this is exactly the *specification gaming* failure the
hackathon guidance warns about), **(b)** no mid-training sample audits
(guidance repeatedly says "sample outputs frequently, don't rely on the
reward scalar"), and **(c)** the 300-step full run hasn't landed yet so
the `before_after_report.md` judges will see is from the 50-step smoke.
Fixing (a) before the submission deadline is the single highest-leverage
change; (b) and (c) are nice-to-haves.

---

## 1. Scoring rubric (20 dimensions, each scored 0–10)

The rubric is assembled directly from the two guidance files. Cross-refs
use `docx.§N` for the self-serve guide and `pdf.Q#` for the FAQ.

| #  | Dimension                                                 | Weight | Score | Evidence / pointer                     |
|----|-----------------------------------------------------------|:-----:|:-----:|----------------------------------------|
| 1  | Task choice: verifiable + step-possible + non-zero-prob   |  3×   | 9/10  | §2                                     |
| 2  | OpenEnv-shaped env (reset/step/state/obs/reward/FastAPI)  |  3×   | 10/10 | §3                                     |
| 3  | RLVR (programmatic verifier, not learned RM)              |  3×   | 9/10  | §4                                     |
| 4  | Multiple independent reward functions + weights           |  3×   | 10/10 | §5.0                                   |
| 5  | Reward-hacking defences / adversarial self-test           |  3×   | **6/10**  | §5.1 (CoT loophole) + §5.2 (cliffs)  |
| 6  | SFT / warm-start before RL                                |  2×   | 9/10  | §6                                     |
| 7  | Curriculum learning (easy → hard)                         |  2×   | 10/10 | §7                                     |
| 8  | Verifier-first design, tests before training              |  2×   | 9/10  | §8                                     |
| 9  | "Start simple" reward progression                         |  1×   | 7/10  | §9                                     |
| 10 | TRL + Unsloth stack                                       |  2×   | 10/10 | §10                                    |
| 11 | GRPO / RLVR for the training algorithm                    |  2×   | 10/10 | §11                                    |
| 12 | Deployed as HF Space (one URL, FastAPI + Docker)          |  2×   | 10/10 | §12                                    |
| 13 | Monitoring: reward + component columns + sampled outputs  |  2×   | **7/10**  | §13 (no mid-training sampling)      |
| 14 | QLoRA/LoRA save path (no 4→16 bit upcast merge)           |  1×   | **7/10**  | §14 (not documented)                |
| 15 | Before/after proof (baseline vs trained, plots)           |  2×   | 9/10  | §15 (smoke artifacts generated, 300-step incomplete) |
| 16 | Test coverage (env + reward + deploy + training)          |  2×   | 10/10 | §16                                    |
| 17 | Docs (README, ARCHITECTURE, CHANGELOG, Space README)      |  2×   | 10/10 | §17                                    |
| 18 | Production stability (timeouts, web UI, deterministic)    |  1×   | 9/10  | §18                                    |
| 19 | Reproducibility (seeds, pinned bank, audit JSONs)         |  1×   | 10/10 | §19                                    |
| 20 | Theme fit / novelty                                       |  2×   | 9/10  | §20                                    |
|    | **Weighted aggregate**                                    | 40×   | **8.54 / 10** | §21                            |

The weighting mirrors the guidance: reward-hacking defences, RLVR, and
multi-component reward design each get 3×, because the guidance spends the
most real estate on those. The 8.54 is pulled down from 9+ almost entirely
by **dimension 5 (reward hacking)** — see §5.1.

---

## 2. Task choice — 9/10

**Guidance (docx.§1, pdf.Q15):** *"Pick a task with a clear success condition,
a verifier you trust, short-to-medium horizons, and adjustable difficulty.
The probability of a good answer must be greater than zero."*

**What we have:**
- `<budget>N</budget><answer>...</answer>` is a **single-step** task with a
  **deterministic format contract** (pdf.Q54: "structured extraction with
  schema validation" is explicitly listed as an ideal hackathon task).
- Success is **three separable, verifiable checks** stacked on top of each
  other: format parses, tokens-used ≤ budget, correctness ≥ 0.3 judge
  score. All three are programmatic — no subjective review needed.
- Difficulty is adjustable via `COMPLEXITY_IDEAL_TOKENS = {easy: 15,
  medium: 60, hard: 130}` **and** via the 4-phase curriculum.
- Probability of success > 0 for a Qwen2.5-3B-Instruct base model on easy
  prompts — confirmed by the 50-step smoke run (baseline eval returned
  non-zero mean reward on the 3-prompt starter holdout).

**Why not 10/10:** the guidance also flags "multi-step interaction matters"
(pdf.Q15). This env is **single-step** (one `reset()`, one `step()`,
`done=true` always). That's fine — single-step RLVR is explicitly
supported (pdf.Q54 lists code-generation-with-tests which is also
single-step) — but it leaves one bullet of the guidance unclaimed.

---

## 3. OpenEnv integration — 10/10

**Guidance (docx.§4, §5, pdf.Q6–Q7):** *"Expose reset/step/state, implement
as FastAPI, deploy as a Space, use the openenv CLI to scaffold."*

**What we have** (file paths relative to `token_efficiency_env/`):
- `server/token_efficiency_env_environment.py` — the env class with
  `reset()`, `step()`, `current_task`, `observation`, `reward` fields.
- `server/app.py` — FastAPI app, `/reset`, `/step`, `/health`, `/docs`,
  and `/web` (interactive UI) mounted correctly. Today's fix also makes
  it robust to both `token_efficiency_env.server.app` and bare `server.app`
  import paths, which is the exact form HF Spaces uses.
- `server/Dockerfile` — multi-stage build, port 8000, works both locally
  (`docker build .` in `server/`) and on the Space.
- `openenv.yaml` — declares `name`, `version`, `entry_point`,
  `action_schema`, `observation_schema`. Passes
  `tests/test_space_deploy_config.py` schema checks.
- `client.py` — a `TokenEfficiencyEnv` WebSocket client that survives
  curriculum state across calls (the common OpenEnv gotcha that HTTP
  hides; see `README.md` §4's explicit warning).
- `deploy/push_to_hf_space.py` — a `HfApi`-based deploy script that
  doesn't require the `openenv` CLI to be installed (which solved a real
  blocker — the team spent time early on fighting `openenv push`).

**Score:** full marks. The guidance's 5-bullet minimum
(`reset`/`step`/`state`/`reward`/`FastAPI`) is a proper subset of what's
shipped.

---

## 4. RLVR (verifiable rewards, not a learned reward model) — 9/10

**Guidance (pdf.Q4, Q21, Q41):** *"Use a verifier, tester, or environment
that can check correctness more directly. Prefer executable checks over
stylistic heuristics. Verifiable rewards are often binary and directly
tied to correctness criteria."*

**What we have** (`token_efficiency_env/scorer.py` + `judge.py`):
- **Correctness (55%)** — LLM-as-judge via HF Inference
  (`meta-llama/Llama-3.1-8B-Instruct`) with **keyword-overlap fallback**
  whenever the judge fails (rate-limit, timeout, network error).
- **Efficiency (15%)** — pure-Python: `1.0 if tokens ≤ ideal else decay`
  based on the Qwen tokenizer count. No learned component.
- **Self-assessment (15%)** — pure-Python asymmetric penalty around the
  predicted `<budget>`. No learned component.
- **Redundancy (5%)** — pure-Python `1 - repeated_word_ratio`.
- **Keyword verification (5%)** — pure-Python string/regex check against
  `expected_keywords`.
- **Format quality (5%)** — pure-Python regex + well-orderedness.

**Five of six components are deterministic verifiable rewards.** Even the
one learned component (correctness) has a deterministic fallback. That's
exactly the RLVR pattern the guidance recommends.

**Why not 10/10:** pdf.Q33 explicitly warns that "just use an LLM as judge
is often risky" because the judge becomes part of the optimization target.
Our judge is 55% of the reward — the single largest weight — and we have
no **judge-stress-test** (run N canned adversarial answers past the judge,
confirm the judge doesn't give high scores to them). Keyword-overlap is a
decent backstop but it's not adversarially verified. One hour of work.

---

## 5. Reward design + anti-hacking — 6/10 (dimension-dragging problem)

This is the one section where we materially underperform the guidance.

### 5.0 Multi-component reward — **10/10** (good)

Guidance (docx.§7, pdf.Q27, Q40): *"use multiple independent reward
functions, not just one. If you only have a single reward signal, it is
easier for the model to hack it."* We ship **6 components + 5 hard cliffs
= 11 independent signals**, each with a weighted contribution or a
deterministic `-1.0` / `-0.5` override. The weights (0.55 / 0.15 / 0.15 /
0.05 / 0.05 / 0.05) sum to 1.0 and are documented in `README.md` and
`ARCHITECTURE.md` §6.

### 5.1 Hidden-chain-of-thought exploit — **UNFIXED** (dragging this section)

**This is the single most important item in the report.** The guidance
(pdf.Q12, Q28, Q31, Q52, docx.§8) is extremely explicit that a reward
channel will be optimized to whatever flaw it has. The audit document
`VULNERABILITY_FIX_PLAN.md` §1 flagged it as **Critical** severity. Status:
**still not patched as of `3efe2d7`**.

Concrete exploit (verbatim from `VULNERABILITY_FIX_PLAN.md` §1.1):

```text
Hmm let me think step by step. The capital of France is famous for...
[...500 tokens of CoT...]
<budget>3</budget><answer>Paris.</answer>
```

Current scoring path:
- `re.search(r"<answer>(.*?)</answer>")` — silently discards the 500-token
  prefix.
- `tokens_used = _count_tokens(answer)` — counts only "Paris." → 1 token.
- Efficiency = 1.0, self-assessment ≈ 1.0, correctness ≈ 1.0.
- **Composite reward ≈ 0.97** while the model actually burned ~500 tokens.

**Why this will dominate training.** This isn't a theoretical concern.
GRPO gives deterministic exploits with massive advantage (here,
~0.97 reward on a ~0 truth) the fastest possible gradient. Within
hundreds of steps this is the behaviour the model learns, *not* token
efficiency. It's the canonical "specification gaming" pattern pdf.Q26
warns about.

**Recommended fix** (from the plan, not executed — 3 layers, ≤50 LOC):
1. Layer A: `tokens_used = _count_tokens(raw)` not `_count_tokens(answer)`.
2. Layer B: record `answer_token_count` as a diagnostic on the observation.
3. Layer C: switch parser from `re.search` to
   `re.match(r"^\s*<budget>(\d+)</budget>\s*<answer>(.*?)</answer>\s*$",
   raw, re.DOTALL)` and `bad_format` any response with text outside the
   tags.

**This is the single highest-leverage change before submission.**

### 5.2 Anti-hacking cliffs — **9/10** (good)

Five deterministic cliffs (`bad_format`, `empty`, `parrot`, `repetition`,
`too_long`) bypass the LLM judge entirely and return `-1.0` or `-0.5`.
That's exactly the "layered verification" pattern pdf.Q44 describes:
hard outcome checks + anti-cheat constraints + minimal shaping on top.

**Minor subtractions** (from `VULNERABILITY_FIX_PLAN.md`, already noted in
the audit but not blocking submission):
- Bigram redundancy overfires on 2-word answers (§5.2 of the plan). Fixed
  by clamping redundancy to 1.0 when `len(answer.split()) < 3`. XS.
- Parrot guard won't fire on current starter bank but will on full-bank
  prompts that share a noun with the answer. Small but real.

### 5.3 Adversarial self-testing — **missing** (dragging)

Guidance (pdf.Q57): *"Do not optimize a reward you have not tried to break
yourself first."* We have `tests/test_anti_cliff.py` which exercises the
five cliffs, but there is **no test that directly demonstrates the CoT
exploit does / does not score high**. If the CoT fix lands, add an
adversarial test: feed the verbatim exploit string, assert `reward < 0.3`.

### 5.4 Static difficulty saturation — **pdf.Q35, Q46, Q48**

Guidance: static datasets saturate. We **mitigated** this by switching to
the 2,300-prompt programmatic bank (`prompt_bank_mode="full"`, active on
Colab). That lands us in RLVR with a large bank, not RLVE. RLVE (pdf.Q22,
Q23) is the stretch; the guidance doesn't require it.

### Net score for §5: **6/10.** The CoT loophole is a critical blocker.

---

## 6. SFT warm-start — 9/10

**Guidance (docx.§3, pdf.Q16):** *"Start from a capable instruct model. Add
a tiny amount of task-format SFT if needed. Use RL only after the model
can occasionally succeed."*

**What we have:**
- `training/sft_format_data.py` — 50 synthetic `(question,
  formatted_response)` pairs, with the `<budget>N</budget>` count
  converged to exactly match the tokenized response length (that's the
  nice part — the SFT examples teach the *arithmetic* of the budget, not
  just the format).
- `notebooks/train_grpo.ipynb` cell "6.5) SFT format warmup" runs one
  epoch on those 50 examples via `trl.SFTTrainer` before GRPO starts.
- Gated by `config.use_sft_warmup=is_colab()` so local tests don't try
  to load the SFT path.
- `tests/test_sft_format_data.py` covers format regex, tier balance, and
  budget-arithmetic correctness.

**Why not 10/10:** the SFT pass happens on-the-fly every Colab run rather
than shipping a pre-SFT-ed checkpoint. That's fine for a 50-example
warmup, but the trade-off of not saving the SFT'd checkpoint is that
every full run redoes it. Minor.

---

## 7. Curriculum — 10/10

**Guidance (docx.§6, pdf.Q14):** *"Start with short horizons, fewer tools,
simpler state spaces, stronger hints, easier test cases, then gradually
remove scaffolding."*

**What we have** (`token_efficiency_env/server/token_efficiency_env_environment.py`
`CURRICULUM_PHASES`):
- 4 phases, each a `(easy, medium, hard)` sampling triple:
  `(1.0, 0.0, 0.0)` → `(0.6, 0.4, 0.0)` → `(0.3, 0.4, 0.3)` → `(0.2, 0.4, 0.4)`.
- Promotion rule: `avg_reward` over last 50 episodes ≥ phase threshold
  (0.40 / 0.50 / 0.60).
- Implemented **server-side** — the training script is curriculum-blind.
  That is the cleanest OpenEnv pattern: curriculum is a property of the
  environment, not the trainer.

Full marks. This directly mirrors the guidance.

---

## 8. Verifier-first design — 9/10

**Guidance (docx.§4, pdf.Q56):** *"Design the environment before you design
the trainer. First debug the environment manually. Then debug the
verifier. Then run scripted baselines. Then run a frozen model. Then run
a tiny RL experiment. Only then scale."*

**What we have:**
- The env + reward + cliffs were built and tested **four phases before**
  the training notebook. `CHANGELOG.md` phase markers confirm this: Phase
  3 (reward redesign) landed at `v0.2.0`; Phase 7 (training scaffold)
  landed later at `v0.3.0-ish`.
- `scripts/demo_session.py` is a scripted-baseline demo (5 fixed trials +
  a "smart" trial) that runs against either a local server or the
  deployed Space. That's the "scripted policy before RL" step.
- `tests/test_reward_invariants.py` and `tests/test_anti_cliff.py` exercise
  the verifier across 100+ canonical inputs.
- 50-step smoke before the 300-step full run is exactly the "tiny RL
  experiment before scale" order.

**Why not 10/10:** there is no `test_judge_stress.py` — we verify the
scorer but haven't adversarially tested the LLM judge (§4 above).

---

## 9. "Start simple" reward progression — 7/10

**Guidance (pdf.Q39, Q44):** *"Start simple, shape carefully. Start with hard
outcome checks. Add anti-cheat constraints. Then add minimal shaping only
where the sparse reward is too weak."*

We arrived at 6 components + 5 cliffs on day one, which is a lot. That's
justified by the task (token efficiency is multi-objective by nature — a
correctness-only reward would learn verbose correct answers, a
length-only reward would learn empty fluent garbage), but it does mean we
have **no single-component baseline to A/B against** and can't point at a
graph showing "efficiency component alone gives X, 6 components give Y,
here's why the extras were worth it."

Not critical for submission; would strengthen the story.

---

## 10. TRL + Unsloth stack — 10/10

**Guidance (docx.§10, pdf.Q8, Q25):** *"TRL for RL training algorithms,
Unsloth to make RL training and inference more efficient, OpenEnv to
standardize environment interaction."*

We have:
- `trl.GRPOTrainer` + `trl.GRPOConfig` — `notebooks/train_grpo.ipynb`
  cells 7 and 11.
- `peft.LoraConfig` for parameter-efficient training (r=16, α=32, 7
  target modules).
- `unsloth.FastLanguageModel` opt-in via `config.use_unsloth=is_colab()`.
  Linux-platform-gated in `requirements-train.txt`.
- `trl.SFTTrainer` for the format warmup (§6).
- TRL paper-index default config (lr=1e-5, num_generations=8, max_steps=300).

Nothing missing.

---

## 11. GRPO / RLVR training algorithm — 10/10

**Guidance (pdf.Q9, Q25, pdf.Q45):** *"GRPO was described as a more
efficient evolution relative to older PPO-style setups, especially by
simplifying away parts like the value model."*

We use GRPO, not PPO, on `Qwen2.5-3B-Instruct`. No value model, LoRA
adapters save memory, Unsloth on Colab T4 shrinks VRAM another ~2×. The
notebook matches the "[Unsloth Qwen2.5 3B GRPO notebook]" starter recipe
pdf.Q59 explicitly recommends.

---

## 12. HF Space deployment — 10/10

**Guidance (docx.§13, pdf.Q7):** *"OpenEnv environments are designed to be
deployed as Hugging Face Spaces, which provide a running server, a Git
repository, and a container registry. Deploy early to catch API and
packaging issues."*

**We deployed** and **verified**:
- Space is live, `/web` serves the interactive UI,
  `/docs` serves Swagger, `/health` returns 200.
- API smoke-test from local machine was run successfully against the Space
  (user confirmation).
- `deploy/push_to_hf_space.py` is idempotent (--dry-run flag), ignores
  `tests/`, `__pycache__/`, `.git/`, etc.
- Today's fix to `deploy/push_to_hf_space.py` automatically injects `ENV
  ENABLE_WEB_INTERFACE=true` into the staged Dockerfile so the `/web` UI
  works without the user manually setting a Space variable.
- Today's fix to `server/app.py` (`except ImportError` instead of
  `except ModuleNotFoundError`) makes it robust to both import paths HF
  uses.

Nothing missing.

---

## 13. Training-time monitoring — 7/10

**Guidance (docx.§15, pdf.Q17, Q43, Q52):** *"Monitor more than the headline
reward. Track component rewards, timeouts, format adherence, diversity,
and sample outputs during training. A rising reward is not enough if the
model is learning to exploit bugs."*

**What we have:**
- `training/reward_adapter.py` logs every rollout with 6-component
  breakdown + `tokens_used` + `allocated_budget` + `complexity`. That's
  the component-column logging the guidance asks for.
- `training/compare_report.py` generates `before_after_report.md` +
  `before_after_plot.png` + `reward_curve.png` — **post-hoc** artifacts.
- 50-step smoke run produced exactly these artifacts on Colab (user
  downloaded them).

**What's missing (why 7/10):**
- **No mid-training sampling callback.** The guidance is explicit (pdf.Q17:
  *"periodically sampling outputs during training rather than letting runs
  continue blindly"*). TRL exposes a `logging_steps` hook; we could print
  3 rollouts every 50 steps. This is ~10 LOC.
- **No live reward-by-component dashboard** while the notebook is running.
  The RewardLog accumulates, but there's no in-cell plot showing
  component trends every N steps. Also ~10 LOC (`display(HTML(...))` on
  a plotly figure).
- **No per-prompt overshoot tracking** during training (only in eval).

None of these are critical for submission. If you have 30 min, add (1).

---

## 14. QLoRA / LoRA save path — 7/10

**Guidance (docx.§16, pdf.Q8):** *"Do not upcast a 4-bit model to 16-bit and
then merge the LoRA weights naively. That can badly damage model quality.
Instead, use the proper merged-save path, or use the adapters directly."*

**Our notebook** calls `trainer.save_model(config.output_dir)` — this saves
the LoRA adapters correctly. But:
- We don't explicitly document the "don't upcast-then-merge" warning in
  the notebook or README.
- We don't call `model.save_pretrained_merged(...)` (Unsloth's safe merge
  helper) — so a judge or downstream user who wants a single merged
  checkpoint would likely do the naive-merge path and lose quality.
- `notebooks/train_grpo.ipynb` should have one markdown cell explaining:
  *"To load this trained model, use the adapter path: `PeftModel.from_pretrained(base_model, 'outputs/grpo_qwen2.5_3b')`.
  Do NOT merge unless you use `save_pretrained_merged` — see Unsloth
  docs."*

One cell of prose. Nothing breaks today, but it's a landmine for anyone
using the artifact.

---

## 15. Before/after proof — 9/10

**Guidance (docx.§19, pdf.Q57):** *"Baseline model attempt, reward/verifier
output, trained model attempt, measurable improvement, short explanation
of safeguards."*

**What we have:**
- `training/compare_report.py` generates a 5-metric table (mean reward,
  correctness, tokens used, cliff rate, overshoot rate) with delta arrows
  → `before_after_report.md`.
- Per-tier breakdown (easy / medium / hard).
- Three qualitative sample pairs (same prompt, baseline vs trained
  answer).
- `before_after_plot.png` bar chart + `reward_curve.png` training-time
  curve.
- 50-step smoke report was generated successfully on Colab.

**Why not 10/10:** the 50-step smoke report **showed mixed signals** —
mean reward was positive but `SMOKE PASSED: False` per the pass
criteria. The 300-step full run (which is what the judge should see) is
**incomplete** due to the Colab debugging session running out of time.
The fallback artifact is a 50-step smoke, which is honest but weak. If
you have the Colab session still open, finishing a 150-step run would
comfortably produce stronger numbers — 300 is the target but 150 is the
minimum believable improvement.

---

## 16. Test coverage — 10/10

| Test file                           | Tests | Covers                                        |
|-------------------------------------|------:|-----------------------------------------------|
| `test_reward_invariants.py`         |  ~20  | 6-component math, curriculum, edge cases      |
| `test_anti_cliff.py`                |  ~10  | 5 cliffs fire when they should                |
| `test_prompt_bank_loader.py`        |   15  | Dispatch, schema, keyword extraction (today's apostrophe regression) |
| `test_training_scaffold.py`         |  ~12  | TrainingConfig, smoke preset, split           |
| `test_sft_format_data.py`           |   10  | SFT warmup dataset                            |
| `test_compare_report.py`            |    8  | Before/after report rendering                 |
| `test_colab_bootstrap.py`           |    5  | Colab detection, token prompting              |
| `test_space_deploy_config.py`       |   17  | Space README, Dockerfile, push_to_hf_space.py |
| `test_demo_session.py`              |    3  | demo_session keywords, connection error       |
| `test_e2e_websocket.py`             |    3  | Skipped unless server is up                   |
| **Total**                           | **147 passing + 7 skipped** |                    |

Coverage is broad and includes deploy-time config, not just runtime. The
guidance doesn't actually ask for this much test coverage — this is above
what most hackathon submissions ship.

---

## 17. Documentation — 10/10

Each document has a clear audience:

| File                                     | Audience                                  |
|------------------------------------------|-------------------------------------------|
| `README.md` (repo root)                   | Judge / new user / cursor-opener          |
| `ARCHITECTURE.md`                         | Reviewer who wants the "why"              |
| `CHANGELOG.md`                            | Release-note reader                       |
| `deploy/SPACE_README.md`                  | Visitor to the HF Space                   |
| `deploy/README.md`                        | Operator re-deploying the Space           |
| `token_efficiency_env/README.md`          | Package-internal contributor              |
| `notebooks/train_grpo.ipynb` (markdown)   | Colab runner                              |
| `test_check_docs/*.md`                    | Internal audit (you/me)                   |

Full marks. The Colab badge in the root README is a 2026 judge's
single-click entry point.

---

## 18. Production stability — 9/10

**Guidance (docx.§8):** *"lock down execution, add time limits, avoid
unrestricted global state, sample frequently."*

**What we have:**
- `MAX_ANSWER_TOKENS = 500` cliff catches runaway generations.
- HF Inference has its own request timeout (default 30s), which we don't
  override.
- Deterministic seed (`TrainingConfig.seed=42`) for split + stratified
  sampling.
- Threading lock on `RewardLog` so the notebook can snapshot mid-training.
- Single-process in-process adapter; no shared mutable state across
  rollouts other than curriculum (which is the point).

**Why not 10/10:** execution-sandboxing doesn't apply (we only score
strings, not execute code), but the guidance does mention "add time
limits" and we have no explicit server-side per-request timeout wrapper.
`slowapi` or `asyncio.timeout` on the `/step` endpoint would give us a
guaranteed p99 latency. Not blocking.

---

## 19. Reproducibility — 10/10

- Seed 42 pinned in `TrainingConfig` → used for stratified split,
  dataset shuffling, generator seeding.
- `training/prompt_bank_report.json` records git SHA + timestamp + source
  counts + samples for every programmatic bank build.
- `pyproject.toml` pins `openenv-token_efficiency_env==0.x.y`.
- `requirements-train.txt` pins TRL / accelerate / peft minor versions.
- HF Space is tagged by git commit in `openenv.yaml`.
- Colab notebook clones a specific repo URL — any Colab runner gets the
  same code.

Fully reproducible; full marks.

---

## 20. Theme fit & novelty — 9/10

**Guidance (docx.§0, §20):** *"A specialized LLM system that can act inside
an environment, get feedback, and improve through RL."*

**Our theme** is token efficiency / budget-honest LLMs. That's:
- **Verifiable** (token counts are deterministic, judge is the only
  stochastic piece).
- **Novel** — `comparative_analysis.md` §4 makes a credible case that the
  *asymmetric self-assessment* component (training a model to predict its
  own compute budget *before* answering) is under-researched. Test-time
  compute budgets are common (o1, DeepSeek R1), but **self-predicted**
  budgets are not.
- **Deployable** — token efficiency maps directly to real-world API cost.
- **Demo-able** — the before/after on a 30-word answer shrinking to a
  3-word answer is an immediately legible judge moment.

**Why not 10/10:** the broader stretch (§10 in the original plan) would be
an RLVE curriculum where difficulty is model-capability-adaptive, not
just stage-gated. That's a stretch goal the guidance highlights (pdf.Q22,
Q35, Q46) and would bump us to 10. Not submission-blocking.

---

## 21. Weighted aggregate — 8.54 / 10

```
weighted_sum = Σ (weight × score)
             = (3×9 + 3×10 + 3×9 + 3×10 + 3×6 + 2×9 + 2×10 + 2×9 + 1×7
              + 2×10 + 2×10 + 2×10 + 2×7 + 1×7 + 2×9 + 2×10 + 2×10
              + 1×9 + 1×10 + 2×9)
             = 341
weight_sum    = 40
aggregate     = 341 / 40 = 8.525 ≈ 8.5
```

---

## 22. Cross-referenced guidance pitfalls

This table maps every "common mistake to avoid" bullet from docx.§21 and
pdf.Q52/Q57 to our status. Green = we're clearly fine. Yellow = partial /
verify. Red = unaddressed.

| Guidance pitfall                                         | Status |  Where / why                                                              |
|----------------------------------------------------------|:-----:|---------------------------------------------------------------------------|
| Picking a task so hard success probability is zero       | 🟢    | Easy-tier prompts are solvable by base Qwen 3B without fine-tuning.       |
| Using only one reward function                           | 🟢    | 6 components + 5 cliffs.                                                  |
| Not checking for reward hacking                          | 🟡    | 5 cliffs + no judge stress test + **CoT loophole unfixed**.               |
| Training before the environment is stable                | 🟢    | Env shipped at Phase 3; training at Phase 7.                              |
| Relying only on average reward and not inspecting outputs | 🟡   | Component columns are logged; no mid-training samples (§13).              |
| Forgetting timeouts and sandbox limits                   | 🟢    | `too_long` cliff, HF inference timeout. No code execution.                |
| Saving LoRA/QLoRA models incorrectly                     | 🟡    | Adapter save is correct; merge path undocumented (§14).                   |
| LLM judge gaming                                         | 🟡    | Judge has keyword fallback; no adversarial stress test (§4).              |
| Rising reward but falling task quality (Goodhart)        | 🔴    | CoT loophole — reward rises, quality drops. Submission-critical (§5.1).   |
| Static-dataset saturation                                | 🟢    | 2,300-prompt programmatic bank active on Colab.                           |
| Environment diversity (one task family)                  | 🟡    | Single task family (Q→A with budget). Fine for hackathon scope.           |
| Skipping SFT warmup                                      | 🟢    | 50-example SFT format warmup in the pipeline.                             |

---

## 23. What a judge will actually experience

1. **Clicks the README Colab badge.** Notebook opens. First cell
   (bootstrap) installs everything in ~3 min on T4.
2. **Opens the HF Space URL** (from `CHANGELOG.md` or README). Sees `/web`
   UI: Reset → type `<budget>5</budget><answer>Paris</answer>` → Step →
   reward ≈ 0.97. Sees `/docs` Swagger schema.
3. **Reads `ARCHITECTURE.md` §5–6** for the reward-design rationale.
4. **Opens `training/before_after_report.md`** — **currently shows 50-step
   smoke, not 300-step full**. This is the weakest judge-facing artifact.
5. **Reviews `tests/`** — 147 passing, good coverage of reward edge cases
   + deploy config.
6. **If they're adversarial,** they try the CoT exploit string and will
   find that `tokens_used` is wrong (see §5.1). Whether they actually try
   this depends on how deeply they read `VULNERABILITY_FIX_PLAN.md`. The
   plan document is in the repo tree.

**Risk:** a sharp reviewer who reads `VULNERABILITY_FIX_PLAN.md` and notices
the CoT fix is unimplemented will mark this down. The plan itself is a
point in our favour (shows we're aware), but an *unfixed critical-severity
item* in our own audit is a credibility drag.

---

## 24. Non-blocking recommendations, prioritised

> User asked for "report only, no changes." These are recommendations
> only; any of them can be implemented later with one-to-three-line
> commits.

| #  | Change                                                      | Effort | Leverage for score |
|----|-------------------------------------------------------------|--------|:------------------:|
| R1 | **Fix CoT loophole** (VULNERABILITY_FIX_PLAN §1, Layers A+B+C) | 30 LOC | **+0.8 pts**       |
| R2 | Finish 150-or-300-step full GRPO run, regenerate comparison report | Colab compute | **+0.4 pts**  |
| R3 | Add mid-training sampling callback to `train_grpo.ipynb` (print 3 rollouts every 50 steps) | 10 LOC | +0.2 pts  |
| R4 | Add `tests/test_cot_exploit.py` asserting the verbatim exploit string scores `< 0.3` | 15 LOC | +0.1 pts |
| R5 | Document QLoRA safe-merge path in notebook + README          | 1 cell | +0.1 pts           |
| R6 | Add judge stress test (3 canned adversarial answers, confirm judge ≤ 0.3) | 30 LOC | +0.1 pts |
| R7 | Redundancy clamp to 1.0 when `len(answer.split()) < 3` (VULNERABILITY_FIX_PLAN §2) | 3 LOC | +0.05 pts |
| R8 | Parrot-guard false-positive fix (VULNERABILITY_FIX_PLAN §5)  | 15 LOC | +0.05 pts          |

**R1 alone would push the aggregate from 8.5 to 9.3.** It's the single
highest-ROI change. R2 pushes to 9.5 with no code changes beyond
triggering the Colab run. R3–R8 together would add another ~0.4.

Ceiling with all R1–R8: **~9.7 / 10.**

---

## 25. Bottom line for submission

**Current state is submittable.** The project is strongly OpenEnv-compatible,
deployed, tested, documented, and conceptually novel. The 8.5 aggregate is
already above what most hackathon submissions will score.

**But three things are low-hanging fruit:**

1. **R1 (CoT fix).** ~30 min, +0.8 pts. **Highest-priority.**
2. **R2 (finish a full GRPO run).** ~2 hours of Colab + 10 min of artifact
   regeneration. This is what the judge actually sees.
3. **R5 (document QLoRA save path).** 10 min, avoids a landmine.

Everything else is a nice-to-have.

---

*End of report. Generated 2026-04-26 from HEAD `3efe2d7`. Nothing in the
project tree was modified to produce this document.*
