# Changelog

All notable changes to **TokenEfficiencyEnv** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project loosely follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — HuggingFace Space deploy scaffolding (hackathon execution plan §8)

Packages up everything needed to push `TokenEfficiencyEnv` to a
HuggingFace Space (`sdk: docker`, `app_port: 8000`) **without requiring
the `openenv` CLI to be installed**. Gives a judge a one-click URL to
poke the env's `/web` UI, and gives a trainer a stable `base_url` to
point their `TokenEfficiencyEnv(...)` client at.

- **`deploy/SPACE_README.md` (new).** Judge-facing Space README with
  full HF YAML frontmatter (`title`, `sdk`, `app_port: 8000`,
  `base_path: /web`, `tags: openenv`). Swapped in for the
  contributor-facing `token_efficiency_env/README.md` at deploy time so
  people visiting the Space landing page see a 30-second pitch + the
  action/observation schema, not the in-package contributor guide.
- **`deploy/push_to_hf_space.py` (new).** Python helper with a clean
  CLI (`--dry-run`, `--repo-id`, `--private`, `--base-image`,
  `--hardware`, `--verbose`). Uses `huggingface_hub.HfApi` directly
  (`create_repo` + `upload_folder`) — no dependency on the `openenv`
  CLI. Key design choices:
  - `DEFAULT_IGNORE_PATTERNS` excludes `__pycache__`, `.pytest_cache`,
    `tests/`, hidden files, etc., keeping the Space upload to ~18
    real env files.
  - `_stage_for_upload()` copies the env to a tempdir, overwrites
    `README.md` with `SPACE_README.md`, and mirrors
    `server/Dockerfile` to the repo root (where HF Spaces looks for
    it). The original `server/Dockerfile` stays intact so local
    `docker build` still works.
  - `_build_plan()` is pure and network-free, so `--dry-run` can show
    the exact file list + target repo id without touching HF.
- **`deploy/README.md` (new).** Operator's guide (not the Space landing
  page): one-time setup, dry-run flow, common flags, post-deploy
  wiring of `HF_TOKEN` and `JUDGE_MODEL` as Space secrets/variables,
  and troubleshooting.
- **`tests/test_space_deploy_config.py` (new, 17 tests).** Pure-config
  validation — no network calls:
  - Files on disk: `SPACE_README.md`, `deploy/README.md`,
    `push_to_hf_space.py`, env's `openenv.yaml` + `server/Dockerfile`.
  - Frontmatter: valid YAML, required keys (`title`, `sdk`,
    `app_port`), correct values (`sdk: docker`, `app_port: 8000`,
    `base_path: /web`, `openenv` tag present).
  - Ignore patterns: catch `.git/`, `__pycache__/`, `tests/`,
    `.pytest_cache/`; keep `models.py`, `scorer.py`, `server/app.py`,
    etc.
  - End-to-end: `_build_plan()` returns a sensible plan;
    `_stage_for_upload()` produces a staging dir with a
    swapped README *and* a root-level Dockerfile; `main(--dry-run)`
    exits 0 without HF credentials.
- **Zero changes to `token_efficiency_env/`.** The env package stays
  deploy-unaware: `openenv.yaml`, `Dockerfile`, and `README.md` (the
  contributor guide) are untouched. All Space-specific concerns live
  under `deploy/`.

**Test status:** 142 passed, 7 skipped, 0 failed (`tests/test_e2e_websocket.py`
still skipped when no live server; unchanged from prior runs).

### Added — Judge-ready before/after comparison artifacts (hackathon execution plan §7)

Turn the notebook's in-cell DataFrame + single matplotlib figure into a
**package of files a hackathon judge can review without running the
notebook**. Addresses the "demo recording" (§9) + "submission polish"
gap: the old cell was fine for the author scrolling the notebook, but
useless for anyone reading the repo on GitHub or trying to screenshot a
result into a submission blurb.

- **`training/compare_report.py` (new).** Four pure-Python helpers,
  designed to be import-safe on a box without a GPU / matplotlib:
  - `build_comparison(baseline, trained)` — aggregates per-tier metrics
    (easy / medium / hard) from `EvalSummary.per_prompt`. Works on
    dataclasses, plain dicts, or `SimpleNamespace` (so tests don't have
    to import the eval module's concrete type).
  - `render_markdown_report(baseline, trained, ...)` — writes
    `before_after_report.md`: headline table with ▲/▼ improvement
    arrows (with a legend explaining that arrows are semantic, not
    sign-following), per-tier breakdown, 3 qualitative samples
    prioritised by cliff→clean transitions, plots embedded via
    relative markdown image links.
  - `plot_before_after(...)` — 5-metric bar chart PNG.
  - `plot_reward_curve(reward_log, ...)` — rolling-mean training-time
    trajectory PNG; returns `None` and skips the write if the log is
    empty (smoke runs with `max_steps` below `num_generations`).
  - All plot helpers lazily switch matplotlib to the `Agg` backend so
    CI / pytest-on-a-laptop work without a display.
- **`notebooks/train_grpo.ipynb` — cell 13 updated.** The existing
  in-notebook `display(comp)` DataFrame + live `plt.show()` chart + the
  `metrics.json` write are kept intact (the change is purely additive).
  Three new files land in `training/` at the end of cell 13:
  `before_after_report.md`, `before_after_plot.png`, `reward_curve.png`.
  `SMOKE` flag is threaded through so the report header correctly
  labels whether it's a smoke or full run.
- **`scripts/smoke_compare_report.py` (new).** Generates the three
  artifacts from synthetic data in ~2 seconds — no models, no HF API
  calls. Catches rendering regressions (missing columns, broken tables,
  matplotlib backend issues) without needing to wait for a real run.
- **`tests/test_compare_report.py` (new, 8 tests).** Covers
  aggregation correctness (per-tier means match hand-computed values),
  tier exclusion when a tier is absent, markdown section presence,
  graceful degradation when baseline and trained evaluated different
  prompt sets, and non-empty PNG writes. `matplotlib.pyplot` tests use
  `pytest.importorskip` so the suite still passes on a machine without
  matplotlib installed.

Validated: `pytest tests/ -m "not slow"` → **124 passed**, 7 skipped
(pre-existing WebSocket skips), 0 failures (from 116 at §5).

### Added — Smoke-run preset + Colab playbook (hackathon execution plan §5)

Ship a one-toggle smoke run that exercises the **full** pipeline
(bootstrap → Unsloth model load → SFT warmup → GRPO → eval → report)
on a Colab free T4 in ~15-20 min, so we catch OOM / TRL drift / API
issues before committing to the 1-2h full run.

- **`training/config.py` — `TrainingConfig.smoke(colab=...)` classmethod.**
  Pre-scaled run: 50 steps, 4 generations, keyword judge (zero HF API
  calls, zero rate-limit risk), starter bank (skips the 2.3k loader
  download), `eval_steps=25` (one mid-training probe), output dir
  namespaced `..._smoke` so checkpoints don't clobber the full run.
  `colab=True` keeps Unsloth + SFT warmup on — those integration
  points are exactly what smoke is designed to stress-test.
- **`notebooks/train_grpo.ipynb` — `SMOKE = True` toggle in the config
  cell.** One boolean flip cycles between `TrainingConfig.smoke(...)`
  and the full Step-6 config. Expected cost for smoke: ~200 judge
  calls (all local keyword), ~15-20 min wall time on T4.
- **`notebooks/train_grpo.ipynb` — Colab playbook markdown cell.**
  Placed right after the bootstrap cell so a cold reader sees it
  before configuring anything. Walks through runtime selection
  (T4 GPU), expected smoke output (cliff_rate drop, reward rise),
  and the flow for moving to the full run.
- **`notebooks/train_grpo.ipynb` — new cell 13.5 "smoke report".**
  Writes `training/smoke_report.json` (tight before/after delta + a
  PASS/FAIL verdict based on `reward_improved ∧ format_improved`) and
  prints a 10-line console summary. This is the artifact the user
  copies back to verify Step 5 passed before running Step 6.
- **`tests/test_training_scaffold.py` — 3 new tests.**
  `test_config_smoke_preset_has_expected_shape`,
  `test_config_smoke_preset_colab_false_disables_gpu_paths`,
  `test_config_smoke_estimate_judge_calls_is_cheap` pin the smoke
  preset's run cost and integration-point flags so a future config
  refactor can't silently blow past the 300-judge-call budget.

Validated: `pytest tests/ -m "not slow"` → **116 passed**, 7 skipped
(pre-existing WebSocket skips), 0 failures.

### Added — SFT format-teaching warmup before GRPO (hackathon execution plan §4)

Fixed the "GRPO cold start" failure mode: a fresh Qwen2.5-3B has never
seen `<budget>N</budget><answer>...</answer>`, so on step 1 every rollout
cliffs out at `bad_format` (-0.8 reward), every group gets the same
reward, and the group-relative policy gradient is identically zero —
the model can't learn anything until it stumbles onto the format by
chance. A one-epoch SFT pass on 50 synthetic examples gets the model
emitting valid format >95% of the time in ~2 min on a T4, so GRPO has
a real reward signal from step 1 instead of wasting the first 50-100
steps on format search.

- **`training/sft_format_data.py` (new).** Pure-Python builder for the
  50-example warmup set. Pulls 24 terse (question, answer) pairs from
  `STARTER_PROMPTS` via the `_first_keyword_as_answer` helper (handles
  flat strings, flat lists, and nested `["9", "nine"]` §7 aliases) plus
  26 hand-crafted extras covering easy/medium/hard tiers. Crucially,
  budget `N` is computed via a fixed-point loop so
  `N == len(tokenize(<budget>N</budget><answer>...</answer>))` exactly
  — the model learns honest self-prediction, not a fixed heuristic.
  Deterministic under `seed`.
- **`training/config.py` — 5 new fields.** `use_sft_warmup` (bool, off
  by default), `sft_warmup_epochs` (1), `sft_warmup_examples` (50),
  `sft_warmup_lr` (2e-4), `sft_warmup_batch_size` (4). `describe()`
  renders the warmup status inline so the banner shows exactly what's
  configured. Post-init validation guards against `epochs < 1` or
  `examples < 1`.
- **`notebooks/train_grpo.ipynb` — new cell 6.5.** Runs conditionally
  on `config.use_sft_warmup`. Builds the HF chat-formatted dataset,
  configures `trl.SFTConfig` + `SFTTrainer` with the same
  bf16/fp16/LoRA setup the GRPO phase will use, trains for 1 epoch,
  and prints a first-row preview so you can eyeball the format being
  taught. Skipped entirely (with a clear log line) when disabled.
- **`notebooks/train_grpo.ipynb` — config cell.** Automatically flips
  `use_sft_warmup=is_colab()` — on Colab the warmup runs before GRPO;
  locally it stays off so dev iteration stays fast.
- **`tests/test_sft_format_data.py` (new, 10 tests).** Proves the
  dataset has 50 examples by default, every response matches the
  scorer's format regex, every complexity tier is represented,
  deterministic seed, keyword-handler behaves on all three keyword
  shapes, `_converge_budget` terminates even on oscillating tokenizers,
  `n=10` cap works, fake tokenizer injection keeps `budget` within ±1
  of actual token count.

Validated: `pytest tests/ -m "not slow"` → **113 passed**, 7 skipped
(pre-existing WebSocket skips), 0 failures.

### Added — Full prompt bank end-to-end wiring (hackathon execution plan §3)

Flipped the last switch on Phase 8: when you run the notebook on Colab
it now actually trains on the ~2.3k programmatic bank, not the 24-prompt
starter. Verified against real HuggingFace data before shipping.

- **`scripts/smoke_prompt_bank.py` (new).** One-command end-to-end
  smoke test for `prompt_bank_loader.py`. Runs the loader against real
  HF data, validates the shared schema (`prompt` / `complexity` /
  `expected_keywords`) on every entry, emits
  `training/prompt_bank_report.json` with per-tier counts, 3 human-readable
  samples, git SHA + timestamp for audit. CLI flags let you scope to a
  subset of sources (`--sources gsm8k`) or cap the prompt count
  (`--max-prompts 50`) for cheap local validation.
- **`training/prompt_bank_report.json` (new).** First smoke-test receipt:
  50 GSM8K prompts loaded in 23s, numeric aliasing confirmed working
  (e.g. `["9", "nine"]`, `["60", "sixty"]` per plan §7), schema clean
  across all entries.
- **`training/prompts_split.py` — mode-aware API.** `train_prompts`,
  `holdout_prompts`, `probe_prompts`, and `describe_split` now accept
  an optional `mode` argument ("starter" / "full"). Zero-arg calls stay
  identical to previous behaviour (starter bank, unchanged indices),
  so every existing caller keeps working. When `mode="full"` is passed,
  the module lazy-loads the programmatic bank via `get_prompt_bank` and
  computes a fresh stratified split. Cached per mode so rerunning
  notebook cells doesn't re-invoke the loader.
- **`notebooks/train_grpo.ipynb` — config cell.** Now sets
  `prompt_bank_mode="full"` automatically when `is_colab()` is True.
  All four downstream call sites (`train_prompts`, `holdout_prompts`
  twice, `describe_split`) thread `config.prompt_bank_mode` through,
  so before/after evals use the same bank the model trained on.
- **`tests/test_training_scaffold.py` — 3 new tests.**
  `test_train_prompts_mode_none_matches_starter` pins back-compat:
  `mode=None` must be identical to `mode="starter"`.
  `test_split_mode_full_uses_injected_bank` proves the full-mode split
  actually partitions a stubbed 120-prompt bank with every tier present
  in holdout + probe, without corrupting the starter module constants.
  `test_describe_split_mode_full_renders_without_dumping_every_prompt`
  covers the "switch to 5-sample summary for big banks" UX so the
  notebook output stays readable.

Validated: `pytest tests/ -m "not slow"` → **103 passed**, 7 skipped
(pre-existing WebSocket skips), 0 failures.

### Added — Colab-ready training notebook (hackathon execution plan §2)

Makes `notebooks/train_grpo.ipynb` one-click runnable on Google Colab's
free tier — critical because the GRPO trainer needs a CUDA GPU and we
don't assume contributors have one locally.

- **`training/colab_bootstrap.py` (new).** Ships `is_colab()` (pure
  detection, no side effects) and `bootstrap()` which, on Colab only,
  git-clones the repo into `/content`, `pip install`s the env package +
  `requirements-train.txt` + `unsloth`, and securely prompts for
  `HF_TOKEN` via `getpass` (session-only, never written to disk). On
  any non-Colab kernel `bootstrap()` is a strict no-op so existing
  editable installs continue to work unchanged.
- **`notebooks/train_grpo.ipynb` — new cell 0.** Self-heals on first
  Colab run (inline git clone if `training` isn't importable yet) then
  delegates to `training.colab_bootstrap.bootstrap`. Downstream cells
  are untouched.
- **Config cell flips `use_unsloth=is_colab()`.** Colab free T4 gets
  the ~2× speed / ~40% VRAM Unsloth path by default; local runs stay on
  the tested vanilla HF + PEFT path.
- **README.md — "Open In Colab" badge** at the top, plus a one-line
  callout directing GPU-less users straight to the notebook.
- **`tests/test_colab_bootstrap.py` (new, 6 tests).** Covers
  `is_colab()` returning False in CI, `bootstrap()` being a true no-op
  outside Colab (cwd unchanged, tmp root untouched), the
  `training.bootstrap` / `training.is_colab` re-exports, and all three
  `HF_TOKEN` prompt paths (already-set short-circuit, valid paste,
  blank → keyword fallback).

Validated with `pytest tests/ -m "not slow"`: **100 passed, 7 skipped
(pre-existing WebSocket skips), 0 failures.**

### Added — Phase 8 scaffolding (§6 of `VULNERABILITY_FIX_PLAN.md`)

The "stop over-fitting a 24-prompt bank" groundwork. All pieces ship
**off** by default — the existing starter bank remains the canonical
path for CI / demo / dashboard — but every seam the plan calls for is
now in place and tested.

- **`token_efficiency_env/prompts.py` — loader dispatch.** The 24-prompt
  list is now exported as `STARTER_PROMPTS`, with `PROMPT_BANK` kept as a
  back-compat alias. New `get_prompt_bank(mode=...)` picks between
  `"starter"` (default) and `"full"`, honouring the `PROMPT_BANK_MODE`
  env var. On `ImportError` from the full-bank loader (missing
  `datasets`), `get_prompt_bank` logs a warning and falls back to
  `STARTER_PROMPTS` — training jobs that don't need the full bank never
  pay for it.
- **`token_efficiency_env/prompt_bank_loader.py` (new).** Lazy,
  opt-in loader for the programmatic ~2.3k-prompt bank sourced from
  GSM8K + TriviaQA + ARC + OpenOrca (per plan §6.A). Every entry is
  emitted in the shared schema (`prompt` / `complexity` / `expected_keywords`),
  with complexity tier derived from Qwen-token count of the reference
  answer so `COMPLEXITY_IDEAL_TOKENS` stays calibrated, and numeric
  keywords expanded to `[digit, word_form]` via `num2words` to match §7.
  Raises a clear `ImportError` when `datasets` is absent.
- **`training/prompts_split.py` — stratified splitter.**
  Hard-coded 18/6 split replaced with `stratified_split(prompts, seed)`
  that partitions by `complexity`, shuffles under a fixed seed, and
  allocates 85/10/5 train/holdout/probe with a **small-tier safety rule**
  that guarantees ≥1 prompt in both holdout and probe for tiers under 20.
  Exposes `TRAIN_INDICES` / `HOLDOUT_INDICES` / `PROBE_INDICES` plus
  `train_prompts()` / `holdout_prompts()` / `probe_prompts()` and a
  `dump_split()` helper that writes `training/prompts_split.json`
  (stable-sorted) for PR-diffable audits. Starter-bank split is now
  **18 / 3 / 3**.
- **`training/config.py` — Phase 8 knobs.** `TrainingConfig` gains
  `prompt_bank_mode`, `train_fraction`, `holdout_fraction`,
  `probe_fraction`, and `eval_steps`, with a `__post_init__` that
  validates the mode name, the fraction sum, and the eval cadence. The
  `describe()` banner now shows the active prompt-bank mode, split, and
  eval cadence up front.
- **`requirements-train.txt`.** Added `num2words>=0.5.13` (for numeric
  alias expansion in the full loader). `datasets` was already present.
- **Tests.**
  - `tests/test_prompt_bank_loader.py` (new, 14 cases): dispatch
    contract, env-var resolution, ImportError fallback to starter,
    cache invalidation via `reset_prompt_bank_cache()`, loader helpers
    (`_extract_keywords`, `_add_numeric_aliases`,
    `_bucket_by_expected_length`, `_build_entry`) producing the
    canonical schema, and an opt-in `@pytest.mark.slow` integration
    smoke-test that actually pulls GSM8K over the network.
  - `tests/test_training_scaffold.py`: legacy `== 6` holdout assertion
    replaced with **stratified invariants** — disjointness, full
    partition, every tier represented in holdout and probe,
    determinism under a fixed seed, sensitivity to seed changes,
    allocator table-driven cases (`_allocate_counts`), ±5% tier ratio
    on a 300-prompt synthetic bank, and a `dump_split()` JSON-shape
    test.
- **`pytest.ini`.** Registered the `slow` marker so the
  integration smoke-test is selectable via `-m slow` without
  `PytestUnknownMarkWarning`.

### Deferred (from Planned)

- **Phase 5** — Deploy as a HuggingFace Space (Docker SDK). Deferred to last.
- **§6.D full rollout** — flipping default `PROMPT_BANK_MODE` to `full`
  requires a ≥50-step pilot with `datasets` installed to confirm that
  the judge cost (`num_generations × max_steps`) and per-tier reward
  distribution stay sane on the new bank. Scaffolding is done; the
  empirical validation step is the open item.
- **PR #5 recalibration** — `CURRICULUM_PHASES.advance_threshold` and
  `COMPLEXITY_IDEAL_TOKENS` may need a second pass once the same pilot
  lands on the v0.3.0 reward distribution.

---

## [0.3.0] — 2026-04-26

The **"reward is honest"** release. Closes 7 of the 8 items in
`test_check_docs/vulnerability_audit.md` / `test_check_docs/VULNERABILITY_FIX_PLAN.md`
(only §6, "memorisation / 24-prompt over-fit", is deferred to Phase 8 as
tracked under "Planned" above).

### Fixed — anti-hacking / scoring correctness

- **§1 Hidden chain-of-thought loophole (critical).** The parser no longer
  uses `re.search`, so a model can't emit hundreds of tokens of scratch
  reasoning before `<budget>` and be scored only on the inner `<answer>`.
  - *Layer A:* `tokens_used = _count_tokens(raw)` — the full shell is
    priced, not just the inner answer.
  - *Layer B:* new diagnostic field `TokenEfficiencyObservation.answer_token_count`
    reports the inner-only count for dashboards.
  - *Layer C:* new `SHELL_RE = r"^\s*<budget>(\d+)</budget>\s*<answer>(.*?)</answer>\s*$"`
    with `re.match`; any non-whitespace outside the tags → `bad_format` cliff.
- **§2 Bigram redundancy ramp.** Short answers no longer collect a bigram
  penalty they couldn't avoid. Bigram weight ramps linearly from 0 at
  `len(words) < 8` to 1 at `len(words) >= 16`; below 8 words, only
  `unique_ratio` is used.
- **§3 Format-vs-efficiency double-penalty removed.** `format_quality`
  no longer subtracts 0.3 for `len(words) < 5` on medium/hard or `> 30`
  on easy — those length checks now live solely in `efficiency`, which
  already scores them on a continuous axis.
- **§4 Whitespace hygiene cleanup.** Dropped the dead
  `response != response.strip()` half of the condition (env already
  strips); kept the live `"  " in response` internal-double-space check.
- **§5 Parrot guard switched to Jaccard.** Brittle substring check
  replaced with a token-set Jaccard ≥ 0.85 rule and a `len(p_tokens) < 4`
  short-prompt exemption. Literal echo still fires; natural restated
  answers ("The boiling point of water is 100 °C.") no longer false-positive.
- **§7 Numeric keyword aliases — list-of-lists schema.**
  `expected_keywords` now accepts either a flat `str` ("paris") or a
  `list[str]` alias group (`["30", "thirty"]`); matching *any* form counts
  the group as one satisfied keyword. Both `scorer.keyword_verification`
  and `judge._keyword_score` (keyword judge / HF fallback) understand the
  new shape. Updated the 4 numeric prompts in `PROMPT_BANK`.
- **§8 Duplicate parrot heuristic in `scorer.format_quality` — deleted.**
  It penalised any short, correct answer whose only word came from the
  prompt ("France." → `format_quality -= 0.3`). Redundant with the new
  Jaccard env-level guard.
- **`models.py` `Optional` import regression.** The v0.3.0 diagnostic
  field added `answer_token_count: Optional[int]` but didn't import
  `Optional`, which broke `TokenEfficiencyObservation` instantiation at
  Pydantic validation time for every successful `step()`. Added the
  missing import.

### Changed

- **`COMPLEXITY_IDEAL_TOKENS`** retuned by +12 for every tier
  (`easy` 15→27, `medium` 60→72, `hard` 130→142) to absorb the wrapper-
  token cost now counted by `tokens_used` (§1 Layer A). Measured overhead
  with the Qwen2.5-3B tokenizer: 10 / 11 / 12 tokens for 1- / 2- / 3-digit
  budgets respectively; `+12` is the safe upper bound. Pre- and post-v0.3
  reward numbers are therefore **not** directly comparable.

### Added — tests

- **`tests/test_anti_cliff.py`** (new, 15 cases) — covers every exploit
  §1 + §5 rule out: leading CoT before tags, trailing text after `</answer>`,
  text between tags, clean-shell with surrounding whitespace, raw-token-count
  invariants, literal / natural / padded parrot variants, short-prompt parrot
  exemption, plus a parametrised `_is_parrot` unit table.
- **`tests/test_reward_invariants.py`** — extended by 13 parametric cases
  for §2 (four bigram-ramp boundary tests + monotonicity) and §7 (six
  list-of-lists schema cases, including the strict-numeric "30 must not
  match 300" invariant inside an alias group). `TASK_*` indices are now
  looked up by prompt text so the bank can be reordered without silently
  breaking assertions (caught during this release).

### Migration notes

- Any `tokens_used` / `recent_rewards` / `avg_reward_50` values recorded
  before v0.3.0 are **not** directly comparable: the numeric shift is
  roughly `+10–12 tokens` and `-0.02 to -0.05` mean reward at parity,
  mostly absorbed by the `COMPLEXITY_IDEAL_TOKENS` bump above.
- Curriculum `advance_threshold`s (`[0.4, 0.5, 0.6]`) were calibrated
  against v0.2.0 reward numbers. They remain unchanged in this release;
  a pilot run will determine whether they need a downward nudge (tracked
  under "Planned" above).

---

## [0.2.1-unreleased-scaffold] — (previously marked "Unreleased")

### Added — Phase 7: Training scaffold

- **`training/`** package — trainer-side glue, kept out of `token_efficiency_env/` so the env stays a pure environment with no ML-framework deps.
  - `config.py` — `TrainingConfig` dataclass holding every knob (model, LoRA, GRPO hyperparams, reward backend, eval settings) with an `estimate_judge_calls()` cost guard so you see "this run will burn N API calls" *before* hitting Train.
  - `prompts_split.py` — deterministic 18-train / 6-holdout split (last 2 of each tier). Hard-coded indices, not random-seeded, so reviewers can read which prompts the model was *not* trained on. Import-time `_assert_split_invariants()` blows up loudly if `prompts.py` ever drifts under us.
  - `server_pool.py` — spawn / health-check (`/health`) / SIGTERM-then-SIGKILL teardown of N uvicorn workers on consecutive ports. Pre-flight checks ports are free; never leaks zombie workers on startup failure. Kept around for the optional WebSocket reward backend.
  - `reward_adapter.py` — bridges `TokenEfficiencyEnvironment` into TRL's `reward_funcs(prompts, completions)` API. Two backends:
    - `InProcessRewardAdapter` (default) — instantiates one env per rollout slot, drives them directly. No servers, no ports, no async; the trainer owns prompt selection.
    - `WSRewardAdapter` — talks to a real `ServerPool` over the WebSocket client. Optional; useful for stress-testing the deployed server path.
    - Both share `__call__` + `close()`, so the notebook flips backends with one config flag.
    - Per-step reward + components logged to a thread-safe `RewardLog` for post-training plots.
- **`notebooks/`** — Phase 7 entrypoints.
  - `train_grpo.ipynb` — 16-cell scaffold: setup → config banner → split → optional pool → reward smoke test → tokenizer + model + LoRA → dataset build → baseline eval → `GRPOTrainer.train()` → trained eval → side-by-side table + reward curve → cleanup.
  - `eval_baseline_vs_trained.py` — reusable eval harness (mean reward / correctness / tokens-used / overshoot rate / cliff rate) shared between the notebook and CLI users.
  - `__init__.py` + `README.md`.
- **`requirements-train.txt`** — `trl`, `peft`, `accelerate`, `transformers`, `datasets`, `bitsandbytes` (skipped on macOS), `jupyter`, `matplotlib`, `pandas`. Kept separate from `pyproject.toml` so the runtime env package stays slim.
- **`tests/test_training_scaffold.py`** — 18 offline smoke tests covering config invariants, split disjointness/tier coverage, in-process adapter happy path + cliff propagation + defensive paths + completion-text normalisation + factory dispatch. Whole suite runs in ~30 s with the keyword judge.

### Fixed — Phase 7

- **`InProcessRewardAdapter` truthy-vs-`is None` bug** — `RewardLog` is falsy when empty (via `__len__`), so `self.log = log or RewardLog()` silently swapped the caller's instance for a new one and every appended row landed in a log nobody could read. Rewritten as `self.log = RewardLog() if log is None else log` (and same in `WSRewardAdapter` + `build_reward_func`). Caught by the new tests.
- **Keyword scoring stem matching** — `judge.matches_keyword()` now uses prefix-stem matching for word keywords (`antibod` → `antibodies`, `purchas` → `purchasing`, `intelligen` → `intelligence`) while keeping strict full-word matching for purely numeric keywords (so `100` doesn't match `1000`). Both `judge._keyword_score` and `scorer.py`'s keyword_verification component go through the same primitive. Photosynthesis prompt's `expected_keywords` updated to `["plant", "sunlight", "oxygen"]` so a natural answer scores 1.0 instead of 0.0.
- **`get_judge()` aliases** — accepts `hf` / `inference` / `remote` for HuggingFace and `offline` / `local` for keyword. Misspellings still raise a clear `ValueError`. Previously a typo broke every `/api/step` silently.
- **Web dashboard prompt mismatch** — `/api/step` now returns the prompt that was actually scored, and the frontend displays "Scored against:" so a UI/backend question mismatch is immediately visible. Question dropdown re-resets the env on `onchange`.

---

## [0.2.0] — 2026-04-25

The "make the env trainable for real" release. Phases 1–4 of the implementation plan landed in this version, plus a Phase 6 docs/test cleanup.

### Added

- **`token_efficiency_env/server/token_efficiency_env_environment.py`** — canonical `Environment` implementation. Hosts curriculum sampling, the rolling 50-episode reward deque, the lazy Qwen tokenizer, and all anti-hacking guards.
- **`token_efficiency_env/judge.py`** — pluggable judge backend layer. `Judge` Protocol, `HFInferenceJudge` (uses `huggingface_hub.InferenceClient` against the Inference Providers router; default model `meta-llama/Llama-3.1-8B-Instruct`), `KeywordJudge` (offline word-boundary keyword overlap), and a `get_judge()` factory keyed off `JUDGE_BACKEND` / `JUDGE_MODEL` / `HF_TOKEN`. The HF judge transparently falls back to keyword scoring on individual API failures so a network blip can't crash a training run.
- **Anti-hacking cliffs** for `empty` (answer < 2 chars) and `parrot` (normalised question is a substring of the answer). Both bypass the 6-component formula and return fixed negative rewards.
- **`tests/`** suite (pytest):
  - `test_reward_invariants.py` — fast in-process tests for the reward formula and cliffs (offline, no `HF_TOKEN` needed).
  - `test_e2e_websocket.py` — integration tests that exercise the WebSocket round-trip against a running server (auto-skip if no server is up).
- **`ARCHITECTURE.md`** — single-source-of-truth onboarding doc covering the 6-component reward, anti-hacking cliffs, curriculum, two-LLMs setup (trainee Qwen vs judge Llama), HTTP-vs-WebSocket statefulness gotcha, and run instructions.
- **`CHANGELOG.md`** (this file).
- `pytest.ini` at the repo root for one-command test discovery.
- `pyproject.toml`: `keywords`, `classifiers`, `[project.urls]`, and a `dev` extra with `pytest`/`pytest-cov`/`httpx`.

### Changed

- **Reward function — 7 components → 6 components, weights rebalanced** (`scorer.py`):
  - Correctness 0.40 → **0.55** (the dominant signal — we want correctness first).
  - Efficiency 0.20 → **0.15** and reformulated against an *absolute* per-complexity ideal (`easy=15`, `medium=60`, `hard=130` tokens) instead of being relative to the model's own budget (so the model can't max it by predicting a tiny budget).
  - Self-Assessment 0.10 → **0.15** and reformulated as **asymmetric**: mild penalty for slack (`budget >> tokens_used`), steep penalty for overshoot (`tokens_used > budget`). The previous symmetric form rewarded matching the budget exactly, which contradicted Efficiency's "use less" pressure.
  - Redundancy 0.10 → **0.05**.
  - Keyword Verification 0.05 → **0.05** (unchanged weight, now uses word-boundary matching to avoid `"30" in "300"` style false positives).
  - Format Quality 0.05 → **0.05** (unchanged).
- `models.py` rewritten to define a typed `TokenEfficiencyAction` (`raw_response: str`) and a rich `TokenEfficiencyObservation` carrying `prompt`, `episode_token_limit`, `answer`, `allocated_budget`, `tokens_used`, `complexity`, `phase`, `episode`, `avg_reward_50`, `reward_components`, and `error`.
- `client.py` rewritten to subclass OpenEnv's `EnvClient` over WebSockets and unpack the new observation fields with `_step_payload` / `_parse_result`.
- `server/app.py` pins `max_concurrent_envs=1` with an explanatory comment. OpenEnv's HTTP layer has no session affinity, so a pool > 1 makes `/reset` and `/step` race onto different env instances.
- `openenv.yaml`: `entry_point` and schema paths updated to point at the canonical modules; description updated to reflect the 6-component design.
- `pyproject.toml`: version `0.1.0` → `0.2.0`. Replaced `anthropic` with `huggingface_hub`. Added `transformers`, `tokenizers`, `pydantic`.
- `server/requirements.txt`: same dependency rotation as `pyproject.toml`.
- `env_server.py` and `env_client.py` reduced to thin back-compat shims that re-export the new modules with a `DeprecationWarning`.

### Removed

- **`budget_reasonableness` reward component** — it rewarded memorising a fixed per-complexity prior rather than learning. Double-dipped with `efficiency`.
- `_phase3_smoketest.py` and `_phase4_e2e.py` from the repo root — replaced by `tests/test_reward_invariants.py` and `tests/test_e2e_websocket.py` respectively.
- All `anthropic` dependencies and `ANTHROPIC_API_KEY` references — judge is HuggingFace Inference now.

### Fixed

- **Prompt drift between `reset` and `step`** — caused by `max_concurrent_envs > 1` round-robining requests across stateless env instances. Now pinned to 1 and documented in `ARCHITECTURE.md` §9.
- **`KeywordJudge` false positives** like `"30" in "300"` and `"central" in "centralized"` — now uses word-boundary regex matching.
- **HF judge poor scoring on math** — judge prompt now passes `expected_keywords` as anchor "key facts" with a clear scoring rubric and a `Score: X.XX` output contract; the parser pulls the score with regex.
- Inline doc staleness in `judge.py` and `models.py` (model name and component count).

### Documentation

- Both `README.md` files (root + `token_efficiency_env/`) rewritten to match the current Phase 1–4 design (was: 7 components, Anthropic judge, outdated quick-start API).
- `web_dashboard.py` + `dashboard.html` ported to delegate scoring to the real `TokenEfficiencyEnvironment` instead of carrying their own stale 7-component reimplementation. Frontend updated to render the 6 components with current weights.

---

## [0.1.0] — 2026-04 (initial)

First public version of the environment. Established:

- The `<budget>N</budget><answer>text</answer>` action contract.
- A 7-component reward function (correctness, efficiency, budget reasonableness, redundancy, self-assessment, keyword verification, format quality).
- A 4-phase curriculum that ramps difficulty as the trainee's rolling 50-episode average reward crosses thresholds.
- A bank of 24 hand-curated questions in `prompts.py` (8 easy / 8 medium / 8 hard).
- An interactive web dashboard (`web_dashboard.py` + `dashboard.html`) with a "training simulation" mode.
- Anthropic Claude Haiku as the LLM judge.
- A first pass at the OpenEnv server wrapper (later found to be using stale boilerplate; rebuilt in 0.2.0).

[Unreleased]: https://github.com/SaishAmbar/token-efficiency-env/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/SaishAmbar/token-efficiency-env/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/SaishAmbar/token-efficiency-env/releases/tag/v0.2.0
[0.1.0]: https://github.com/SaishAmbar/token-efficiency-env/releases/tag/v0.1.0
