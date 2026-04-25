# Changelog

All notable changes to **TokenEfficiencyEnv** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project loosely follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned

- **Phase 7** — Training notebook scaffold (TRL `GRPOTrainer` + Qwen2.5-3B + N parallel env servers).
- **Phase 5** — Deploy as a HuggingFace Space (Docker SDK). Deferred to last.

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

[Unreleased]: https://github.com/SaishAmbar/token-efficiency-env/compare/main...HEAD
[0.2.0]: https://github.com/SaishAmbar/token-efficiency-env/releases/tag/v0.2.0
[0.1.0]: https://github.com/SaishAmbar/token-efficiency-env/releases/tag/v0.1.0
