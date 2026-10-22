# Fly Tweets v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a background-only MaleCNS/connectome-based English math/fly Twitter author that generates 10 tweets per day, publishes them through Google Sheets/Buffer, learns once daily from collected engagement, and automatically produces the next batch.

**Architecture:** Add a stateful text policy beside the existing Fly Window policy, with a fixed 1024-token vocabulary and fixed biological recurrent matrix. Build deterministic pretraining and public-launch gates first, then autoregressive generation/provenance, Sheet/Buffer integration, DOM metrics collection, conservative daily policy-gradient learning, and durable GitHub-to-Windows background orchestration.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, Google Apps Script, Buffer API, Playwright/Chromium headless, GitHub Actions, Windows self-hosted runner.

**Spec:** `docs/superpowers/specs/2026-09-17-fly-tweets-design.md`

## Global Constraints

- No LLM or other language model in generation, filtering, rewriting, ranking, reward assignment, vocabulary expansion, or daily training.
- Vocabulary size is exactly 1024 unique tokens and English-only for v1.
- Biological recurrent connectome weights remain fixed in v1.
- The fly/meme-specific subset remains below 5% of the full vocabulary.
- Exactly 10 tweets are generated after each successful daily update.
- Normal operation on the home PC is invisible: no console, browser, modal prompt, or manual start.
- Offline daily work remains durable and resumes automatically when the Windows self-hosted runner returns.
- No semantic rejection sampling; only structurally invalid outputs may be withheld/retried.
- Every substantial checkpoint is committed and pushed before the next major phase.

---

### Task 1: CI branch coverage, vocabulary, and stateful FlyTextPolicy

**Files:**
- Modify: `.github/workflows/tests.yml`
- Create: `src/fly_window/text/__init__.py`
- Create: `src/fly_window/text/vocabulary.py`
- Create: `src/fly_window/text/policy.py`
- Create: `tests/test_text_vocabulary.py`
- Create: `tests/test_text_policy.py`

**Interfaces:**
- Produces: `FlyVocabulary(tokens: tuple[str, ...])`, `build_v1_vocabulary() -> FlyVocabulary`
- Produces: `FlyTextPolicy.initial_state(batch_size)`, `FlyTextPolicy.step(token_ids, state) -> TextPolicyOutput`, `FlyTextPolicy.detach_state(state)`
- Consumes: existing `ConnectomeGraph` and sparse recurrent tensor from `fly_window.neural.graph`.

- [ ] Write failing vocabulary tests asserting exactly 1024 unique tokens, required math/fly tokens, English-only control policy, and meme-specific proportion <5%.
- [ ] Push the failing-test checkpoint and verify CI fails because the text module does not yet exist.
- [ ] Implement the minimal immutable vocabulary builder and token/id mappings.
- [ ] Write failing stateful-policy tests proving state changes across sequential token steps, resets to zero between tweets, recurrent tensor is non-trainable, and logits have shape `(batch, 1024)`.
- [ ] Push the failing-test checkpoint and verify the expected failure.
- [ ] Implement the minimal stateful policy using the existing input/output neuron masks and sparse recurrent matrix.
- [ ] Verify all existing and new tests pass in GitHub Actions; push the green checkpoint.

### Task 2: Deterministic curriculum and initial pretraining gate

**Files:**
- Create: `src/fly_window/text/curriculum.py`
- Create: `src/fly_window/text/pretraining.py`
- Create: `src/fly_window/text/gate.py`
- Create: `scripts/train_text.py`
- Create: `configs/text_training_v1.json`
- Create: `tests/test_text_curriculum.py`
- Create: `tests/test_text_pretraining.py`
- Create: `tests/test_text_gate.py`

**Interfaces:**
- Produces: deterministic `generate_curriculum(seed, count, vocabulary)` sequences.
- Produces: corpus tokenizer/filter that maps only the fixed vocabulary plus `<UNK>`.
- Produces: `evaluate_launch_gate(model, heldout, probe_seeds, config) -> GateReport`.

- [ ] Add failing tests for deterministic grammar generation and absence of out-of-vocabulary ids.
- [ ] Implement static grammar classes and seeded sampling without any model calls.
- [ ] Add failing tests for teacher-forcing batch preparation and scheduled-sampling transition behavior.
- [ ] Implement minimal pretraining utilities and CLI checkpoint/manifest writing.
- [ ] Add failing tests for EOS rate, repetition-run limit, finite-state/logit check, median-length bounds, and deterministic replay gate fields.
- [ ] Implement the gate and verify the full suite green; push checkpoint.

### Task 3: Autoregressive tweet generation and provenance

**Files:**
- Create: `src/fly_window/text/generation.py`
- Create: `src/fly_window/text/provenance.py`
- Create: `scripts/generate_tweets.py`
- Create: `tests/test_text_generation.py`
- Create: `tests/test_text_provenance.py`

**Interfaces:**
- Produces: `generate_tweet(model, vocabulary, seed, config) -> GeneratedTweet`.
- Produces: `generate_batch(..., count=10) -> list[GeneratedTweet]`.
- `GeneratedTweet` stores rendered text, token ids, token logprobs, RNG seed, termination reason, checkpoint id, git SHA, and batch id.

- [ ] Write failing tests for exact seed replay, BOS/reset behavior, EOS stop, 240-character cap, and exactly 10 outputs per batch.
- [ ] Implement minimal autoregressive sampler with fixed temperature and no semantic filtering.
- [ ] Write failing serialization tests for generation provenance.
- [ ] Implement JSON/Sheet-safe provenance serialization; verify green and push.

### Task 4: Google Sheet contract and stochastic Buffer scheduler

**Files:**
- Create: `apps_script/FlyTweetsQueue.js`
- Create: `apps_script/tests/fly_tweets_queue.test.js`
- Create or modify: `package.json` only if needed for Apps Script pure-function tests.
- Create: `src/fly_window/publishing/rows.py`
- Create: `tests/test_publishing_rows.py`

**Interfaces:**
- Sheet row schema matches the design spec.
- Pure scheduling function takes ten baseline slots plus RNG seed and returns jittered, sorted slots obeying sigma 22 min, clip 45 min, minimum gap 35 min.
- Buffer idempotency key is `batch_id:tweet_index`.

- [ ] Write failing Python tests for row serialization and stable idempotency key.
- [ ] Implement row serializer.
- [ ] Write failing JS tests for jitter bounds, ordering, minimum gap, and duplicate prevention.
- [ ] Implement Apps Script queue functions by adapting the existing Buffer queue pattern.
- [ ] Verify Python and JS tests green; push checkpoint.

### Task 5: Headless X metrics collector and reconciliation

**Files:**
- Create: `collector/collect-fly-tweet-stats.js`
- Create: `collector/package.json`
- Create: `collector/tests/metrics-parser.test.js`
- Create: `src/fly_window/metrics/reward_inputs.py`
- Create: `tests/test_reward_inputs.py`

**Interfaces:**
- Collector emits `tweet_url`, timestamp, text, views, likes, reposts, replies, bookmarks, collected_at.
- DOM parser reuses the proven `role=group` / `aria-label` metric extraction approach.
- Browser launch is always headless.

- [ ] Write failing parser fixtures for singular/plural/missing metric combinations.
- [ ] Implement DOM-label parsing and tweet-row extraction.
- [ ] Add tests ensuring collector launch configuration is headless-only and row reconciliation keys by status URL/id.
- [ ] Implement Sheet reconciliation payload generation; verify green and push.

### Task 6: Reward calculation and conservative daily learning

**Files:**
- Create: `src/fly_window/text/reward.py`
- Create: `src/fly_window/text/daily_training.py`
- Create: `scripts/run_daily_text_training.py`
- Create: `configs/daily_reward_v1.json`
- Create: `tests/test_text_reward.py`
- Create: `tests/test_daily_training.py`

**Interfaces:**
- `compute_daily_rewards(rows, config)` requires age >=24h and returns centered/rank-normalized tweet rewards.
- `run_daily_update(...)` consumes rows once, applies bounded policy-gradient update with entropy, anchor/KL term, clipping, writes immutable checkpoint/manifest, then generates the next ten tweets.

- [ ] Write failing reward tests covering age eligibility, log exposure, smoothed rates, weighting, centering, and stable rank normalization.
- [ ] Implement reward calculation.
- [ ] Write failing daily-update tests for double-consumption prevention, rollback on numerical failure, immutable checkpoint ids, and next-batch generation only after successful checkpoint write.
- [ ] Implement the minimal update transaction and training step; verify green and push.

### Task 7: Durable GitHub request orchestration and invisible Windows execution

**Files:**
- Create: `.github/workflows/fly-tweets-daily.yml`
- Create: `scripts/windows/fly_tweets_worker.ps1`
- Create: `scripts/windows/install_fly_tweets_worker.ps1`
- Create: `src/fly_window/orchestration/pending.py`
- Create: `tests/test_pending_training.py`

**Interfaces:**
- Daily request key is an ISO date; states are `pending`, `running`, `complete`, `failed`.
- Requests are processed serially by date.
- Runner/worker executes non-interactively; Python uses `pythonw.exe` or equivalent hidden-process launch and metrics browser is headless.

- [ ] Write failing state-machine tests for offline accumulation, oldest-first processing, retry idempotency, and completed-request immutability.
- [ ] Implement durable request manifest/state functions.
- [ ] Add workflow and Windows service scripts that never require a visible terminal or desktop browser.
- [ ] Verify workflow YAML, script syntax, and unit tests; push checkpoint.

### Task 8: End-to-end dry run, operational docs, and final verification

**Files:**
- Create: `docs/fly-tweets-operations.md`
- Create: `scripts/smoke_fly_tweets.py`
- Create: `tests/test_fly_tweets_smoke.py`
- Modify: `README.md`

**Interfaces:**
- Smoke flow: load test graph/checkpoint -> generate ten tweets -> serialize Sheet rows -> schedule slots -> ingest fixture metrics -> calculate rewards -> perform dry-run update -> generate next batch.

- [ ] Write a failing end-to-end smoke test using deterministic fixtures and no external services.
- [ ] Implement smoke orchestration until the deterministic pipeline passes.
- [ ] Document credential locations, one-time setup, failure recovery, and the background-only invariant.
- [ ] Run full pytest/JS tests and relevant CLI `--help` smoke checks in CI.
- [ ] Inspect final branch diff for accidental LLM dependencies, visible-browser launches, semantic filtering, or non-idempotent posting.
- [ ] Make the final commit, push, and report branch plus pushed SHA.
