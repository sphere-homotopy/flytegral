# Fly Window Neural & Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the fixed MaleCNS task graph as the recurrent substrate of a trainable stochastic policy, train only visual-input/output-readout parameters with PPO, and demonstrate that the trained biological graph beats untrained and shuffled-connectome controls on held-out window-exit episodes.

**Architecture:** Convert the exported task graph to a normalized signed sparse recurrent matrix. Inject the 16-ray retina only into selected visual seed neurons, initialize neural activity to zero for each environment observation, evolve it for four recurrent microsteps through the fixed graph, and decode only designated descending neurons into yaw/thrust. v1 is therefore a reactive connectome policy with within-step neural dynamics, not a claim of biologically faithful temporal membrane state. PPO trains the input encoder, descending readout, and exploration variance; MaleCNS topology and internal graph weights remain frozen in v1.

**Tech Stack:** Python 3.12, PyTorch, numpy, scipy, pandas, pytest

**Spec:** `docs/superpowers/specs/2026-09-14-fly-window-design.md`

## Global Constraints

- MaleCNS task-graph topology remains fixed during v1 training.
- v1 resets neural activity to zero at every environment step; recurrent propagation happens only within four neural microsteps for the current visual observation.
- Connectome-derived internal edge weights remain fixed during v1 training after deterministic normalization/signing.
- Only visual encoder, descending readout, and policy exploration parameters are trainable.
- Visual input may enter only nodes marked by `input_mask`; action readout may consume only nodes marked by `output_mask`.
- Final evaluation uses a frozen checkpoint and exactly 100 held-out seeds, `10000..10099`.
- Release target is at least 80 successful exits out of 100 held-out episodes and at least a 10 percentage-point success-rate advantage over both untrained and shuffled-connectome controls.
- Training/evaluation artifacts must record config, source graph manifest hash, git commit, seed, and metrics.
- No final demo may be cherry-picked from training rollouts.
- All major completed checkpoints must be committed and pushed before starting the next major phase.

---

## File map

- Modify `pyproject.toml` — add PyTorch runtime dependency.
- Create `src/fly_window/neural/graph.py` — load graph artifact and build frozen signed recurrent matrix.
- Create `src/fly_window/neural/policy.py` — recurrent connectome policy and stochastic action distribution.
- Create `src/fly_window/neural/control.py` — deterministic shuffled-connectome control.
- Create `src/fly_window/training/buffer.py` — PPO rollout buffer/GAE.
- Create `src/fly_window/training/ppo.py` — PPO loss/update.
- Create `src/fly_window/training/run.py` — training loop/checkpoints/logging.
- Create `src/fly_window/evaluation/run.py` — frozen 100-seed evaluation.
- Create `configs/training_v1.json` — exact hyperparameters.
- Create `scripts/train.py`, `scripts/evaluate.py`.
- Create tests under `tests/neural`, `tests/training`, `tests/evaluation`.

### Task 1: Load and normalize the fixed MaleCNS recurrent graph

**Files:**
- Modify: `pyproject.toml`
- Create: `src/fly_window/neural/__init__.py`
- Create: `src/fly_window/neural/graph.py`
- Test: `tests/neural/test_graph.py`

**Interfaces:**
- Produces: `ConnectomeGraph(body_ids, src_idx, dst_idx, weights, input_mask, output_mask, transmitter_sign)`.
- Produces: `load_connectome_graph(npz_path: Path, nodes_parquet: Path) -> ConnectomeGraph`.
- Produces: `to_sparse_recurrent(graph: ConnectomeGraph) -> torch.Tensor` sparse COO shape `(N,N)`.
- Signing rule by presynaptic neurotransmitter: `gaba=-1.0`, `glutamate=-1.0`, `acetylcholine=+1.0`, `dopamine=+0.25`, `serotonin=+0.25`, `octopamine=+0.25`, missing/other=`+1.0`.
- Magnitude rule: `log1p(synapse_weight)` followed by per-destination absolute-sum normalization to `<=1.0`.

- [ ] **Step 1: Write a tiny signed-graph test**

Create three neurons and edges where an acetylcholine source and GABA source both feed one destination. Assert signs, normalized absolute incoming sum `1.0`, sparse shape, and that graph tensors have `requires_grad=False`.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/neural/test_graph.py -v`
Expected: FAIL.

- [ ] **Step 3: Add PyTorch and implement graph loading/signing/normalization**

Add `torch` to runtime dependencies. Map transmitter strings case-insensitively. Sparse matrix orientation must satisfy `next_input = torch.sparse.mm(W.transpose(0,1), activity[:,None])[:,0]`, i.e. each edge `src -> dst` contributes source activity to destination.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/neural/test_graph.py -v`
Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add pyproject.toml src/fly_window/neural tests/neural/test_graph.py
git commit -m "feat: build frozen MaleCNS recurrent matrix"
git push
```

### Task 2: Implement the connectome-constrained stochastic policy

**Files:**
- Create: `src/fly_window/neural/policy.py`
- Test: `tests/neural/test_policy.py`

**Interfaces:**
- Produces: `ConnectomePolicy(graph: ConnectomeGraph, recurrent: torch.Tensor, visual_dim=16, microsteps=4, leak=0.35)`.
- Trainable parameters only:
  - `input_gain`: shape `(input_count, visual_dim)`
  - `input_bias`: shape `(input_count,)`
  - `readout`: `nn.Linear(output_count, 2)`
  - `log_std`: shape `(2,)`
- Produces: `forward(obs: Tensor[B,16]) -> PolicyOutput(mean_action, activity)`, where `activity` is the final within-step neural state used for logging/rendering.
- Produces: `distribution(mean_action) -> Independent[Normal]` in latent action space; environment action transforms are `yaw=tanh(z0)`, `thrust=sigmoid(z1)`.

- [ ] **Step 1: Write parameter-isolation tests**

Assert all recurrent graph tensors are frozen; the exact trainable parameter names are `input_gain`, `input_bias`, `readout.weight`, `readout.bias`, `log_std`; no other trainable tensor exists.

- [ ] **Step 2: Write mask-enforcement tests**

Backpropagate a scalar output and assert gradients exist for encoder/readout but there is no graph-weight gradient. Set all non-input node activity injection opportunities to sentinel values and assert only `input_mask` indices receive sensory injection.

- [ ] **Step 3: Run and verify failure**

Run: `uv run pytest tests/neural/test_policy.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement recurrent microsteps**

For each forward call initialize `activity = torch.zeros((obs.shape[0], graph.node_count), device=obs.device)`. Then for each of four microsteps:

```python
recurrent_drive = torch.sparse.mm(recurrent.transpose(0, 1), activity.T).T
injected = torch.zeros_like(activity)
injected[:, input_indices] = obs @ input_gain.T + input_bias
proposal = torch.tanh(recurrent_drive + injected)
activity = (1.0 - leak) * activity + leak * proposal
```

Use an implementation that does not materialize dense `N x N` for the real graph; unit tests may use dense toy graphs. Read only `activity[:, output_mask]` into the 2D readout.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/neural/test_policy.py -v`
Expected: PASS.

- [ ] **Step 6: Commit and push Checkpoint C core**

```bash
git add src/fly_window/neural/policy.py tests/neural/test_policy.py
git commit -m "feat: drive actions through fixed MaleCNS graph"
git push
```

### Task 3: Add a deterministic shuffled-connectome control

**Files:**
- Create: `src/fly_window/neural/control.py`
- Test: `tests/neural/test_control.py`

**Interfaces:**
- Produces: `shuffle_destinations(graph: ConnectomeGraph, seed: int) -> ConnectomeGraph`.
- Control rule: keep every source index and edge weight fixed; deterministically permute the full `dst_idx` vector with `numpy.random.default_rng(seed).permutation`; input/output masks and node metadata remain attached to original node identities.

- [ ] **Step 1: Write invariance/disruption tests**

Assert same seed yields identical shuffled graph, different seed differs, source vector and edge-weight multiset are unchanged, destination-index multiset is unchanged, and at least one edge endpoint changes for a nontrivial graph.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/neural/test_control.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement control and collision aggregation**

After shuffling destinations, aggregate duplicate `(src,dst)` pairs by summing weights before rebuilding the recurrent matrix. Record control seed in the evaluation manifest.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/neural/test_control.py -v`
Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add src/fly_window/neural/control.py tests/neural/test_control.py
git commit -m "feat: add shuffled-connectome control"
git push
```

### Task 4: Implement GAE rollout storage and PPO losses

**Files:**
- Create: `src/fly_window/training/__init__.py`
- Create: `src/fly_window/training/buffer.py`
- Create: `src/fly_window/training/ppo.py`
- Test: `tests/training/test_buffer.py`
- Test: `tests/training/test_ppo.py`

**Interfaces:**
- `RolloutBuffer` stores observation, latent action, log_prob, reward, done, and value. Neural activity is logged separately for diagnostics/rendering but is not PPO state.
- `compute_gae(last_value, gamma=0.99, gae_lambda=0.95)` returns normalized advantages and returns.
- `ppo_loss(new_log_prob, old_log_prob, advantage, new_value, target_return, entropy, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01)` returns total/policy/value/entropy terms.

- [ ] **Step 1: Write hand-computed GAE test**

Use a 3-step deterministic reward/value sequence and assert advantages/returns against explicitly calculated numeric values to `1e-6`.

- [ ] **Step 2: Write PPO clipping test**

Construct ratios below, inside, and above `[0.8,1.2]`; assert the policy objective uses the clipped branch where expected and value/entropy coefficients match exact defaults.

- [ ] **Step 3: Run and verify failure**

Run: `uv run pytest tests/training/test_buffer.py tests/training/test_ppo.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement GAE and PPO losses**

No environment logic belongs in these files. Keep functions tensor-only and independently testable.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/training/test_buffer.py tests/training/test_ppo.py -v`
Expected: PASS.

- [ ] **Step 6: Commit and push**

```bash
git add src/fly_window/training tests/training
git commit -m "feat: add PPO optimization primitives"
git push
```

### Task 5: Build reproducible training and checkpoint logging

**Files:**
- Create: `configs/training_v1.json`
- Create: `src/fly_window/training/run.py`
- Create: `scripts/train.py`
- Test: `tests/training/test_run.py`

**Interfaces:**
- Training config exact defaults:
  - `seed=20260914`
  - `parallel_envs=16`
  - `rollout_steps=256`
  - `gamma=0.99`
  - `gae_lambda=0.95`
  - `clip_ratio=0.2`
  - `learning_rate=3e-4`
  - `ppo_epochs=4`
  - `minibatch_size=512`
  - `value_coef=0.5`
  - `entropy_coef=0.01`
  - `max_environment_steps=2000000`
  - `checkpoint_every_steps=50000`
  - `eval_every_steps=50000`
- CLI accepts `--control biological|shuffled`; `shuffled` applies `shuffle_destinations(graph, seed=314159)` before recurrent-matrix construction.
- Produces run directory `artifacts/runs/<run_id>/` with `config.json`, `manifest.json`, `metrics.jsonl`, `checkpoints/*.pt`.
- Run id: UTC `YYYYMMDDTHHMMSSZ` plus first 8 chars of graph `artifact_sha256`.

- [ ] **Step 1: Write a tiny fake-environment training test**

Inject a two-step fake vector environment and toy graph; run one rollout/update and assert metrics line, checkpoint, optimizer state, graph hash, git SHA field, and seed are present. Assert checkpoint reload reproduces deterministic action means for a fixed observation.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/training/test_run.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement training runner**

Use Adam only on `policy.parameters()` that require gradients. Store `control_kind` and, for shuffled runs, `control_seed=314159` in config/manifest. Add a separate linear value head fed from concatenated mean activity of output neurons plus current 16-element observation; value head is training-only and never feeds the action path. Clip gradient norm at `0.5`.

Checkpoint must include policy state, value-head state, optimizer state, total environment steps, RNG states for Python/numpy/torch, config, graph manifest hash, and git SHA from `git rev-parse HEAD`.

- [ ] **Step 4: Implement periodic evaluation hook on training seeds only**

During training, quick evaluation uses 32 fixed development seeds `9000..9031`; held-out seeds `10000..10099` are prohibited here.

- [ ] **Step 5: Run tests and CLI help**

Run: `uv run pytest tests/training/test_run.py -v && uv run python scripts/train.py --help`
Expected: PASS.

- [ ] **Step 6: Commit and push**

```bash
git add configs/training_v1.json src/fly_window/training/run.py scripts/train.py tests/training/test_run.py
git commit -m "feat: train connectome policy with reproducible PPO"
git push
```

### Task 6: Frozen evaluation against untrained and shuffled controls

**Files:**
- Create: `src/fly_window/evaluation/__init__.py`
- Create: `src/fly_window/evaluation/run.py`
- Create: `scripts/evaluate.py`
- Test: `tests/evaluation/test_run.py`

**Interfaces:**
- Produces `EvaluationSummary(model_label, episode_count, successes, success_rate, mean_reward, mean_steps, seed_start, seed_end)`.
- Held-out seeds exactly `10000..10099`.
- CLI evaluates three labels into one `evaluation.json`: `trained`, `untrained`, `shuffled`.

- [ ] **Step 1: Write held-out seed and frozen-weight tests**

Use fake policies with deterministic success/failure behavior. Assert exactly 100 unique seeds, no optimizer is instantiated, `model.eval()` is called, and parameters are byte-identical before/after evaluation.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/evaluation/test_run.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement evaluation runner**

For `trained`, load the frozen checkpoint. For `untrained`, instantiate the same architecture with seed `20260914` and no training. For `shuffled`, build the deterministic shuffled graph with seed `314159` and require a separately trained shuffled-control checkpoint produced by the same training runner and hyperparameters. The evaluation CLI takes explicit `--trained-checkpoint` and `--shuffled-checkpoint`; missing either is an error.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/evaluation/test_run.py -v`
Expected: PASS.

- [ ] **Step 5: Run real training/evaluation experiment**

Run training until one of:
- development success rate is `>=0.80` for three consecutive 50k-step evaluations, or
- `2,000,000` environment steps are reached.

Run `scripts/train.py` once with `--control biological` and once with `--control shuffled`, using the same step budget, hyperparameters, initialization seed, and development seeds. Then run held-out evaluation for trained, untrained, and shuffled.

Success criterion: trained `>=80/100` and trained success rate is at least `0.10` higher than each control. If not met, record the result and iterate scientifically on a new versioned config rather than altering metrics or cherry-picking seeds.

- [ ] **Step 6: Commit code/config and push Checkpoint D**

Do not commit large checkpoints/run data. Commit only any new versioned config needed plus summary JSON/CSV small enough for git.

```bash
git add src/fly_window/evaluation scripts/evaluate.py tests/evaluation configs
git commit -m "feat: add frozen held-out connectome evaluation"
git push
```

## Plan acceptance gate

Checkpoint C is reached when the real connectome graph controls actions end-to-end and all neural tests pass. Checkpoint D is reached only when a pushed code/config revision has produced an auditable learning run and 100-seed held-out evaluation. If the `>=80%` target fails, preserve the negative result and create a new versioned training config; never silently change the held-out seed set.
