# Fly Window Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 2D room simulator in which a fly can sense a bright open window with a coarse retina, collide with walls, and exit through the opening under a two-dimensional control signal.

**Architecture:** Keep physics intentionally simple and explicit. The environment owns geometry, dynamics, visual sensing, reward, and termination; neural code only sees the 16-element visual vector and sends `[yaw, thrust]`. A scripted oracle controller exists solely as a sanity check that the task is physically solvable and is never available to training.

**Tech Stack:** Python 3.12, numpy, pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-09-14-fly-window-design.md`

## Global Constraints

- Training environment is 2D top-down.
- Agent state is position, heading, and speed.
- The fly receives only synthetic visual stimulus, not privileged target coordinates.
- Window/physics simplifications must be documented; no claim of exact fly aerodynamics or retina reconstruction.
- Environment reset/evaluation must be reproducible from fixed integer seeds.
- Success is crossing the room boundary through the open-window segment.
- All major completed checkpoints must be committed and pushed before starting the next major phase.

---

## File map

- Create `src/fly_window/env/config.py` — immutable room/dynamics/sensor/reward config.
- Create `src/fly_window/env/types.py` — `FlyState`, `StepResult`.
- Create `src/fly_window/env/geometry.py` — ray/window intersections and wall collision helpers.
- Create `src/fly_window/env/sensor.py` — 16-ray brightness retina.
- Create `src/fly_window/env/core.py` — reset/step state machine and reward.
- Create `src/fly_window/env/oracle.py` — non-training scripted controller.
- Create `scripts/smoke_env.py` — deterministic CLI sanity episode.
- Create `tests/env/test_geometry.py`, `test_sensor.py`, `test_core.py`, `test_oracle.py`.

### Task 1: Define immutable environment config and state types

**Files:**
- Create: `src/fly_window/env/__init__.py`
- Create: `src/fly_window/env/config.py`
- Create: `src/fly_window/env/types.py`
- Test: `tests/env/test_config.py`

**Interfaces:**
- Produces: `RoomConfig(width=10.0, height=6.0, window_center_y=3.0, window_width=1.5)`
- Produces: `DynamicsConfig(dt=0.05, max_speed=3.0, max_turn_rate=3.141592653589793, acceleration=2.0, drag=0.6)`
- Produces: `SensorConfig(ray_count=16, fov_radians=3.141592653589793, ambient=0.02)`
- Produces: `RewardConfig(progress_scale=0.5, collision_penalty=-0.1, step_penalty=-0.002, success_reward=5.0, max_steps=400)`
- Produces: `FlyState(x: float, y: float, heading: float, speed: float, step_count: int)`

- [ ] **Step 1: Write exact-default tests**

Assert the values above and validate window bounds are within the right wall and `ray_count >= 2`.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/env/test_config.py -v`
Expected: FAIL because config module does not exist.

- [ ] **Step 3: Implement frozen Pydantic models/dataclasses**

Use `pydantic.BaseModel` with `model_config = ConfigDict(frozen=True)` for configs and a frozen dataclass for state. Add validators rejecting nonpositive room dimensions, invalid window width, nonpositive `dt`, and nonpositive `max_steps`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/env/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add src/fly_window/env tests/env/test_config.py
git commit -m "feat: define fly-window environment config"
git push
```

### Task 2: Implement room geometry and open-window crossing

**Files:**
- Create: `src/fly_window/env/geometry.py`
- Test: `tests/env/test_geometry.py`

**Interfaces:**
- Produces: `window_interval(room: RoomConfig) -> tuple[float, float]`
- Produces: `ray_to_right_wall(x: float, y: float, angle: float, room: RoomConfig) -> tuple[float, float] | None`
- Produces: `crosses_window(prev_xy: tuple[float,float], next_xy: tuple[float,float], room: RoomConfig) -> bool`
- Produces: `clamp_collision(prev: FlyState, candidate: FlyState, room: RoomConfig) -> tuple[FlyState, bool]`

- [ ] **Step 1: Write boundary geometry tests**

Cover a ray that intersects the opening, a ray that hits solid right wall, crossing exactly through the opening, crossing beside it, and collisions with left/top/bottom walls. Define the window interval as `[2.25, 3.75]` for defaults.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/env/test_geometry.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement analytic intersections**

For ray direction `(cos(a), sin(a))`, intersect only when `cos(a) > 1e-9` using `t=(room.width-x)/cos(a)`; opening visibility is determined by the resulting `y_hit`. For collisions, clamp inside `[0,width]x[0,height]` except a valid right-wall window crossing; on collision return speed multiplied by `0.2`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/env/test_geometry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add src/fly_window/env/geometry.py tests/env/test_geometry.py
git commit -m "feat: implement room and window geometry"
git push
```

### Task 3: Implement the 16-ray synthetic retina

**Files:**
- Create: `src/fly_window/env/sensor.py`
- Test: `tests/env/test_sensor.py`

**Interfaces:**
- Produces: `observe_window(state: FlyState, room: RoomConfig, sensor: SensorConfig) -> np.ndarray` shape `(16,)`, dtype `float32`, range `[ambient, 1.0]`.

- [ ] **Step 1: Write visibility tests**

At `(5,3)` facing right, assert center rays are brighter than ambient. Facing left, all rays equal ambient. At `(9,0.5)` facing toward the opening, assert at least one ray is bright. Also assert deterministic exact output for repeated calls.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/env/test_sensor.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement ray brightness**

Ray angles are evenly spaced over the configured FOV centered on heading. A ray that hits the right-wall opening receives brightness `1 / (1 + 0.05 * distance**2)` clipped to `[ambient, 1]`; any wall hit outside the opening returns `ambient`. No target coordinates or distance are returned separately.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/env/test_sensor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add src/fly_window/env/sensor.py tests/env/test_sensor.py
git commit -m "feat: add coarse window retina"
git push
```

### Task 4: Implement reset, dynamics, reward, and termination

**Files:**
- Create: `src/fly_window/env/core.py`
- Test: `tests/env/test_core.py`

**Interfaces:**
- Produces: `WindowExitEnv(config: EnvironmentConfig)`
- Produces: `reset(seed: int) -> tuple[np.ndarray, FlyState]`
- Produces: `step(action: np.ndarray) -> StepResult`
- `action[0]` yaw is clipped to `[-1,1]`; `action[1]` thrust is clipped to `[0,1]`.
- `StepResult` fields: `observation`, `state`, `reward`, `terminated`, `truncated`, `success`, `collision`.

- [ ] **Step 1: Write reproducible reset and dynamics tests**

Assert same seed yields identical state/observation; different seeds usually differ. For reset sample `x ~ U(1.0, 8.5)`, `y ~ U(0.75, 5.25)`, heading `~ U(-pi, pi)`, speed `0`.

- [ ] **Step 2: Write reward tests**

For a hand-placed state, moving closer to window center must have greater reward than moving equally far away. A collision adds exactly `-0.1`; success adds exactly `+5.0`; every nonterminal step includes `-0.002`.

- [ ] **Step 3: Run and verify failure**

Run: `uv run pytest tests/env/test_core.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement dynamics**

Per step:

```python
heading += yaw * max_turn_rate * dt
speed += (thrust * acceleration - drag * speed) * dt
speed = clip(speed, 0, max_speed)
x += cos(heading) * speed * dt
y += sin(heading) * speed * dt
```

Reward progress uses Euclidean distance from fly position to window center `(room.width, room.window_center_y)` before and after the candidate move, but this distance is not exposed in observation.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/env/test_core.py -v`
Expected: PASS.

- [ ] **Step 6: Commit and push**

```bash
git add src/fly_window/env/core.py src/fly_window/env/types.py tests/env/test_core.py
git commit -m "feat: implement fly-window episode dynamics"
git push
```

### Task 5: Prove the environment is solvable with a non-training oracle

**Files:**
- Create: `src/fly_window/env/oracle.py`
- Create: `scripts/smoke_env.py`
- Test: `tests/env/test_oracle.py`

**Interfaces:**
- Produces: `oracle_action(state: FlyState, room: RoomConfig) -> np.ndarray`
- Oracle may read target coordinates and therefore must never be imported by neural/training modules.

- [ ] **Step 1: Write a 100-seed solvability test**

Run 100 seeds with the oracle for at most 400 steps and require at least 95 successes. This tests physics/task geometry, not learned behavior.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/env/test_oracle.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement proportional heading controller**

Compute desired angle to window center, wrap angular error to `[-pi, pi]`, set `yaw=clip(error / (pi/3), -1, 1)` and `thrust=1.0`. When within `1.0` room unit of the right wall, aim at the exact window center.

- [ ] **Step 4: Run all environment tests and smoke CLI**

Run: `uv run pytest tests/env -v && uv run python scripts/smoke_env.py --seed 7`
Expected: all tests PASS and CLI reports `success=true` with steps `<400`.

- [ ] **Step 5: Commit and push Checkpoint B**

```bash
git add src/fly_window/env scripts/smoke_env.py tests/env
git commit -m "test: verify fly-window environment solvability"
git push
```

## Plan acceptance gate

Checkpoint B is complete only when all environment tests pass and the oracle succeeds on at least 95/100 deterministic seeds. The training stack must not import or call the oracle. Push the checkpoint before starting neural integration.
