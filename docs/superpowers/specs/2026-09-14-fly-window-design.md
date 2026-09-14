# Fly Window — connectome-constrained drosophila window-exit experiment

Date: 2026-09-14
Status: Draft for user review
Repository: sphere-homotopy/flytegral
Proposed branch: spec/fly-window-design

## 1. Summary

Build a new experiment that uses a real subgraph extracted from the published MaleCNS drosophila connectome to control a simulated fly in a simple room with an open bright window. The agent should learn, across many episodes, to find and exit through the window. The project must generate assets suitable for an X post: a training montage, a final evaluation demo, and a hero still.

The project is intentionally honest rather than sensational. We will not claim a full biological simulation or a resurrected fly mind. We will claim a connectome-constrained agent: real connectome topology, simplified neural dynamics, simplified sensory encoding, and simplified flight physics.

## 2. Goals

1. Use real MaleCNS-derived neural wiring rather than a decorative or arbitrary network.
2. Train the connectome-constrained agent so performance improves over episodes.
3. Produce reproducible evaluation metrics and saved checkpoints.
4. Produce a polished short video for X that shows both training progress and final behavior.
5. Preserve enough provenance so public claims are auditable.

## 3. Non-goals

1. Full whole-brain biophysical simulation.
2. Exact reconstruction of drosophila retina or muscle biomechanics.
3. Claims about consciousness or exact biological behavior.
4. A high-fidelity 3D scientific simulator in the first release.

## 4. Public framing

Allowed framing:
- trained a simulated fly to exit through an open window using a real drosophila connectome
- connectome-constrained agent
- real MaleCNS wiring with simplified neural dynamics
- task-focused subgraph extracted from the full adult male fly CNS

Forbidden or misleading framing:
- resurrected a fly
- actual fly consciousness learned this
- exact biological simulation
- this is necessarily how a real fly brain would behave

## 5. High-level architecture

The project is split into four layers.

### 5.1 Data layer
- Fetch MaleCNS data and metadata from neuPrint / official MaleCNS sources.
- Save a reproducible local snapshot of the exact retrieved data.
- Build a task-specific subgraph from selected visual-input and descending/motor-output populations.
- Export a compact graph artifact usable by the simulator.

### 5.2 Simulation layer
- 2D top-down room environment.
- Agent state: position, heading, speed.
- A single bright open window acts as the exit target.
- Simplified retina/ray-based brightness sensor.
- Collision detection and success detection.
- Deterministic seeds for repeatable evaluation.

### 5.3 Neural/training layer
- Fixed real connectome topology.
- Simplified neural dynamics on top of the graph.
- A learned mapping from synthetic visual stimulus into designated input neurons and from designated output neurons into low-dimensional flight controls.
- Training by reward over episodes.
- Periodic checkpoints and metric logging.

### 5.4 Rendering layer
- Read saved trajectories and neural activations without altering simulation.
- Produce split-screen video:
  - room and trajectory on one side
  - canonical CNS visualization with activity on the other
- Produce raw clips, final montage, and hero still.

## 6. Data and provenance

### 6.1 Source data
The source of truth is MaleCNS v1.0 and associated neuPrint-accessible annotations/metadata where available.

Stored provenance must include:
- dataset name and version
- retrieval date
- source URLs / dataset identifiers
- graph extraction query or selection logic
- counts of full-graph and subgraph neurons/synapses
- any pruning filters applied

### 6.2 Subgraph extraction
We will not simulate the entire CNS initially. We will extract a task-focused subgraph:
1. choose designated visual or visually relevant input populations
2. choose descending or motor-related output populations
3. include nodes on relevant paths between them up to configured depth / relevance thresholds
4. prune weak or irrelevant regions if needed

The extraction process must be encoded in code and recorded in provenance output.

## 7. Environment design

### 7.1 Room
A simple rectangular room with opaque walls and one open bright window. The environment should be intentionally sparse to emphasize the connectome-driven behavior.

### 7.2 State and actions
State:
- x position
- y position
- heading
- speed

Decoded actions:
- turn left/right (yaw)
- forward drive / thrust
- optional stabilization scalar if needed

### 7.3 Sensory input
The simulated fly receives a synthetic visual stimulus, likely a low-resolution brightness vector from a coarse forward-facing retinal layout. The mapping from this stimulus to designated connectome input neurons is explicit, documented, and potentially partially trainable.

### 7.4 Success and failure
Success: the fly enters the window exit region.
Failure cases: timeout, repeated collision, or persistent stagnation.

## 8. Neural model

### 8.1 Representation
Use the real extracted topology as a directed weighted graph. Initial edge weights come from connectome-derived connectivity strength or a normalized proxy.

### 8.2 Dynamics
First release will use simplified neural dynamics rather than full membrane simulation. A rate-based or rate/spike hybrid is preferred for tractability and speed.

Requirements:
- deterministic forward pass given state and seed
- batched execution if feasible for training speed
- ability to expose per-neuron activity for visualization

### 8.3 What may learn
Topology remains fixed.
Learning is allowed in a constrained way:
- input encoding parameters
- output decoding parameters
- possibly a limited subset or transformation of internal weights

The chosen trainable parameter set must be explicit and small enough to preserve the meaning of “connectome-constrained”.

## 9. Reward and training

### 9.1 Reward design
Recommended reward structure:
- positive shaping reward for reducing distance to the window
- terminal reward for exiting
- penalty for collisions
- penalty for stagnation or excessive episode length

### 9.2 Training loop
Each run stores:
- config snapshot
- random seed
- periodic checkpoints
- episode reward
- success/failure
- path length and duration
- evaluation summaries

### 9.3 Evaluation isolation
Training and evaluation must be separate modes. Final demo clips come from frozen evaluation checkpoints, not cherry-picked training rollouts.

## 10. Controls and success criteria

### 10.1 Baselines / controls
At minimum compare against:
1. untrained connectome-constrained model
2. trained model
3. shuffled-topology or topology-scrambled control with similar gross statistics

### 10.2 Success criteria for v1
A release is successful when a frozen checkpoint satisfies all of the following:
- evaluated on at least 100 episodes across a fixed held-out seed set
- at least 80% successful exits
- clearly better than untrained baseline
- clearly better than shuffled-topology control
- learning curve shows improvement over training

## 11. Visual output and X assets

### 11.1 Core deliverables
1. training montage video
2. final evaluation demo video
3. hero still image
4. metrics snapshot suitable for a thread

### 11.2 Video structure
Target duration: 25–40 seconds.
Suggested sequence:
1. hook: “Can a real fruit-fly connectome learn to fly out of an open window?”
2. early failed episode
3. montage of later training episodes with improving behavior
4. frozen-evaluation successes from new starts
5. closing factual summary with caveats

### 11.3 Visualization
Preferred split-screen:
- left: room, fly, path, and status annotations
- right: canonical CNS rendering using real neuron coordinates / structure, with activity pulses
- overlays: episode number, reward, running success rate, and concise methodological labels

## 12. Repository structure

```text
flytegral/
  docs/
    superpowers/
      specs/
        2026-09-14-fly-window-design.md
  data/
  scripts/
    fetch_malecns.py
    build_subgraph.py
  src/
    connectome/
    neural/
    environment/
    training/
    evaluation/
    render/
  configs/
  artifacts/
    graphs/
    checkpoints/
    runs/
    videos/
  tests/
  PROVENANCE.md
  README.md
```

Note: if a separate repository is created later, the structure moves intact.

## 13. Checkpoints / phases

Checkpoint A:
- MaleCNS data fetched
- subgraph extracted
- counts and provenance verified

Checkpoint B:
- room environment complete
- manual/scripted controller can exit via the window

Checkpoint C:
- connectome activity drives agent control end-to-end

Checkpoint D:
- training improves success rate over baseline

Checkpoint E:
- frozen evaluation complete
- video assets rendered

Each checkpoint should be committed and pushed before the next major phase.

## 14. Testing strategy

Automated tests should cover:
- graph loading and subgraph extraction determinism
- environment reset/step correctness
- reward calculation
- checkpoint save/load
- evaluation reproducibility for fixed seeds
- rendering pipeline smoke tests

Manual verification:
- visualize a sample subgraph
- inspect a handful of trajectories
- verify that final video annotations match logged metrics

## 15. Risks and mitigations

1. **Subgraph selection is poor**
   - mitigation: begin with a simple but auditable heuristic and keep extraction configurable

2. **Training is unstable or too slow**
   - mitigation: start with small environments and simplified dynamics; only add complexity after a baseline works

3. **Connectome contribution is not obvious**
   - mitigation: include shuffled-topology controls and clear provenance overlays

4. **Visual output becomes detached from actual metrics**
   - mitigation: generate video from logged evaluation runs and embed run IDs/seeds in artifacts

5. **Repository choice may change**
   - mitigation: keep the initial spec and structure portable so it can be moved to a new repository if needed

## 16. Open implementation decisions already narrowed

The following decisions are considered settled for implementation planning:
- use a task-focused MaleCNS-derived subgraph rather than the whole CNS
- use a 2D top-down simulation for training
- use a pseudo-3D style only in the final renderer if helpful
- separate training and frozen evaluation
- optimize for both honesty and X-appropriate presentation

## 17. Immediate next step

After user review of this design doc, create a concrete implementation plan covering:
1. repository bootstrap
2. data acquisition and provenance pipeline
3. environment implementation
4. neural model implementation
5. training/evaluation loop
6. rendering/video generation
7. milestone-by-milestone commit plan
