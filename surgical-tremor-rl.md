# SKILL.md — Surgical Robot Tremor Compensator RL

## Project Identity
- *Repo:* surgical-robot-tremor-compensator-rl
- *Goal:* Train an RL agent (SAC) to distinguish intentional surgeon movement
  from involuntary hand tremor and compensate in real-time
- *Stack:*
  - **Backend/RL:** Python 3.11+, Gymnasium, Stable-Baselines3, PyTorch, NumPy,
    SciPy, PyBullet (physics sim), Matplotlib (offline plots)
  - **API:** FastAPI, Uvicorn (human feedback + training control + WebSocket)
  - **Frontend:** React (Vite + TypeScript), Recharts (charting)
  - **Tooling:** uv (package manager), Ruff (linter), Pytest (testing)

---

## Claude CLI Behavior Rules

When helping with this project, Claude must always:

1. *Write production-quality Python* with type hints, docstrings, and clear
   separation of concerns across these folders:
   
   surgical-robot-tremor-compensator-rl/
   ├── env/
   │   ├── surgical_env.py        # Custom Gymnasium environment
   │   ├── tremor_generator.py    # Simulates hand tremor signals
   │   └── physics_sim.py         # PyBullet robot arm simulation
   ├── agents/
   │   ├── sac_agent.py           # SAC implementation
   │   └── reward_model.py        # Human-in-the-loop reward model
   ├── safety/
   │   └── constraints.py         # Hard safety constraints layer
   ├── dashboard/
   │   └── visualizer.py          # Real-time compensation dashboard
   ├── utils/
   │   ├── signal_processing.py   # FFT-based tremor frequency analysis
   │   └── logger.py              # Training metrics logger
   ├── train.py                   # Main training entrypoint
   ├── evaluate.py                # Evaluation + human feedback collection
   └── config.yaml                # All hyperparameters in one place
   

2. *Never hardcode hyperparameters* — always pull from config.yaml

3. *Always include unit tests* for environment step logic and reward functions

4. *Comment every reward function line* explaining the clinical rationale
   behind each penalty/bonus

---

## Environment Design (SurgicalEnv)

### State Space (what the agent observes)
python
state = {
    "robot_tip_position":     np.ndarray,  # shape (3,) — x, y, z in mm
    "robot_tip_velocity":     np.ndarray,  # shape (3,) — velocity vector
    "surgeon_input_raw":      np.ndarray,  # shape (3,) — raw hand signal
    "surgeon_input_filtered": np.ndarray,  # shape (3,) — low-pass filtered
    "tremor_frequency_band":  np.float32,  # dominant tremor Hz (3-12 Hz range)
    "tissue_proximity":       np.float32,  # distance to tissue boundary in mm
    "time_in_episode":        np.float32,  # normalized 0-1
}


### Action Space (what the agent controls)
python
# Continuous action space — compensation offset applied to surgeon input
action = np.ndarray  # shape (3,) — delta x, y, z correction
# Clipped to [-MAX_CORRECTION_MM, +MAX_CORRECTION_MM] per axis


### Reward Function (clinically grounded)
python
def compute_reward(state, action, next_state):
    """
    Clinical rationale for each term:

    1. tracking_accuracy: Reward for matching surgeon's TRUE intended
       trajectory (low-pass filtered signal), not raw signal.
       Motivates tremor removal without over-correction.

    2. smoothness_penalty: Penalize jerky compensation — sudden robot
       movements are dangerous near tissue.

    3. tissue_safety_penalty: Hard penalty if robot tip enters safety
       exclusion zone around tissue boundary. Simulates perforation risk.

    4. latency_penalty: Penalize compensation that arrives too late
       (>20ms delay). Real surgery requires sub-20ms response.

    5. human_feedback_bonus: Sparse bonus from human evaluator rating
       the compensation quality (collected via FastAPI endpoint).
    """
    r_tracking   = -np.linalg.norm(next_state["robot_tip_position"]
                                   - next_state["surgeon_input_filtered"])
    r_smooth     = -0.1 * np.linalg.norm(action - prev_action)
    r_safety     = -100.0 if next_state["tissue_proximity"] < SAFETY_MARGIN_MM else 0.0
    r_latency    = -0.05 * max(0, compensation_delay_ms - 20)
    r_human      = human_feedback_signal  # 0.0 most steps, sparse +/- signal

    return r_tracking + r_smooth + r_safety + r_latency + r_human


---

## Tremor Signal Generator

Claude must implement tremor using a *physiologically accurate model*:

python
# Pathological tremor ranges (from medical literature):
# Essential tremor:   4–8 Hz
# Parkinson's tremor: 3–6 Hz
# Physiological:      8–12 Hz

def generate_tremor(t, tremor_type="essential"):
    frequencies = TREMOR_PROFILES[tremor_type]["frequencies"]
    amplitudes  = TREMOR_PROFILES[tremor_type]["amplitudes"]  # in mm
    signal = sum(
        amp * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
        for freq, amp in zip(frequencies, amplitudes)
    )
    return signal  # Add to surgeon's true intended trajectory


---

## SAC Agent Guidelines

- Use *Stable-Baselines3 SAC* as baseline, then implement custom SAC
  from scratch in Phase 3 to deepen understanding
- Key SAC hyperparameters to tune for this environment:
  yaml
  sac:
    learning_rate: 3e-4
    buffer_size: 1_000_000
    batch_size: 256
    tau: 0.005               # soft update coefficient
    gamma: 0.99
    ent_coef: "auto"         # automatic entropy tuning — critical for SAC
    target_entropy: "auto"
    train_freq: 1
    gradient_steps: 1
  
- Always log: entropy, actor_loss, critic_loss, ent_coef per step

---

## Safety Constraint Layer

Claude must implement a *hard constraint wrapper* around the base env:

python
class SafetySurgicalEnv(gymnasium.Wrapper):
    """
    Intercepts actions before execution.
    Clips any action that would move robot tip into tissue exclusion zone.
    This is non-negotiable — agent cannot learn to violate tissue safety.
    """
    def step(self, action):
        safe_action = self.project_to_safe_action(action)
        return self.env.step(safe_action)

    def project_to_safe_action(self, action):
        # Project action onto safe manifold using tissue proximity gradient
        ...


---

## Human-in-the-Loop Reward Pipeline


Training Loop
     │
     ▼
Every N episodes → Save trajectory video/plot
     │
     ▼
FastAPI endpoint → Human evaluator scores compensation (1-5 scale)
     │
     ▼
RewardModel (small neural net) trained on human scores
     │
     ▼
Sparse r_human injected back into SAC replay buffer


- Store human feedback in feedback/human_labels.jsonl
- Retrain reward model every 50 human labels
- This teaches the RL pipeline used in modern RLHF systems

---

## Key RL Concepts This Project Teaches

| Concept | Where It Appears |
|---|---|
| Continuous action spaces | SAC on 3D correction vector |
| Entropy maximization | SAC's automatic ent_coef tuning |
| Reward shaping | 5-component clinically-grounded reward |
| Safety constraints | Hard action projection layer |
| Sparse rewards | Human feedback signal |
| Human-in-the-loop RL | FastAPI feedback → reward model |
| Sim-to-real gap | Tremor model vs real hand signals |
| Signal processing in RL | FFT for tremor frequency detection |

---

## How to Start Each Claude CLI Session

Paste this at the beginning of every session:


Read the skill at ~/.claude/skills/surgical-tremor-rl.md

We are building surgical-robot-tremor-compensator-rl.
Today's session goal: [DESCRIBE YOUR PHASE/TASK HERE]

Always follow the folder structure, reward function design,
and safety constraint rules defined in the skill file.


---

## Evaluation Metrics (track these during training)

- compensation_error_mm — RMS error between compensated and true trajectory
- tremor_rejection_ratio — how much tremor power was removed (dB)
- safety_violations — count of tissue boundary breaches (target: 0)
- human_feedback_score — average human rating per evaluation batch
- compensation_latency_ms — must stay below 20ms threshold

---

## Common Pitfalls to Avoid

1. *Reward hacking:* Agent learns to keep robot perfectly still (zero tracking
   error on filtered signal but ignores surgeon intent entirely). Fix: add
   correlation reward between robot movement and surgeon intended movement.

2. *Over-smoothing:* Agent over-penalizes its own movement and becomes
   unresponsive. Fix: balance smoothness_penalty coefficient carefully.

3. *Safety layer killing exploration:* Hard clipping reduces action diversity
   early in training. Fix: use soft penalty first, hard constraint after
   50k steps.

4. *Human feedback noise:* Inconsistent human ratings destabilize reward
   model. Fix: collect inter-rater reliability scores, filter outliers.

---

## Phased Development Plan

### Phase 1 — Core Environment & Tremor Simulation
- Scaffold full project folder structure
- Implement `tremor_generator.py` with all 3 tremor profiles
- Implement `surgical_env.py` (Gymnasium env with state/action spaces)
- Implement `signal_processing.py` (FFT-based tremor frequency detection)
- Write `config.yaml` with all hyperparameters
- Unit tests for env step logic and tremor generator
- **Exit criteria:** `env.step()` runs, tremor signals match expected frequency bands

### Phase 2 — SAC Training with SB3
- Implement `sac_agent.py` using Stable-Baselines3 SAC
- Implement `reward_model.py` (initially returns 0 for human feedback)
- Implement `constraints.py` (SafetySurgicalEnv wrapper)
- Implement `logger.py` for training metrics
- Write `train.py` and `evaluate.py`
- Unit tests for reward function and safety constraints
- **Exit criteria:** Agent trains for 100k steps, compensation_error_mm decreases

### Phase 3 — Custom SAC from Scratch
- Reimplement SAC (actor-critic, entropy tuning, replay buffer) in pure PyTorch
- Side-by-side comparison with SB3 baseline
- Log entropy, actor_loss, critic_loss, ent_coef per step
- **Exit criteria:** Custom SAC matches SB3 performance within 10%

### Phase 4 — PyBullet Physics Simulation
- Implement `physics_sim.py` with a robot arm model (e.g., 6-DOF manipulator)
- Connect PyBullet simulation to `surgical_env.py`
- Add realistic tissue collision detection for safety layer
- **Exit criteria:** Agent trains with physics sim, tissue proximity computed from collision mesh

### Phase 5 — Human-in-the-Loop Feedback Pipeline
- Implement FastAPI backend endpoints for human evaluation
- Build trajectory visualization for human reviewers
- Store feedback in `feedback/human_labels.jsonl`
- Train `RewardModel` neural net on human scores
- Inject sparse `r_human` into SAC replay buffer
- **Exit criteria:** Reward model trained on 50+ human labels, r_human affects agent behavior

### Phase 6 — Frontend Dashboard (React + Vite + TypeScript)
- Scaffold React app with `npm create vite@latest frontend -- --template react-ts`
- Install Recharts for charting, configure WebSocket client hook
- Configure FastAPI `CORSMiddleware` for dev (localhost:5173 → localhost:8000)
- Implement 5 dashboard components:
  - `TrajectoryPlot.tsx` — 3-axis real-time trajectory overlay
  - `FrequencySpectrum.tsx` — live FFT tremor spectrum
  - `RewardBreakdown.tsx` — stacked reward component chart
  - `SafetyZone.tsx` — 2D tissue proximity visualization
  - `TrainingMetrics.tsx` — episode return with rolling average
- Implement `useWebSocket.ts` hook consuming `/ws/metrics` endpoint
- Add Vite proxy config for API requests during development
- **Exit criteria:** Dashboard renders all 5 panels with live data during training

### Phase 7 — Evaluation & Polishing
- Run full evaluation suite across all tremor types
- Generate final metrics report (all 5 evaluation metrics)
- Documentation and README update
- **Exit criteria:** All metrics within target thresholds, zero safety violations

---

## Dependencies & Project Management

**Package/project manager:** [uv](https://docs.astral.sh/uv/)

```toml
# pyproject.toml
[project]
name = "surgical-robot-tremor-compensator-rl"
version = "0.1.0"
description = "RL agent (SAC) for real-time surgical robot tremor compensation"
requires-python = ">=3.11"
dependencies = [
    "gymnasium>=0.29.0",
    "stable-baselines3>=2.1.0",
    "numpy>=1.24.0",
    "pybullet>=3.2.5",
    "matplotlib>=3.7.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pyyaml>=6.0",
    "torch>=2.0.0",
    "scipy>=1.10.0",
]

[dependency-groups]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### uv Commands Reference
```bash
uv init                    # Initialize project (already done via pyproject.toml)
uv sync                    # Install all dependencies + dev group
uv run train.py            # Run scripts within the managed environment
uv run pytest              # Run tests
uv add <package>           # Add a new dependency
uv add --group dev <pkg>   # Add a dev-only dependency
uv lock                    # Regenerate uv.lock after manual pyproject.toml edits
```

---

## Frontend Dashboard (dashboard/visualizer.py)

### Purpose
Real-time visualization of tremor compensation during training and evaluation.
Uses Matplotlib for live plotting (Phase 6 may upgrade to a web-based dashboard).

### Dashboard Panels

1. **Trajectory Comparison Plot**
   - Overlay of 3 signals: raw surgeon input, filtered (intended) trajectory,
     and compensated robot tip position
   - X-axis: time (ms), Y-axis: position (mm) — one subplot per axis (x, y, z)

2. **Tremor Frequency Spectrum**
   - Live FFT of raw surgeon input showing tremor frequency peaks
   - Highlight the detected dominant tremor band
   - Overlay FFT of compensated signal to show tremor rejection

3. **Reward Components Breakdown**
   - Stacked bar or line chart of all 5 reward terms per step
   - Helps diagnose which reward term dominates training

4. **Safety Zone Visualization**
   - 2D/3D plot of robot tip position relative to tissue boundary
   - Color-coded: green (safe), yellow (approaching boundary), red (violation)

5. **Training Metrics Over Time**
   - Episode return, compensation_error_mm, tremor_rejection_ratio
   - Rolling average with configurable window

### Web Dashboard Extension (Phase 6+)
- Optional upgrade from Matplotlib to a browser-based dashboard
- Stack: FastAPI (backend already exists) + React (Vite + TypeScript)
- WebSocket stream of live metrics from training loop
- Endpoints:
  ```
  GET  /dashboard              → Serve dashboard page
  WS   /ws/metrics             → Stream live training metrics
  GET  /api/episodes/{id}      → Fetch trajectory data for replay
  GET  /api/metrics/summary    → Aggregated metrics for current run
  ```

---

## FastAPI Backend (evaluate.py endpoints)

### Human Feedback Endpoints

```
POST /feedback/evaluate
  Body: { "episode_id": int, "score": 1-5, "evaluator_id": str }
  → Stores rating in feedback/human_labels.jsonl

GET  /feedback/trajectory/{episode_id}
  → Returns trajectory plot image or data for human review

GET  /feedback/stats
  → Returns inter-rater reliability, label count, score distribution

POST /feedback/retrain-reward-model
  → Triggers reward model retraining on collected labels
```

### Training Control Endpoints

```
POST /training/start
  Body: { "config_overrides": dict }
  → Starts training with optional config overrides

POST /training/stop
  → Gracefully stops training, saves checkpoint

GET  /training/status
  → Returns current episode, step, metrics snapshot
```

---

## Safety Constraint Modes

To reconcile the hard constraint wrapper with Pitfall #3 (safety killing
exploration), the `SafetySurgicalEnv` supports two modes controlled via
`config.yaml`:

```yaml
safety:
  mode: "adaptive"          # "hard", "soft", or "adaptive"
  safety_margin_mm: 2.0     # tissue exclusion zone radius
  soft_penalty_weight: 10.0 # penalty coefficient in soft mode
  hard_threshold_steps: 50000  # switch from soft → hard in adaptive mode
```

- **hard:** Always projects actions onto safe manifold (original design)
- **soft:** Applies penalty to reward but allows boundary violations (better
  early exploration)
- **adaptive:** Starts with soft penalties, switches to hard constraints after
  `hard_threshold_steps` (recommended)

---

## Testing Strategy

### Unit Tests (`tests/`)
```
tests/
├── test_tremor_generator.py   # Verify frequency bands, amplitude ranges
├── test_surgical_env.py       # Step logic, observation shapes, reset behavior
├── test_reward_function.py    # Each reward component in isolation
├── test_safety_constraints.py # Action projection, mode switching
├── test_signal_processing.py  # FFT accuracy, dominant frequency detection
└── test_reward_model.py       # Forward pass, training on mock labels
```

### Key Test Cases
- Tremor signal FFT peaks fall within expected Hz bands
- Env observation and action spaces match spec shapes
- Safety wrapper clips actions that would breach tissue boundary
- Reward function returns expected values for known state transitions
- Soft → hard mode transition happens at correct step count
- Human feedback JSONL is written and read correctly

---

## Folder Structure (updated)

```
surgical-robot-tremor-compensator-rl/
├── env/
│   ├── __init__.py
│   ├── surgical_env.py        # Custom Gymnasium environment
│   ├── tremor_generator.py    # Simulates hand tremor signals
│   └── physics_sim.py         # PyBullet robot arm simulation
├── agents/
│   ├── __init__.py
│   ├── sac_agent.py           # SAC implementation (SB3 + custom)
│   └── reward_model.py        # Human-in-the-loop reward model
├── safety/
│   ├── __init__.py
│   └── constraints.py         # Hard/soft/adaptive safety constraints
├── dashboard/
│   ├── __init__.py
│   └── visualizer.py          # Real-time compensation dashboard
├── utils/
│   ├── __init__.py
│   ├── signal_processing.py   # FFT-based tremor frequency analysis
│   └── logger.py              # Training metrics logger
├── tests/
│   ├── test_tremor_generator.py
│   ├── test_surgical_env.py
│   ├── test_reward_function.py
│   ├── test_safety_constraints.py
│   ├── test_signal_processing.py
│   └── test_reward_model.py
├── feedback/                  # Created at runtime
│   └── human_labels.jsonl
├── docs/                      # Project documentation (not tracked in git)
│   ├── architecture.md        # System design, data flow diagrams
│   ├── api_reference.md       # FastAPI endpoint documentation
│   ├── config_guide.md        # Explanation of every config.yaml field
│   ├── tremor_model.md        # Clinical references, frequency band rationale
│   └── deployment.md          # How to run training, dashboard, evaluation
├── frontend/                  # React dashboard (Vite + TypeScript)
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── components/
│       │   ├── TrajectoryPlot.tsx      # 3-axis trajectory overlay (Recharts)
│       │   ├── FrequencySpectrum.tsx   # Live FFT tremor spectrum
│       │   ├── RewardBreakdown.tsx     # Stacked reward components
│       │   ├── SafetyZone.tsx          # 2D/3D tissue proximity viz
│       │   └── TrainingMetrics.tsx     # Episode return rolling avg
│       ├── hooks/
│       │   └── useWebSocket.ts        # WebSocket client for live metrics
│       └── types/
│           └── metrics.ts             # Shared TypeScript interfaces
├── checkpoints/               # Saved model weights
├── logs/                      # Training logs and TensorBoard events
├── train.py                   # Main training entrypoint
├── evaluate.py                # Evaluation + FastAPI feedback server
├── config.yaml                # All hyperparameters in one place
├── pyproject.toml             # Project config & dependencies (uv)
├── uv.lock                    # Locked dependency versions
└── README.md
```

---

## Full config.yaml Specification

```yaml
# ── Environment ──────────────────────────────────────────────
environment:
  simulation_timestep_ms: 5.0     # dt = 5ms → 200 Hz control loop
  episode_length_steps: 2000      # 2000 steps × 5ms = 10 seconds per episode
  max_correction_mm: 2.0          # per-axis action clamp
  tissue_boundary_position:       # static tissue target (mm)
    x: 0.0
    y: 0.0
    z: 50.0

  termination:
    on_tissue_perforation: true   # end episode if tissue breached (hard mode)
    on_max_steps: true            # end episode at episode_length_steps
    max_consecutive_violations: 3 # end if 3 consecutive safety violations (soft mode)

# ── Tremor Generator ────────────────────────────────────────
tremor:
  default_type: "essential"
  profiles:
    essential:
      frequencies: [4.0, 5.5, 7.0]    # Hz
      amplitudes:  [0.15, 0.10, 0.05]  # mm
    parkinsons:
      frequencies: [3.5, 4.5, 5.5]
      amplitudes:  [0.20, 0.12, 0.06]
    physiological:
      frequencies: [8.0, 9.5, 11.0]
      amplitudes:  [0.05, 0.03, 0.02]

# ── SAC Hyperparameters ─────────────────────────────────────
sac:
  learning_rate: 3e-4
  buffer_size: 1_000_000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  ent_coef: "auto"
  target_entropy: "auto"
  train_freq: 1
  gradient_steps: 1

# ── Reward Weights ──────────────────────────────────────────
reward:
  tracking_weight: 1.0
  smoothness_weight: 0.1
  safety_penalty: -100.0
  latency_weight: 0.05
  latency_threshold_ms: 20.0
  human_feedback_weight: 1.0

# ── Safety ──────────────────────────────────────────────────
safety:
  mode: "adaptive"
  safety_margin_mm: 2.0
  soft_penalty_weight: 10.0
  hard_threshold_steps: 50000

# ── Checkpointing ───────────────────────────────────────────
checkpointing:
  save_freq_episodes: 100         # save every N episodes
  keep_last_n: 5                  # rolling window of recent checkpoints
  save_best: true                 # always keep best model
  best_metric: "compensation_error_mm"  # lower is better
  best_mode: "min"
  checkpoint_dir: "checkpoints/"
  naming: "{timestamp}_{episode}_{metric:.4f}"  # e.g. 20260306_ep500_0.1234

# ── Logging ─────────────────────────────────────────────────
logging:
  backend: "tensorboard"          # "tensorboard" or "csv"
  log_dir: "logs/"
  log_freq_steps: 100             # log scalar metrics every N steps
  log_histograms: false           # weight/gradient histograms (slow)

# ── Evaluation ──────────────────────────────────────────────
evaluation:
  eval_freq_episodes: 50          # run evaluation every N training episodes
  eval_episodes: 10               # number of episodes per evaluation
  save_trajectory_plots: true
  human_feedback_freq: 100        # prompt human review every N episodes

# ── Reproducibility ─────────────────────────────────────────
seed: 42                          # master seed for all RNGs
# Derived seeds: env=seed, torch=seed+1, numpy=seed+2, tremor=seed+3
```

---

## Simulation Timing & Latency Model

The environment runs at a **200 Hz control loop** (`dt = 5ms`):
- Each `env.step()` call advances simulation by 5ms
- Episode = 2000 steps = 10 seconds of simulated surgery
- Compensation latency is measured as the number of steps between
  tremor onset and compensating action × dt
- The latency penalty activates when `compensation_delay_ms > 20ms`
  (i.e., the agent takes more than 4 steps to respond)

---

## Episode Termination Conditions

An episode ends when any of these conditions are met:
1. **Max steps reached** — `episode_length_steps` (default 2000)
2. **Tissue perforation** — robot tip crosses tissue boundary (hard/adaptive mode only)
3. **Consecutive violations** — 3+ consecutive safety margin breaches (soft mode)

The `terminated` flag distinguishes safety termination (bad) from truncation
(max steps). This matters for SAC's value bootstrapping.

---

## Observation Space: prev_action

The smoothness penalty requires `prev_action`. This is included in the
observation by stacking it into the state vector:

```python
state = {
    "robot_tip_position":     np.ndarray,  # shape (3,)
    "robot_tip_velocity":     np.ndarray,  # shape (3,)
    "surgeon_input_raw":      np.ndarray,  # shape (3,)
    "surgeon_input_filtered": np.ndarray,  # shape (3,)
    "tremor_frequency_band":  np.float32,  # dominant Hz
    "tissue_proximity":       np.float32,  # mm
    "time_in_episode":        np.float32,  # 0–1
    "prev_action":            np.ndarray,  # shape (3,) — last action taken
}
# Total flat observation: 3+3+3+3+1+1+1+3 = 18 dimensions
```

---

## Tremor Generator: Phase Coherence Fix

The tremor phase must be sampled **once per episode** at reset, not per call:

```python
class TremorGenerator:
    def reset(self, rng: np.random.Generator):
        """Sample random phases once per episode for coherent tremor."""
        self.phases = rng.uniform(0, 2 * np.pi, size=len(self.frequencies))

    def generate(self, t: float) -> np.ndarray:
        """Generate tremor signal with stable per-episode phases."""
        return sum(
            amp * np.sin(2 * np.pi * freq * t + phase)
            for freq, amp, phase in zip(self.frequencies, self.amplitudes, self.phases)
        )
```

---

## Reproducibility & Seeding

All random number generators are seeded from a single master seed in `config.yaml`:

```python
def seed_everything(master_seed: int):
    np.random.seed(master_seed)
    torch.manual_seed(master_seed + 1)
    env.reset(seed=master_seed + 2)
    tremor_generator.reset(np.random.default_rng(master_seed + 3))
```

- Master seed is logged with every training run for exact reproduction
- Evaluation runs use a separate fixed seed (master_seed + 1000)

---

## Logging Backend

**Primary:** TensorBoard (via `torch.utils.tensorboard.SummaryWriter`)

Logged scalars per step (every `log_freq_steps`):
- `reward/total`, `reward/tracking`, `reward/smooth`, `reward/safety`,
  `reward/latency`, `reward/human`
- `sac/entropy`, `sac/actor_loss`, `sac/critic_loss`, `sac/ent_coef`
- `env/tissue_proximity_min`, `env/compensation_delay_ms`

Logged per episode:
- `eval/compensation_error_mm`, `eval/tremor_rejection_ratio_dB`
- `eval/safety_violations`, `eval/human_feedback_score`

Launch: `tensorboard --logdir logs/`

---

## Checkpointing Strategy

```
checkpoints/
├── best_model.pt                        # best compensation_error_mm ever
├── 20260306_120000_ep500_0.1234.pt      # periodic checkpoint
├── 20260306_123000_ep600_0.1100.pt
└── ...                                  # keep last 5, delete older
```

Each checkpoint contains:
- SAC model weights (actor, critic, target critic)
- Optimizer states
- Replay buffer (optional, large — controlled by `save_buffer: false`)
- `ent_coef` value
- Training step count and episode count
- Config snapshot (for reproducibility)

Resume training: `uv run train.py --resume checkpoints/best_model.pt`

---

## CORS Configuration (FastAPI ↔ React)

```python
# In evaluate.py or a shared server.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

In production, replace with the actual deployment origin.

---

## Documentation (docs/)

The `docs/` directory contains detailed project documentation written for
developers, researchers, and clinical collaborators. These files are not
tracked in version control because they are generated and maintained locally
as living documents that evolve alongside the codebase.

### docs/architecture.md — System Architecture

Describes the overall system design and how each component connects to form
the complete tremor compensation pipeline.

**Contents:**

- **System Overview:** A high-level description of the project's purpose —
  training a reinforcement learning agent to filter involuntary hand tremor
  from a surgeon's input signal in real time, producing a stable compensated
  trajectory for the robot arm.

- **Component Diagram:** A visual representation of the five major subsystems
  and how data flows between them:
  1. **Tremor Generator** produces a simulated tremor signal and adds it to
     the surgeon's intended trajectory.
  2. **Surgical Environment** receives the combined signal, applies the agent's
     compensation action, advances the physics simulation, and returns the
     next observation and reward.
  3. **SAC Agent** observes the environment state, selects a compensation
     action, and updates its policy based on the reward signal.
  4. **Safety Constraint Layer** intercepts every action before it reaches the
     environment and projects it onto a safe manifold to prevent tissue
     boundary violations.
  5. **Human Feedback Pipeline** periodically collects expert evaluations of
     the agent's compensation quality, trains a reward model on those scores,
     and injects a sparse reward signal back into the replay buffer.

- **Data Flow:** A step-by-step walkthrough of a single training iteration,
  from raw surgeon input through tremor injection, observation construction,
  action selection, safety projection, environment step, reward computation,
  and replay buffer storage.

- **React Dashboard Integration:** How the FastAPI backend streams live
  training metrics over a WebSocket connection to the React frontend, and
  how the frontend renders each of the five dashboard panels.

### docs/api_reference.md — FastAPI Endpoint Documentation

A complete reference for every HTTP and WebSocket endpoint exposed by the
FastAPI server.

**Contents:**

- **Human Feedback Endpoints:**
  - `POST /feedback/evaluate` — Accepts an episode identifier, a score from
    one to five, and an evaluator identifier. Appends the rating as a JSON
    line to `feedback/human_labels.jsonl` and returns a confirmation with the
    current total label count.
  - `GET /feedback/trajectory/{episode_id}` — Returns the trajectory data for
    a given episode so that a human evaluator can review the agent's
    compensation performance before scoring it. The response includes the raw
    surgeon input, the filtered intended trajectory, and the compensated robot
    tip positions over time.
  - `GET /feedback/stats` — Returns aggregate statistics about the collected
    human feedback, including the total number of labels, the score
    distribution, and an inter-rater reliability coefficient if multiple
    evaluators have contributed.
  - `POST /feedback/retrain-reward-model` — Triggers a retraining run of the
    reward model neural network on all collected human labels. Returns the
    updated model's training loss and validation accuracy.

- **Training Control Endpoints:**
  - `POST /training/start` — Begins a training run. Accepts an optional
    dictionary of configuration overrides that are merged on top of the
    values in `config.yaml`. Returns a run identifier.
  - `POST /training/stop` — Gracefully stops the current training run, saves
    a checkpoint, and returns the final training step count and metrics.
  - `GET /training/status` — Returns a snapshot of the current training
    state, including the episode number, total step count, and the most
    recent values of all logged metrics.

- **Dashboard Endpoints:**
  - `GET /dashboard` — Serves the React frontend's production build as
    static files.
  - `WS /ws/metrics` — A WebSocket endpoint that streams live training
    metrics to the React dashboard. Each message is a JSON object containing
    the current step, episode, and all logged scalar values.
  - `GET /api/episodes/{id}` — Returns the full trajectory and reward
    breakdown data for a specific episode, used by the dashboard to render
    historical episode replays.
  - `GET /api/metrics/summary` — Returns aggregated metrics for the current
    training run, including rolling averages and best-so-far values.

- **Request and Response Schemas:** Typed definitions for every request body
  and response payload, with example values.

### docs/config_guide.md — Configuration Reference

A field-by-field explanation of every parameter in `config.yaml`, organised
by section.

**Contents:**

- **Environment Section:** Explains the simulation timestep and why 5ms was
  chosen (200 Hz matches the control frequency of real surgical robots),
  the episode length and its relationship to simulated surgery duration,
  the maximum correction magnitude and why it is clamped to prevent
  over-correction, the tissue boundary position, and the three termination
  conditions with their clinical rationale.

- **Tremor Section:** Documents each tremor profile (essential, Parkinson's,
  physiological), the frequency and amplitude values chosen for each, and
  the medical literature that informed those choices. Explains why the
  default is set to essential tremor and how to add custom profiles.

- **SAC Section:** Describes every SAC hyperparameter, its role in training,
  and why the chosen default value is appropriate for this environment.
  Includes guidance on which parameters to adjust first when debugging
  training instability.

- **Reward Section:** Walks through each reward weight, explains how
  changing it affects agent behaviour, and provides recommended ranges
  based on experimentation. Highlights the relationship between the
  tracking weight and the smoothness weight as the most important balance
  to get right.

- **Safety Section:** Explains the three safety modes (hard, soft, adaptive),
  when to use each one, and why adaptive mode is recommended for most
  training runs. Documents the safety margin and soft penalty weight.

- **Checkpointing Section:** Describes the save frequency, the rolling
  checkpoint window, the best-model tracking logic, and the naming
  convention. Explains how to resume training from a checkpoint.

- **Logging Section:** Documents the TensorBoard backend configuration,
  the logging frequency, and every scalar metric that is recorded.

- **Evaluation Section:** Explains the evaluation frequency, the number of
  evaluation episodes, and the human feedback collection schedule.

- **Seed Section:** Documents the master seed and the derived seed offsets
  for each random number generator, and explains why reproducibility
  matters for comparing the SB3 baseline against the custom SAC
  implementation.

### docs/tremor_model.md — Tremor Signal Model & Clinical References

Provides the scientific foundation for the tremor simulation used in this
project.

**Contents:**

- **What Is Tremor:** A concise explanation of involuntary rhythmic muscle
  contractions, why they occur, and why they are problematic during
  microsurgery where sub-millimetre precision is required.

- **Tremor Classification:** Describes the three categories of tremor
  modelled in this project:
  - **Essential tremor** (4–8 Hz) is the most common movement disorder,
    affecting roughly four percent of adults over 40. It produces a
    postural and kinetic tremor that worsens during fine motor tasks
    like surgery.
  - **Parkinson's tremor** (3–6 Hz) is a resting tremor that can persist
    during surgical manoeuvres, characterised by lower frequency and
    higher amplitude than essential tremor.
  - **Physiological tremor** (8–12 Hz) is present in all healthy
    individuals and is usually imperceptible, but fatigue, stress, and
    caffeine intake can amplify it to clinically relevant levels during
    long surgical procedures.

- **Amplitude Ranges:** Documents the tremor amplitudes used in the
  simulation (0.02–0.20 mm) and explains that these values are drawn from
  published accelerometry studies of surgeon hand tremor during
  microsurgery.

- **Signal Model:** Explains the superposition-of-sinusoids approach, why
  multiple frequency components are used per tremor type, and the phase
  coherence fix that ensures each episode produces a realistic,
  continuous tremor waveform rather than random noise.

- **Limitations:** Acknowledges that the current model does not capture
  tremor amplitude modulation over time, inter-limb coupling, or the
  non-stationary nature of pathological tremor. Suggests future
  improvements such as amplitude envelopes and stochastic frequency drift.

- **Clinical References:** A bibliography of the medical and engineering
  papers that informed the frequency bands, amplitude ranges, and
  signal modelling approach.

### docs/deployment.md — Deployment & Usage Guide

Step-by-step instructions for setting up the project, running training,
launching the dashboard, and collecting human feedback.

**Contents:**

- **Prerequisites:** Lists required software (Python 3.11+, Node.js 18+,
  uv, git) and hardware recommendations (GPU optional but recommended for
  training speed; CPU is sufficient for evaluation and dashboard use).

- **Installation:**
  ```bash
  git clone <repo-url>
  cd surgical-robot-tremor-compensator-rl
  uv sync                          # install Python dependencies
  cd frontend && npm install       # install React dependencies
  ```

- **Running Training:**
  ```bash
  uv run train.py                             # train with default config
  uv run train.py --config config.yaml        # explicit config path
  uv run train.py --resume checkpoints/best_model.pt  # resume from checkpoint
  ```
  Describes what to expect during training, how to monitor progress with
  TensorBoard, and when to stop.

- **Launching the Dashboard:**
  ```bash
  # Terminal 1: start the FastAPI backend
  uv run uvicorn evaluate:app --reload --port 8000

  # Terminal 2: start the React frontend
  cd frontend && npm run dev
  ```
  Explains that the React dev server runs on port 5173 and proxies API
  requests to port 8000. Describes each of the five dashboard panels and
  what to look for when evaluating training quality.

- **Collecting Human Feedback:** Walks through the evaluation workflow —
  how to open the trajectory viewer, score an episode, and trigger reward
  model retraining. Explains the scoring rubric (1 = severe over-correction
  or missed tremor, 5 = indistinguishable from a tremor-free trajectory).

- **Running Evaluation:**
  ```bash
  uv run evaluate.py --checkpoint checkpoints/best_model.pt
  ```
  Describes the five evaluation metrics, what target values to aim for,
  and how to interpret the results.

- **Running Tests:**
  ```bash
  uv run pytest                    # all tests
  uv run pytest tests/test_surgical_env.py  # single test file
  uv run pytest --cov              # with coverage report
  ```

- **Production Deployment:** Notes on building the React frontend for
  production (`npm run build`), serving it from FastAPI as static files,
  and updating the CORS origin to match the deployment domain.

---

## .gitignore Additions

```gitignore
# Python / uv
.venv/
__pycache__/
*.egg-info/

# Training artifacts
checkpoints/
logs/
feedback/

# Documentation (generated locally, not tracked)
docs/

# Frontend
frontend/node_modules/
frontend/dist/

# IDE
.idea/
.vscode/

# Environment variables
.env
.env.local
.env.production
```

---

## Regulatory Compliance

This project targets medical device software and must be developed in
accordance with the following regulatory standards. Even during the research
and simulation phase, establishing these practices early prevents costly
retrofitting when moving toward clinical deployment.

### IEC 62304 — Medical Device Software Lifecycle

IEC 62304 defines the software development lifecycle for medical devices.
This project follows its three safety classifications:

- **Class A** — No injury possible. Not applicable here.
- **Class B** — Non-serious injury possible. The tremor compensator falls
  into this class during simulation and benchtop testing.
- **Class C** — Death or serious injury possible. The compensator enters
  this class when connected to a physical surgical robot operating on tissue.

**Required processes:**

1. **Software Development Planning** — This spec document serves as the
   initial software development plan. It defines the architecture, risk
   controls, testing strategy, and traceability requirements.

2. **Software Requirements Analysis** — Each component (environment, agent,
   safety layer, feedback pipeline) has defined inputs, outputs, and
   acceptance criteria documented in the phased development plan.

3. **Software Architectural Design** — The component diagram in
   `docs/architecture.md` satisfies this requirement. Each subsystem has
   a defined interface and responsibility boundary.

4. **Software Detailed Design** — Implemented through the code itself, with
   type hints, docstrings, and the reward function commentary providing
   design-level documentation.

5. **Software Unit Verification** — The `tests/` directory with defined
   test cases for every component.

6. **Software Integration Testing** — End-to-end tests that verify the
   full training loop from environment reset through action selection,
   safety projection, and reward computation.

7. **Software System Testing** — Evaluation runs across all tremor types
   with formal pass/fail criteria on the five evaluation metrics.

8. **Software Release** — Checkpointing strategy with config snapshots
   ensures every released model is fully reproducible.

### IEC 62443 — Cybersecurity for Medical Devices

Relevant because the system exposes FastAPI endpoints and a WebSocket
connection:

- **Network Segmentation:** The FastAPI server should only be accessible
  on the local network during development. Production deployment must use
  HTTPS with proper TLS certificates.

- **Authentication:** Human feedback endpoints must require authentication
  to prevent unauthorised score injection that could corrupt the reward
  model. Implement API key or OAuth2 bearer token authentication.

- **Input Validation:** All FastAPI request bodies must be validated with
  Pydantic models. Episode IDs, scores, and evaluator IDs must be
  sanitised to prevent injection attacks.

- **Dependency Security:** Regular dependency audits using `uv audit` or
  `pip-audit`. Known vulnerabilities in any dependency block deployment.

### FDA Software Pre-Submission Pathway

For eventual clinical deployment in the United States:

- **Software as a Medical Device (SaMD):** This system qualifies as SaMD
  because it processes surgeon input to produce a compensated output that
  directly affects patient safety.

- **Predetermined Change Control Plan (PCCP):** Document which model
  updates (retraining, hyperparameter changes) require a new submission
  versus which fall under the PCCP.

- **Algorithm Transparency:** The reward function with its five clinically
  documented components, the safety constraint layer, and the human feedback
  pipeline must all be explainable to FDA reviewers. Avoid black-box
  justifications.

---

## Risk Management (ISO 14971)

ISO 14971 requires a formal risk management process throughout the entire
product lifecycle. The following risk analysis identifies the primary
hazards associated with the tremor compensation system.

### Risk Analysis Matrix

Severity levels:
- **S1 (Negligible):** No impact on patient safety
- **S2 (Minor):** Temporary discomfort, no lasting harm
- **S3 (Serious):** Injury requiring medical intervention
- **S4 (Critical):** Permanent injury or life-threatening
- **S5 (Catastrophic):** Death

Probability levels:
- **P1 (Improbable):** < 1 in 1,000,000 uses
- **P2 (Remote):** 1 in 100,000 to 1,000,000
- **P3 (Occasional):** 1 in 1,000 to 100,000
- **P4 (Probable):** 1 in 100 to 1,000
- **P5 (Frequent):** > 1 in 100

### Identified Hazards

| ID | Hazard | Severity | Probability | Risk | Mitigation |
|---|---|---|---|---|---|
| H-01 | Agent over-corrects, pushing robot tip into tissue | S4 | P3 | High | Safety constraint layer (hard mode) clips all actions that approach tissue boundary. Tested with `test_safety_constraints.py`. |
| H-02 | Agent fails to compensate, tremor passes through to robot | S3 | P3 | Medium | Monitoring `compensation_error_mm` with automatic training halt if error exceeds threshold. Fallback to raw low-pass filter. |
| H-03 | Compensation latency exceeds 20ms, delayed correction causes overshoot | S3 | P2 | Medium | Latency penalty in reward function. Hard limit enforced in safety layer: if latency exceeds 40ms, system reverts to filtered input. |
| H-04 | Reward model corrupted by inconsistent human feedback | S2 | P3 | Medium | Inter-rater reliability checks. Outlier filtering. Minimum 50 labels before reward model influences training. |
| H-05 | Safety constraint mode transitions (soft to hard) cause sudden behaviour change | S3 | P2 | Medium | Gradual transition with linear interpolation of penalty weight over 10,000 steps rather than abrupt switch. |
| H-06 | Model checkpoint loaded from incompatible config | S2 | P2 | Low | Config snapshot stored in every checkpoint. Loader validates config compatibility before resuming. |
| H-07 | WebSocket endpoint exploited to inject false metrics | S2 | P3 | Medium | Authentication required on all endpoints. Metrics are read-only from the dashboard. |
| H-08 | PyBullet simulation diverges from real robot physics | S3 | P4 | High | Sim-to-real gap documented. Real-robot validation required before clinical use. Simulation parameters calibrated against manufacturer specs. |

### Residual Risk Acceptance

After all mitigations are applied, the overall residual risk must be
evaluated against the clinical benefit. The safety constraint layer
eliminates the highest-severity hazard (H-01) by making tissue penetration
physically impossible at the software level. Remaining risks are addressed
through monitoring, fallback mechanisms, and mandatory real-robot validation
before any clinical deployment.

---

## Audit Trail & Traceability

Medical device software requires complete traceability from requirements
through implementation to verification. This project implements audit
trails at three levels.

### Code-Level Traceability

Every source file maps to a requirement in this spec:

```
Requirement                    → Implementation           → Test
─────────────────────────────────────────────────────────────────────
State space (18-dim obs)       → env/surgical_env.py      → test_surgical_env.py
Tremor profiles (3 types)      → env/tremor_generator.py  → test_tremor_generator.py
5-component reward function    → env/surgical_env.py      → test_reward_function.py
Safety constraint (3 modes)    → safety/constraints.py    → test_safety_constraints.py
FFT frequency detection        → utils/signal_processing  → test_signal_processing.py
SAC agent (SB3 + custom)       → agents/sac_agent.py      → (training integration test)
Reward model (neural net)      → agents/reward_model.py   → test_reward_model.py
Human feedback storage         → evaluate.py              → test_feedback_pipeline.py
```

### Training Run Audit Log

Every training run produces an immutable audit log stored in
`logs/{run_id}/audit.jsonl`. Each line records:

```json
{
  "timestamp": "2026-03-06T12:00:00Z",
  "event": "training_start | checkpoint_save | config_change | feedback_received | reward_model_retrain | training_stop",
  "details": { ... },
  "config_hash": "sha256:abc123...",
  "code_commit": "git:39039b6"
}
```

This log is append-only and must never be modified after creation.

### Model Registry

Every saved model is registered with:
- Git commit hash of the code used to train it
- SHA-256 hash of the config.yaml used
- Training step count and episode count
- All five evaluation metrics at save time
- The exact versions of all Python dependencies (from `uv.lock`)

---

## Clinical Validation Protocol

The simulation environment is a necessary first step, but clinical
deployment requires validation against real surgical data. This protocol
defines the bridge from simulation to reality.

### Phase A — Benchtop Validation

1. **Record real tremor data** from surgeons using an accelerometer attached
   to a surgical instrument during simulated (non-patient) procedures.
2. **Replay recorded tremor** through the trained agent in place of the
   synthetic tremor generator.
3. **Compare compensation quality** between synthetic and real tremor inputs.
   The agent must achieve within 20% of its simulation performance on real
   tremor data.
4. **Measure actual computation latency** on the target hardware to verify
   the sub-20ms requirement is met in practice, not just in simulation.

### Phase B — Cadaver/Phantom Study

1. Connect the compensation system to a physical robot arm operating on a
   tissue phantom or cadaver specimen.
2. Inject recorded tremor signals into the control loop.
3. Measure tissue contact force, trajectory deviation, and compensation
   latency with physical sensors.
4. An independent clinical evaluator scores compensation quality using the
   same 1-5 rubric from the human feedback pipeline.

### Phase C — First-in-Human (requires regulatory approval)

1. Operate under an Investigational Device Exemption (IDE) or equivalent.
2. Begin with low-risk procedures where the consequence of compensation
   failure is minimal.
3. The safety constraint layer operates in hard mode at all times. A physical
   emergency stop is available to the supervising surgeon.
4. Collect post-operative outcomes data to demonstrate clinical benefit.

### Sim-to-Real Gap Mitigation

- **Domain randomisation:** During training, randomise tremor amplitudes,
  frequencies, tissue boundary positions, and simulation timestep within
  clinically plausible ranges to build robustness.
- **System identification:** Measure the real robot's joint friction,
  backlash, and control delay, then calibrate the PyBullet simulation to
  match these physical properties.
- **Transfer learning:** Fine-tune the simulation-trained model on a small
  dataset of real tremor recordings before deployment.

---

## Data Privacy (HIPAA/GDPR)

Even though the current system operates on simulated data, the human
feedback pipeline and future clinical validation will involve real people.
Privacy protections must be established from the start.

### Data Classification

| Data Type | Classification | Storage | Retention |
|---|---|---|---|
| Simulated tremor signals | Non-sensitive | Local filesystem | Indefinite |
| Training metrics and logs | Non-sensitive | Local filesystem | Indefinite |
| Human feedback scores | Pseudonymised PII | `feedback/human_labels.jsonl` | 3 years post-study |
| Evaluator identifiers | PII | Separate mapping file | 3 years post-study |
| Recorded surgeon tremor data (Phase A+) | PHI / Special Category | Encrypted storage | Per IRB protocol |
| Patient outcome data (Phase C) | PHI / Special Category | Encrypted storage | Per IRB protocol |

### HIPAA Compliance (United States)

- **Minimum Necessary Rule:** Only collect the minimum data required for
  the feedback pipeline. Evaluator scores need an anonymised identifier,
  not names or contact details.
- **Access Controls:** The `feedback/` directory and any clinical data must
  be access-controlled. Only authorised researchers can read or write
  human labels.
- **Encryption:** All data classified as PHI must be encrypted at rest
  (AES-256) and in transit (TLS 1.3).
- **Audit Logging:** All access to PHI must be logged in the audit trail.
- **Business Associate Agreements:** Required if any cloud services are
  used for training, storage, or feedback collection.

### GDPR Compliance (European Union)

- **Lawful Basis:** Consent for evaluator participation in the feedback
  pipeline. Legitimate interest for simulated data processing.
- **Data Subject Rights:** Evaluators can request deletion of their
  feedback scores. The system must support selective deletion from
  `human_labels.jsonl` without corrupting the reward model.
- **Data Protection Impact Assessment:** Required before collecting real
  surgeon tremor data or patient outcome data.
- **Data Processing Agreement:** Required with any third-party processor.

### Environment Variables for Sensitive Configuration

```bash
# .env (never committed to git)
FEEDBACK_API_KEY=sk-feedback-...      # API key for feedback endpoints
ENCRYPTION_KEY=base64:...             # AES-256 key for PHI encryption
DATABASE_URL=postgresql://...         # If using a database for audit logs
TENSORBOARD_AUTH_TOKEN=...            # If exposing TensorBoard remotely
```

---

## Software Bill of Materials (SBOM)

The FDA requires a comprehensive SBOM for cybersecurity review of medical
device software. This project generates SBOMs in CycloneDX format.

### Generating the SBOM

```bash
# Python dependencies
uv run pip install cyclonedx-bom
uv run cyclonedx-py environment -o sbom-python.json --format json

# Frontend dependencies
cd frontend
npx @cyclonedx/cyclonedx-npm --output-file ../sbom-frontend.json
```

### SBOM Contents

The SBOM must include:
- Every direct and transitive Python dependency with exact version
- Every direct and transitive npm dependency with exact version
- The Python interpreter version
- The Node.js runtime version
- The operating system and architecture used for training
- Known vulnerabilities (CVEs) for each component at time of generation

### Vulnerability Monitoring

```bash
# Check Python dependencies for known vulnerabilities
uv run pip-audit

# Check npm dependencies
cd frontend && npm audit
```

Run these checks:
- Before every release
- Weekly during active development
- Immediately when a security advisory is published for any dependency

### SBOM Storage

SBOMs are regenerated for each release and stored alongside the release
artifacts. They are not committed to the repository but are archived with
the model registry entry for each deployed model version.

---

## CI/CD Pipeline

Automated testing, linting, and security scanning on every push and pull
request. The pipeline uses GitHub Actions.

### .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pip-audit
      - name: Generate SBOM
        run: uv run cyclonedx-py environment -o sbom.json --format json
      - uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: sbom.json

  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 18
      - run: npm ci
      - run: npm run lint
      - run: npm run build
      - run: npm audit --audit-level=moderate

  integration:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest tests/integration/ -v
```

### Branch Protection Rules

- `main` branch requires all CI checks to pass before merge
- At least one approving review required for pull requests to `main`
- `dev` branch runs CI but does not block merge (used for active development)

---

## Docker / Containerisation

Reproducible environments for training and deployment.

### Dockerfile

```dockerfile
# ── Training image ───────────────────────────────────────────
FROM python:3.11-slim AS training

WORKDIR /app

# Install system dependencies for PyBullet
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source code
COPY env/ env/
COPY agents/ agents/
COPY safety/ safety/
COPY utils/ utils/
COPY train.py evaluate.py config.yaml ./

ENTRYPOINT ["uv", "run"]
CMD ["train.py"]

# ── Dashboard image ──────────────────────────────────────────
FROM node:18-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM training AS dashboard
COPY --from=frontend-build /frontend/dist /app/frontend/dist
COPY dashboard/ dashboard/
CMD ["uvicorn", "evaluate:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
services:
  training:
    build:
      context: .
      target: training
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
      - ./config.yaml:/app/config.yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  dashboard:
    build:
      context: .
      target: dashboard
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./feedback:/app/feedback
      - ./checkpoints:/app/checkpoints
    environment:
      - FEEDBACK_API_KEY=${FEEDBACK_API_KEY}
    depends_on:
      - training

  tensorboard:
    image: tensorflow/tensorflow:latest
    command: tensorboard --logdir /logs --host 0.0.0.0 --port 6006
    ports:
      - "6006:6006"
    volumes:
      - ./logs:/logs
```

### .dockerignore

```
.git/
.venv/
__pycache__/
*.egg-info/
.idea/
.vscode/
frontend/node_modules/
docs/
tests/
.env
```

---

## ML Experiment Tracking

Track every training run, compare models, and version datasets using
MLflow.

### Why MLflow

MLflow is chosen over Weights & Biases because it can be self-hosted,
which is important for healthcare projects where training data and metrics
may be sensitive and cannot leave the organisation's infrastructure.

### Integration

```python
# In train.py
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # local MLflow server
mlflow.set_experiment("surgical-tremor-compensator")

with mlflow.start_run():
    # Log config as parameters
    mlflow.log_params(flatten_config(config))

    # Log metrics during training
    mlflow.log_metric("compensation_error_mm", error, step=episode)
    mlflow.log_metric("tremor_rejection_ratio_dB", ratio, step=episode)
    mlflow.log_metric("safety_violations", violations, step=episode)

    # Log model artifact
    mlflow.pytorch.log_model(model, "sac_model")

    # Log config file
    mlflow.log_artifact("config.yaml")
```

### MLflow Server (docker-compose addition)

```yaml
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /artifacts
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/artifacts
      - mlflow-db:/mlflow.db

volumes:
  mlflow-data:
  mlflow-db:
```

### What Gets Tracked Per Run

- All `config.yaml` parameters (flattened as key-value pairs)
- All five evaluation metrics at every evaluation interval
- SAC training metrics (entropy, actor loss, critic loss, ent_coef)
- The trained model artifact (weights, optimizer state)
- The config.yaml and git commit hash
- Training duration, hardware used, Python/PyTorch versions

### Model Comparison

MLflow's comparison UI allows side-by-side comparison of:
- SB3 SAC (Phase 2) versus custom SAC (Phase 3)
- Different hyperparameter configurations
- Different tremor types as evaluation conditions
- Impact of human feedback on compensation quality

---

## Integration Tests

Beyond unit tests, integration tests verify that components work together
correctly through the full pipeline.

### tests/integration/

```
tests/
├── integration/
│   ├── test_training_loop.py       # Full train loop for 100 steps
│   ├── test_evaluation_pipeline.py # Evaluate checkpoint, verify metrics
│   ├── test_feedback_pipeline.py   # Submit feedback via API, verify storage
│   ├── test_safety_modes.py        # Soft → hard transition during training
│   └── test_checkpoint_resume.py   # Save, load, verify identical behaviour
```

### Key Integration Test Cases

- **Training Loop (test_training_loop.py):**
  Train the SAC agent for 100 steps on the surgical environment. Verify
  that the replay buffer fills, the agent produces non-zero actions after
  warmup, the reward is computed correctly with all five components, and
  the logger writes to TensorBoard without errors.

- **Evaluation Pipeline (test_evaluation_pipeline.py):**
  Load a checkpoint, run 5 evaluation episodes, and verify that all five
  evaluation metrics are computed and fall within plausible ranges. Ensure
  that trajectory plots are saved when configured.

- **Feedback Pipeline (test_feedback_pipeline.py):**
  Start the FastAPI server, submit 10 feedback scores via the API, verify
  they are stored in `human_labels.jsonl`, trigger reward model retraining,
  and verify the model produces non-zero predictions.

- **Safety Mode Transition (test_safety_modes.py):**
  Run training in adaptive mode and verify that the safety layer uses soft
  penalties for the first 50,000 steps and switches to hard constraints
  afterward. Confirm the transition does not cause a spike in safety
  violations.

- **Checkpoint Resume (test_checkpoint_resume.py):**
  Train for 200 steps, save a checkpoint, load it in a new process, train
  for 100 more steps, and verify that the model's behaviour is identical
  to continuous training for 300 steps (given the same seed).

---

## Error Handling & Recovery Strategy

Training runs can fail due to hardware issues, numerical instability, or
external interruptions. The system must handle these failures gracefully.

### Automatic Checkpointing on Failure

```python
import signal
import sys

def graceful_shutdown(signum, frame):
    """Save emergency checkpoint on SIGINT/SIGTERM."""
    save_checkpoint(model, optimizer, step, episode, config, path="checkpoints/emergency.pt")
    logger.info(f"Emergency checkpoint saved at step {step}")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)
```

### Numerical Instability Detection

- Monitor for NaN values in loss functions and reward computation.
  If NaN is detected, halt training, save a diagnostic checkpoint, and
  log the exact state that caused the instability.
- Common causes: entropy coefficient diverging, critic loss explosion
  from large reward penalties (especially `r_safety = -100`).
- Recovery: reload the last stable checkpoint and reduce the learning
  rate by half.

### Replay Buffer Corruption

- The replay buffer is stored in memory during training. If training
  crashes, the buffer is lost.
- Periodic buffer snapshots (every 10,000 steps) can be enabled in config
  but are disabled by default due to size (buffer_size × observation_dim
  × 4 bytes ≈ 72 MB).
- When resuming from checkpoint with `save_buffer: true`, the buffer
  snapshot is loaded alongside the model weights.

### FastAPI Server Recovery

- The FastAPI server runs independently from the training loop. If the
  server crashes, training continues unaffected.
- Uvicorn's `--reload` flag automatically restarts the server on code
  changes during development.
- In production, use a process manager (systemd or Docker restart policy)
  to ensure the server is always available.

### Filesystem Full

- Before saving a checkpoint, verify that at least 500 MB of free disk
  space is available. If not, delete the oldest non-best checkpoint and
  try again. If still insufficient, log a warning and skip the save.

---

## Performance Benchmarks

Baseline performance targets for training throughput and inference latency
on reference hardware.

### Reference Hardware

- **Development:** CPU-only (Intel i7 or equivalent), 16 GB RAM
- **Training:** NVIDIA RTX 3080 or equivalent, 32 GB RAM
- **Deployment target:** NVIDIA Jetson Orin or equivalent edge GPU

### Training Throughput Targets

| Metric | CPU | GPU |
|---|---|---|
| Environment steps per second | 500–1,000 | 2,000–5,000 |
| SAC updates per second | 50–100 | 200–500 |
| Time per 100k training steps | 3–6 hours | 30–60 minutes |

### Inference Latency Targets (single step)

| Component | Target | Hard Limit |
|---|---|---|
| Observation construction | < 0.5 ms | 1 ms |
| SAC forward pass (action selection) | < 1 ms | 2 ms |
| Safety projection | < 0.5 ms | 1 ms |
| Total pipeline (obs → safe action) | < 2 ms | 5 ms |
| End-to-end with physics sim | < 5 ms | 10 ms |

The 5ms simulation timestep provides a 5ms budget per step. The total
pipeline must complete within this budget to maintain real-time operation.
The 20ms latency penalty threshold allows for up to 4 steps of delay
between tremor onset and compensation.

### Profiling

```bash
# Profile a training run
uv run python -m cProfile -o profile.prof train.py --steps 1000
uv run snakeviz profile.prof

# Profile inference latency
uv run pytest tests/test_inference_latency.py -v --benchmark
```

Add `py-spy` and `snakeviz` to dev dependencies for profiling:

```toml
[dependency-groups]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-benchmark>=4.0.0",
    "ruff>=0.4.0",
    "py-spy>=0.3.0",
    "snakeviz>=2.2.0",
    "pip-audit>=2.6.0",
    "cyclonedx-bom>=4.0.0",
]
```

---

## Monitoring & Alerting (Production)

When the system is deployed for real-time compensation, it must be
continuously monitored to detect degradation or failure.

### Metrics Stack

- **Prometheus** — Collects time-series metrics from the FastAPI server
  and the inference pipeline.
- **Grafana** — Dashboards and alerting rules built on Prometheus data.

### Exposed Metrics (Prometheus format)

```python
# In the inference server
from prometheus_client import Histogram, Counter, Gauge

inference_latency = Histogram(
    "tremor_inference_latency_seconds",
    "Time to compute compensation action",
    buckets=[0.001, 0.002, 0.005, 0.010, 0.020]
)

safety_violations_total = Counter(
    "tremor_safety_violations_total",
    "Cumulative tissue boundary violations"
)

compensation_error_mm = Gauge(
    "tremor_compensation_error_mm",
    "Current RMS compensation error"
)

model_version = Gauge(
    "tremor_model_version",
    "Currently loaded model checkpoint epoch"
)
```

### Alerting Rules

| Alert | Condition | Severity |
|---|---|---|
| High inference latency | p99 latency > 10ms for 5 minutes | Critical |
| Safety violation | Any tissue boundary breach | Critical |
| Compensation error spike | Error > 2× rolling average | Warning |
| Model not loaded | model_version == 0 for 30 seconds | Critical |
| API server down | No heartbeat for 60 seconds | Critical |
| Disk space low | < 1 GB free on checkpoint volume | Warning |

### Grafana Dashboard Panels

1. **Inference Latency** — Histogram heatmap with p50, p95, p99 lines
2. **Compensation Error** — Time series with rolling average
3. **Safety Violations** — Counter with rate-of-change overlay
4. **System Resources** — CPU, memory, GPU utilisation
5. **API Request Rate** — Requests per second by endpoint

### docker-compose addition

```yaml
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}

volumes:
  grafana-data:
```

---

## Project Hygiene

### LICENSE

This project uses the **Apache License 2.0**, which permits commercial use,
modification, and distribution while requiring attribution and a statement
of changes. Apache 2.0 is preferred over MIT for medical device software
because it includes an explicit patent grant, protecting contributors and
users from patent litigation.

The `LICENSE` file must be present in the repository root.

### Semantic Versioning

This project follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (X.0.0) — Breaking changes to the environment API, observation
  space, action space, or config.yaml structure.
- **MINOR** (0.X.0) — New features that are backward compatible (new tremor
  profiles, new dashboard panels, new API endpoints).
- **PATCH** (0.0.X) — Bug fixes, documentation updates, dependency bumps.

Current version: `0.1.0` (initial development, no stability guarantees).

The version is defined in `pyproject.toml` and must be updated with every
release.

### CHANGELOG.md

Maintained in [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Initial project specification and skill file
- Phased development plan (7 phases)

## [0.1.0] — YYYY-MM-DD

### Added
- Project scaffolding and folder structure
- config.yaml with all hyperparameters
```

### CONTRIBUTING.md

```markdown
# Contributing

## Getting Started

1. Fork the repository
2. Create a feature branch from `dev`: `git checkout -b feature/your-feature`
3. Install dependencies: `uv sync`
4. Make your changes
5. Run tests: `uv run pytest`
6. Run linting: `uv run ruff check . && uv run ruff format --check .`
7. Submit a pull request to `dev`

## Code Standards

- Python code must pass `ruff check` and `ruff format` with no errors
- All new functions require type hints and docstrings
- Reward function changes must include clinical rationale in comments
- Safety-critical code changes require two reviewer approvals
- Never hardcode hyperparameters — use config.yaml

## Pull Request Requirements

- All CI checks must pass
- At least one approving review
- Safety-critical changes (safety/, reward function) require two reviews
- Update CHANGELOG.md with your changes
- Update docs/ if your change affects architecture or configuration

## Reporting Issues

- Use GitHub Issues with the appropriate label (bug, feature, safety)
- Security vulnerabilities: email security@[org].com, do not open a public issue
```

### Dashboard Accessibility (WCAG 2.1 AA)

The React dashboard must meet WCAG 2.1 Level AA accessibility standards,
which is particularly important for healthcare applications where users
may have diverse abilities.

**Requirements:**

- **Colour Contrast:** All text must have a contrast ratio of at least
  4.5:1 against its background. Chart colours must be distinguishable
  by users with colour vision deficiency. Use patterns or labels in
  addition to colour to differentiate data series.

- **Keyboard Navigation:** Every interactive element (buttons, dropdowns,
  chart tooltips) must be reachable and operable with keyboard alone.
  Focus order must follow a logical sequence.

- **Screen Reader Support:** All charts must have descriptive `aria-label`
  attributes summarising the data they display. Status updates (new
  metrics arriving via WebSocket) must be announced using `aria-live`
  regions.

- **Text Scaling:** The dashboard must remain usable when browser text
  is scaled to 200%. No content should be clipped or overlapping.

- **Motion Sensitivity:** Live-updating charts must respect the
  `prefers-reduced-motion` media query. When reduced motion is preferred,
  charts update values without animation.

---

## Updated Folder Structure (final)

```
surgical-robot-tremor-compensator-rl/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline
├── env/
│   ├── __init__.py
│   ├── surgical_env.py            # Custom Gymnasium environment
│   ├── tremor_generator.py        # Simulates hand tremor signals
│   └── physics_sim.py             # PyBullet robot arm simulation
├── agents/
│   ├── __init__.py
│   ├── sac_agent.py               # SAC implementation (SB3 + custom)
│   └── reward_model.py            # Human-in-the-loop reward model
├── safety/
│   ├── __init__.py
│   └── constraints.py             # Hard/soft/adaptive safety constraints
├── dashboard/
│   ├── __init__.py
│   └── visualizer.py              # Matplotlib-based offline plots
├── utils/
│   ├── __init__.py
│   ├── signal_processing.py       # FFT-based tremor frequency analysis
│   └── logger.py                  # Training metrics + audit trail logger
├── tests/
│   ├── test_tremor_generator.py
│   ├── test_surgical_env.py
│   ├── test_reward_function.py
│   ├── test_safety_constraints.py
│   ├── test_signal_processing.py
│   ├── test_reward_model.py
│   ├── test_inference_latency.py  # Benchmark tests
│   └── integration/
│       ├── test_training_loop.py
│       ├── test_evaluation_pipeline.py
│       ├── test_feedback_pipeline.py
│       ├── test_safety_modes.py
│       └── test_checkpoint_resume.py
├── docs/                          # Not tracked in git
│   ├── architecture.md
│   ├── api_reference.md
│   ├── config_guide.md
│   ├── tremor_model.md
│   └── deployment.md
├── frontend/                      # React dashboard (Vite + TypeScript)
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── components/
│       │   ├── TrajectoryPlot.tsx
│       │   ├── FrequencySpectrum.tsx
│       │   ├── RewardBreakdown.tsx
│       │   ├── SafetyZone.tsx
│       │   └── TrainingMetrics.tsx
│       ├── hooks/
│       │   └── useWebSocket.ts
│       └── types/
│           └── metrics.ts
├── monitoring/                    # Prometheus/Grafana configs
│   └── prometheus.yml
├── feedback/                      # Created at runtime, not tracked
│   └── human_labels.jsonl
├── checkpoints/                   # Not tracked in git
├── logs/                          # Not tracked in git
├── train.py                       # Main training entrypoint
├── evaluate.py                    # Evaluation + FastAPI feedback server
├── config.yaml                    # All hyperparameters
├── pyproject.toml                 # Project config & dependencies (uv)
├── uv.lock                       # Locked dependency versions
├── Dockerfile
├── docker-compose.yml
├── LICENSE                        # Apache License 2.0
├── CHANGELOG.md                   # Version history
├── CONTRIBUTING.md                # Contribution guidelines
├── .env.example                   # Template for environment variables
└── README.md
```