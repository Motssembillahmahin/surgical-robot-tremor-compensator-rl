# Surgical Robot Tremor Compensator RL

A reinforcement learning system that trains a SAC (Soft Actor-Critic) agent to filter involuntary hand tremor from a surgeon's input in real time, producing stable compensated trajectories for robotic microsurgery.

## Results

Evaluated across all 3 tremor profiles (10 episodes each):

| Tremor Type   | Error (mm) | Safety Violations | Latency (ms) |
|---------------|------------|-------------------|--------------|
| Essential     | 0.297      | 0                 | 0.28         |
| Parkinson's   | 0.331      | 0                 | 0.28         |
| Physiological | 0.268      | 0                 | 0.27         |
| **Average**   | **0.299**  | **0**             | **0.28**     |

- Compensation error reduced from 2.17 mm to 0.30 mm (86% improvement)
- Zero safety violations across all evaluations
- Sub-millisecond inference latency (well under 20ms threshold)

## Architecture

```
Surgeon Input → Tremor Generator → Surgical Environment → SAC Agent
                                         ↕                    ↕
                                   Safety Wrapper      Reward Model
                                         ↕                    ↕
                                   Robot Tip Output    Human Feedback
```

**Core components:**
- **`env/surgical_env.py`** — Custom Gymnasium environment (18-dim obs, 3-dim action)
- **`env/tremor_generator.py`** — Physiologically accurate tremor simulation (3-12 Hz)
- **`env/physics_sim.py`** — 6-DOF robot arm with DH kinematics and tissue collision
- **`agents/sac_custom.py`** — Pure PyTorch SAC with automatic entropy tuning
- **`agents/sac_agent.py`** — SB3 SAC baseline wrapper
- **`agents/reward_model.py`** — Human-in-the-loop reward model (MLP)
- **`safety/constraints.py`** — Hard/soft/adaptive safety constraint wrapper
- **`dashboard/server.py`** — Live training dashboard with WebSocket streaming

## Quick Start

```bash
# Install dependencies
uv sync
cd frontend && npm install && cd ..

# Train (custom SAC)
uv run train.py --agent custom --steps 100000

# Train (SB3 baseline)
uv run train.py --agent sb3 --steps 100000

# Train with physics simulation
uv run train.py --agent custom --physics --steps 100000

# Compare SB3 vs Custom SAC
uv run compare_sac.py --steps 100000

# Full evaluation across all tremor types
uv run full_evaluation.py --checkpoint checkpoints/best_model_custom.pt

# Launch live dashboard (training + React frontend)
uv run python -m dashboard.server --agent custom --steps 100000
cd frontend && npm run dev  # separate terminal → http://localhost:5173
```

## Human Feedback Pipeline

```bash
# Start the feedback server
uv run uvicorn evaluate:app --reload --port 8000

# Evaluate and save trajectories for review
uv run evaluate.py --checkpoint checkpoints/best_model_custom.pt --save-trajectories

# Submit feedback (scoring rubric: 1=severe issues, 5=excellent)
curl -X POST http://localhost:8000/feedback/evaluate \
  -H "Content-Type: application/json" \
  -d '{"episode_id": 1042, "score": 4, "evaluator_id": "surgeon_1"}'

# Retrain reward model on collected labels
curl -X POST http://localhost:8000/feedback/retrain-reward-model
```

## Testing

```bash
uv run pytest                     # all tests (96 tests)
uv run pytest tests/ -v           # verbose output
uv run pytest --cov               # with coverage
```

## Project Structure

```
├── env/                    # Gymnasium environment + tremor simulation
├── agents/                 # SAC agents (SB3 + custom PyTorch) + reward model
├── safety/                 # Hard/soft/adaptive safety constraints
├── dashboard/              # Live training server + Matplotlib visualizer
├── utils/                  # FFT signal processing + TensorBoard logger
├── frontend/               # React dashboard (Vite + TypeScript + Recharts)
├── tests/                  # 96 unit + integration tests
├── train.py                # Training entrypoint
├── evaluate.py             # FastAPI evaluation server
├── compare_sac.py          # SB3 vs Custom SAC comparison
├── full_evaluation.py      # Cross-tremor-type evaluation suite
├── config.yaml             # All hyperparameters (never hardcoded)
├── Dockerfile              # Multi-stage: training + dashboard
└── docker-compose.yml      # GPU training + dashboard + TensorBoard + MLflow
```

## Tech Stack

- **RL:** PyTorch, Gymnasium, Stable-Baselines3
- **API:** FastAPI, Uvicorn, WebSocket
- **Frontend:** React, Vite, TypeScript, Recharts
- **Tooling:** uv, Ruff, Pytest, TensorBoard
- **Infra:** Docker, Prometheus, Grafana, MLflow
