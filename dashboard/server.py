"""Live training dashboard server.

Runs the SAC training loop in a background thread while serving the
FastAPI dashboard. Training metrics are broadcast to all connected
WebSocket clients in real time.

Usage:
    uv run python -m dashboard.server
    uv run python -m dashboard.server --agent custom --steps 100000
"""

from __future__ import annotations

import argparse
import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agents.sac_agent import SACAgent
from agents.sac_custom import CustomSACAgent
from env.surgical_env import SurgicalTremorEnv
from safety.constraints import SafetySurgicalEnv
from utils.logger import TrainingLogger

# ── Shared State ─────────────────────────────────────────────

_ws_clients: list[WebSocket] = []
_metrics_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
_training_status: dict[str, Any] = {
    "running": False,
    "step": 0,
    "episode": 0,
    "agent_type": "custom",
}
_cli_args: argparse.Namespace | None = None


# ── Broadcast ────────────────────────────────────────────────

async def _broadcast_loop() -> None:
    """Background task that sends queued metrics to all WebSocket clients."""
    while True:
        metrics = await _metrics_queue.get()
        dead: list[WebSocket] = []
        for ws in _ws_clients:
            try:
                await ws.send_json(metrics)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in _ws_clients:
                _ws_clients.remove(ws)


def push_metrics(metrics: dict[str, Any]) -> None:
    """Thread-safe push of metrics to the broadcast queue."""
    try:
        _metrics_queue.put_nowait(metrics)
    except Exception:
        pass


# ── Training Thread ──────────────────────────────────────────

def _run_training(
    config_path: str,
    agent_type: str,
    total_steps: int,
    use_physics: bool,
) -> None:
    """Run training in a background thread, pushing metrics to WebSocket."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed + 1)

    base_env = SurgicalTremorEnv(config_path=config_path, use_physics=use_physics)
    env = SafetySurgicalEnv(base_env, config_path=config_path)

    logger = TrainingLogger(
        log_dir=config["logging"]["log_dir"],
        config_path=config_path,
    )

    _training_status.update({
        "running": True,
        "step": 0,
        "episode": 0,
        "agent_type": agent_type,
    })

    checkpoint_dir = Path(config["checkpointing"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_freq = config["logging"].get("log_freq_steps", 100)

    if agent_type == "custom":
        agent = CustomSACAgent(env=env, config_path=config_path)
        _train_custom(agent, env, total_steps, logger, checkpoint_dir, log_freq)
    else:
        agent = SACAgent(env=env, config_path=config_path)
        _train_sb3(agent, env, total_steps, logger, checkpoint_dir, log_freq)

    _training_status["running"] = False
    logger.close()
    print("Training complete.")


def _train_custom(
    agent: CustomSACAgent,
    env: SafetySurgicalEnv,
    total_steps: int,
    logger: TrainingLogger,
    checkpoint_dir: Path,
    log_freq: int,
) -> None:
    ep_reward = 0.0
    ep_errors: list[float] = []
    episodes_done = 0
    best_error = float("inf")

    obs, _ = env.reset()
    for step in range(1, total_steps + 1):
        if not _training_status["running"]:
            break

        if agent.replay_buffer.size < agent.batch_size:
            action = env.action_space.sample()
        else:
            action, _ = agent.predict(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.add(obs, action, reward, next_obs, terminated)

        ep_reward += reward
        if "compensation_error_mm" in info:
            ep_errors.append(info["compensation_error_mm"])

        obs = next_obs

        if agent.replay_buffer.size >= agent.batch_size and step % agent.train_freq == 0:
            for _ in range(agent.gradient_steps):
                agent._update()

        if step % log_freq == 0:
            metrics_msg = {
                "type": "metrics",
                "step": step,
                "episode": episodes_done,
                "reward_total": float(reward),
                "reward_tracking": float(info.get("reward_tracking", 0)),
                "reward_smooth": float(info.get("reward_smooth", 0)),
                "reward_safety": float(info.get("reward_safety", 0)),
                "reward_latency": float(info.get("reward_latency", 0)),
                "reward_human": float(info.get("reward_human", 0)),
                "compensation_error_mm": float(info.get("compensation_error_mm", 0)),
                "tissue_proximity_min": float(info.get("tissue_proximity_mm", 50)),
                "sac_entropy": float(agent.metrics.get("sac/entropy", 0)),
                "sac_actor_loss": float(agent.metrics.get("sac/actor_loss", 0)),
                "sac_critic_loss": float(agent.metrics.get("sac/critic_loss", 0)),
                "sac_ent_coef": float(agent.ent_coef),
            }
            push_metrics(metrics_msg)
            _training_status["step"] = step
            _training_status["episode"] = episodes_done

        if done:
            episodes_done += 1
            avg_error = sum(ep_errors) / len(ep_errors) if ep_errors else 0.0

            if avg_error < best_error and episodes_done > 5:
                best_error = avg_error
                agent.save(checkpoint_dir / "best_model_custom.pt")

            if episodes_done % 10 == 0:
                print(f"  Episode {episodes_done} | Step {step:,} | Avg Error: {avg_error:.4f} mm")

            ep_reward = 0.0
            ep_errors = []
            obs, _ = env.reset()

    agent.save(checkpoint_dir / "final_model.pt")


def _train_sb3(
    agent: SACAgent,
    env: SafetySurgicalEnv,
    total_steps: int,
    logger: TrainingLogger,
    checkpoint_dir: Path,
    log_freq: int,
) -> None:
    from stable_baselines3.common.callbacks import BaseCallback

    class LiveCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self._ep_reward = 0.0
            self._ep_errors: list[float] = []
            self._episodes = 0
            self._best_error = float("inf")

        def _on_step(self) -> bool:
            if not _training_status["running"]:
                return False

            infos = self.locals.get("infos", [{}])
            rewards = self.locals.get("rewards", [0.0])
            dones = self.locals.get("dones", [False])

            for info, reward, done in zip(infos, rewards, dones):
                self._ep_reward += reward
                if "compensation_error_mm" in info:
                    self._ep_errors.append(info["compensation_error_mm"])

                if self.num_timesteps % log_freq == 0:
                    metrics_msg = {
                        "type": "metrics",
                        "step": self.num_timesteps,
                        "episode": self._episodes,
                        "reward_total": float(reward),
                        "reward_tracking": float(info.get("reward_tracking", 0)),
                        "reward_smooth": float(info.get("reward_smooth", 0)),
                        "reward_safety": float(info.get("reward_safety", 0)),
                        "reward_latency": float(info.get("reward_latency", 0)),
                        "reward_human": float(info.get("reward_human", 0)),
                        "compensation_error_mm": float(info.get("compensation_error_mm", 0)),
                        "tissue_proximity_min": float(info.get("tissue_proximity_mm", 50)),
                        "sac_entropy": 0.0,
                        "sac_actor_loss": 0.0,
                        "sac_critic_loss": 0.0,
                        "sac_ent_coef": 0.0,
                    }
                    push_metrics(metrics_msg)
                    _training_status["step"] = self.num_timesteps
                    _training_status["episode"] = self._episodes

                if done:
                    self._episodes += 1
                    avg_err = sum(self._ep_errors) / len(self._ep_errors) if self._ep_errors else 0
                    if avg_err < self._best_error and self._episodes > 5:
                        self._best_error = avg_err
                        self.model.save(str(checkpoint_dir / "best_model"))
                    self._ep_reward = 0.0
                    self._ep_errors = []

            return True

    agent.train(total_timesteps=total_steps, callback=LiveCallback())
    agent.save(str(checkpoint_dir / "final_model.zip"))


# ── App Setup ────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    args = _cli_args
    if args:
        training_thread = threading.Thread(
            target=_run_training,
            args=(args.config, args.agent, args.steps, args.physics),
            daemon=True,
        )
        asyncio.create_task(_broadcast_loop())
        training_thread.start()
        print(f"Training started ({args.agent} SAC, {args.steps:,} steps)")
        print(f"Dashboard: http://localhost:{args.port}")
        print(f"Frontend:  http://localhost:5173 (run 'cd frontend && npm run dev')")
    yield


app = FastAPI(title="Surgical Tremor Compensator - Live", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Live Endpoints (defined BEFORE importing evaluate routes) ─

@app.websocket("/ws/metrics")
async def ws_metrics_endpoint(websocket: WebSocket) -> None:
    """Stream live training metrics to connected clients."""
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        await websocket.send_json({"type": "status", **_training_status})
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


@app.get("/training/status")
async def training_status_live() -> dict[str, Any]:
    return _training_status


@app.post("/training/stop")
async def training_stop() -> dict[str, str]:
    _training_status["running"] = False
    return {"status": "stopping"}


# Import evaluate routes AFTER live endpoints (so live ones take priority)
from evaluate import app as eval_app  # noqa: E402
for route in eval_app.routes:
    # Skip routes that we've overridden
    if hasattr(route, "path") and route.path in ("/training/status", "/ws/metrics"):
        continue
    app.routes.append(route)


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    global _cli_args

    parser = argparse.ArgumentParser(description="Live training dashboard server")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--agent", default="custom", choices=["sb3", "custom"])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    _cli_args = parser.parse_args()

    uvicorn.run(app, host=_cli_args.host, port=_cli_args.port, log_level="warning")


if __name__ == "__main__":
    main()
