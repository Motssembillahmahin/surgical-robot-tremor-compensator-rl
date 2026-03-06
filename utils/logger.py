"""Training metrics logger with TensorBoard backend and audit trail."""

from __future__ import annotations

import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class TrainingLogger:
    """Logs training metrics to TensorBoard and maintains an audit trail.

    Attributes:
        log_dir: Directory for TensorBoard event files.
        run_id: Unique identifier for this training run.
    """

    def __init__(
        self,
        log_dir: str = "logs/",
        config_path: str = "config.yaml",
        run_id: str | None = None,
    ) -> None:
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(log_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._writer = None
        self._audit_path = self.run_dir / "audit.jsonl"
        self._config_path = config_path

        # Compute config hash for audit trail
        with open(config_path) as f:
            config_text = f.read()
        self._config_hash = hashlib.sha256(config_text.encode()).hexdigest()

        # Get git commit hash
        self._code_commit = self._get_git_commit()

    @property
    def writer(self):
        """Lazy-init TensorBoard SummaryWriter."""
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self.run_dir))
        return self._writer

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric to TensorBoard."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log multiple scalar metrics to TensorBoard."""
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)

    def log_audit_event(self, event: str, details: dict[str, Any] | None = None) -> None:
        """Append an immutable audit log entry."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "details": details or {},
            "config_hash": f"sha256:{self._config_hash}",
            "code_commit": self._code_commit,
        }
        with open(self._audit_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()

    @staticmethod
    def _get_git_commit() -> str:
        """Get current git commit hash, or 'unknown' if not in a repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return f"git:{result.stdout.strip()}"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "git:unknown"
