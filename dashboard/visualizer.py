"""Matplotlib-based offline visualization for tremor compensation.

Generates static plots for trajectory comparison, frequency spectrum,
reward breakdown, safety zone, and training metrics.
Phase 6 adds a React frontend for live visualization.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq


def plot_trajectory_comparison(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    compensated_signal: np.ndarray,
    dt_ms: float = 5.0,
    save_path: str | Path | None = None,
) -> None:
    """Plot overlay of raw, filtered, and compensated trajectories.

    Args:
        raw_signal: Raw surgeon input, shape (T, 3).
        filtered_signal: Low-pass filtered intended trajectory, shape (T, 3).
        compensated_signal: Robot tip after compensation, shape (T, 3).
        dt_ms: Timestep in milliseconds.
        save_path: If provided, save figure to this path.
    """
    t = np.arange(len(raw_signal)) * dt_ms
    labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, raw_signal[:, i], alpha=0.5, label="Raw (with tremor)")
        ax.plot(t, filtered_signal[:, i], "--", label="Intended (filtered)")
        ax.plot(t, compensated_signal[:, i], label="Compensated")
        ax.set_ylabel(f"{label} (mm)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Trajectory Comparison")
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_frequency_spectrum(
    raw_signal_1d: np.ndarray,
    compensated_signal_1d: np.ndarray,
    sample_rate_hz: float,
    save_path: str | Path | None = None,
) -> None:
    """Plot FFT spectrum showing tremor rejection.

    Args:
        raw_signal_1d: 1D raw signal (single axis).
        compensated_signal_1d: 1D compensated signal.
        sample_rate_hz: Sampling rate in Hz.
        save_path: If provided, save figure to this path.
    """
    n = len(raw_signal_1d)
    xf = fftfreq(n, d=1.0 / sample_rate_hz)
    mask = xf > 0  # Positive frequencies only

    raw_fft = np.abs(fft(raw_signal_1d))[mask]
    comp_fft = np.abs(fft(compensated_signal_1d))[mask]
    freqs = xf[mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, raw_fft, alpha=0.7, label="Raw signal")
    ax.plot(freqs, comp_fft, alpha=0.7, label="Compensated")
    ax.axvspan(3, 12, alpha=0.1, color="red", label="Tremor band (3-12 Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Tremor Frequency Spectrum")
    ax.set_xlim(0, 20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_reward_breakdown(
    reward_components: dict[str, np.ndarray],
    save_path: str | Path | None = None,
) -> None:
    """Plot stacked area chart of reward components over time.

    Args:
        reward_components: Dict mapping component name to array of values per step.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = np.arange(len(next(iter(reward_components.values()))))

    for name, values in reward_components.items():
        ax.plot(steps, values, label=name, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Components Breakdown")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_safety_zone(
    robot_positions: np.ndarray,
    tissue_boundary: np.ndarray,
    safety_margin_mm: float,
    save_path: str | Path | None = None,
) -> None:
    """Plot 2D scatter of robot tip positions relative to tissue boundary.

    Args:
        robot_positions: Robot tip positions, shape (T, 3).
        tissue_boundary: Tissue boundary position, shape (3,).
        safety_margin_mm: Safety exclusion zone radius.
        save_path: If provided, save figure to this path.
    """
    distances = np.linalg.norm(robot_positions - tissue_boundary, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = np.where(
        distances < safety_margin_mm,
        "red",
        np.where(distances < safety_margin_mm * 2, "orange", "green"),
    )
    ax.scatter(range(len(distances)), distances, c=colors, s=2, alpha=0.6)
    ax.axhline(y=safety_margin_mm, color="red", linestyle="--", label="Safety margin")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to tissue (mm)")
    ax.set_title("Safety Zone Visualization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
