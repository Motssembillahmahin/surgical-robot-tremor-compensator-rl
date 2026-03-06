"""FFT-based tremor frequency analysis utilities."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft, fftfreq


def compute_dominant_frequency(
    signal: np.ndarray,
    sample_rate_hz: float,
    freq_range: tuple[float, float] = (3.0, 12.0),
) -> float:
    """Detect the dominant tremor frequency in a signal using FFT.

    Args:
        signal: 1D time-domain signal array.
        sample_rate_hz: Sampling rate in Hz.
        freq_range: (min_hz, max_hz) band to search for dominant frequency.

    Returns:
        Dominant frequency in Hz within the specified range.
    """
    n = len(signal)
    if n < 2:
        return 0.0

    yf = np.abs(fft(signal))
    xf = fftfreq(n, d=1.0 / sample_rate_hz)

    # Only consider positive frequencies within the tremor band
    mask = (xf >= freq_range[0]) & (xf <= freq_range[1])
    if not np.any(mask):
        return 0.0

    filtered_magnitudes = yf[mask]
    filtered_freqs = xf[mask]

    dominant_idx = np.argmax(filtered_magnitudes)
    return float(filtered_freqs[dominant_idx])


def compute_tremor_rejection_ratio(
    raw_signal: np.ndarray,
    compensated_signal: np.ndarray,
    sample_rate_hz: float,
    freq_range: tuple[float, float] = (3.0, 12.0),
) -> float:
    """Compute tremor rejection ratio in dB.

    Measures how much tremor power was removed by compensation.

    Args:
        raw_signal: Original signal with tremor (1D).
        compensated_signal: Signal after compensation (1D).
        sample_rate_hz: Sampling rate in Hz.
        freq_range: Tremor frequency band to measure.

    Returns:
        Rejection ratio in dB. Higher is better.
    """
    n = len(raw_signal)
    if n < 2:
        return 0.0

    xf = fftfreq(n, d=1.0 / sample_rate_hz)
    mask = (xf >= freq_range[0]) & (xf <= freq_range[1])

    raw_power = np.sum(np.abs(fft(raw_signal))[mask] ** 2)
    comp_power = np.sum(np.abs(fft(compensated_signal))[mask] ** 2)

    if comp_power < 1e-12:
        return 60.0  # Cap at 60 dB if tremor is fully removed

    return float(10 * np.log10(raw_power / comp_power))


def low_pass_filter(
    signal: np.ndarray,
    cutoff_hz: float,
    sample_rate_hz: float,
) -> np.ndarray:
    """Simple frequency-domain low-pass filter.

    Args:
        signal: 1D input signal.
        cutoff_hz: Cutoff frequency in Hz.
        sample_rate_hz: Sampling rate in Hz.

    Returns:
        Filtered signal of the same shape.
    """
    n = len(signal)
    if n < 2:
        return signal.copy()

    yf = fft(signal)
    xf = fftfreq(n, d=1.0 / sample_rate_hz)

    yf[np.abs(xf) > cutoff_hz] = 0
    return np.real(np.fft.ifft(yf)).astype(signal.dtype)
