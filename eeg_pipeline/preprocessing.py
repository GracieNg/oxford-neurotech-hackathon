"""Filtering. StreamingFilter is causal + stateful (no look-ahead, no window-edge
artefacts) so it is safe for live use; filter_offline is zero-phase for analysis."""
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos

from .config import PipelineConfig


def design_sos(cfg: PipelineConfig) -> np.ndarray:
    sos = butter(cfg.filter_order, cfg.bandpass, btype="band", fs=cfg.fs, output="sos")
    if cfg.notch_hz and cfg.notch_hz < cfg.fs / 2:
        b, a = iirnotch(cfg.notch_hz, cfg.notch_q, fs=cfg.fs)
        sos = np.vstack([sos, tf2sos(b, a)])
    return sos


class StreamingFilter:
    """Feed arbitrary-sized chunks of (n_samples, n_channels); returns filtered chunk."""

    def __init__(self, cfg: PipelineConfig):
        self.sos = design_sos(cfg)
        self.n_channels = cfg.n_channels
        self.zi = None

    def reset(self):
        self.zi = None

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.atleast_2d(np.asarray(chunk, float))
        if self.zi is None:
            # initialise to the first sample to avoid a huge DC-offset transient
            base = sosfilt_zi(self.sos)[:, :, None]
            self.zi = base * chunk[0][None, None, :]
        y, self.zi = sosfilt(self.sos, chunk, axis=0, zi=self.zi)
        return y


def filter_offline(x: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Zero-phase filter of a whole recording (n_samples, n_channels)."""
    x = np.asarray(x, float)
    return sosfiltfilt(design_sos(cfg), x - np.median(x, axis=0), axis=0)
