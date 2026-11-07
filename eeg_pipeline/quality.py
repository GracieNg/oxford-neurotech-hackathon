import numpy as np

from .config import PipelineConfig


def channel_quality(x_filt: np.ndarray, cfg: PipelineConfig) -> list[str]:
    """Per-channel label for a filtered window: 'good' | 'flat' | 'noisy' | 'artefact'."""
    std = x_filt.std(axis=0)
    ptp = np.ptp(x_filt, axis=0)
    labels = []
    for s, p in zip(std, ptp):
        if s < cfg.min_std_uv:
            labels.append("flat")
        elif s > cfg.max_std_uv:
            labels.append("noisy")
        elif p > cfg.max_ptp_uv:
            labels.append("artefact")
        else:
            labels.append("good")
    return labels
