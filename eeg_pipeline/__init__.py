"""Reusable EEG processing pipeline for the g.tec Unicorn Hybrid Black (8 ch, 250 Hz).

Works on recorded files, synthetic data, or a live stream (LSL), so the team can
develop without the headset and plug the same code into the live demo later.
"""
from .config import PipelineConfig, BANDS
from .io import load_unicorn_csv, synthetic_eeg, Recording
from .preprocessing import StreamingFilter, filter_offline
from .features import band_powers, welch_psd
from .quality import channel_quality
from .pipeline import EEGPipeline, process_recording

__all__ = [
    "PipelineConfig", "BANDS", "load_unicorn_csv", "synthetic_eeg", "Recording",
    "StreamingFilter", "filter_offline", "band_powers", "welch_psd",
    "channel_quality", "EEGPipeline", "process_recording",
]
