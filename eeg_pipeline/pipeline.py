"""Ring buffer + sliding windows -> features. Same object for live and offline."""
import json
import time

import numpy as np

from .config import PipelineConfig
from .features import band_powers
from .preprocessing import StreamingFilter
from .quality import channel_quality


class EEGPipeline:
    """push(chunk) as samples arrive; every `step_s` it emits a feature dict.

    Feature dict (JSON-serialisable, easy to send over a WebSocket to the app):
      {"t": seconds, "channels": [...], "quality": [...],
       "alpha": [...], "alpha_rel": [...], ..., "summary": {...}}
    """

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.filter = StreamingFilter(self.cfg)
        self.buf = np.zeros((self.cfg.window_n, self.cfg.n_channels))
        self.filled = 0
        self.since_emit = 0
        self.n_seen = 0

    def push(self, chunk: np.ndarray) -> list[dict]:
        y = self.filter(chunk)
        out = []
        for row in y:  # per-sample loop keeps emit timing exact; fast enough at 250 Hz
            self.buf = np.roll(self.buf, -1, axis=0)
            self.buf[-1] = row
            self.filled = min(self.filled + 1, self.cfg.window_n)
            self.n_seen += 1
            self.since_emit += 1
            # skip first window: filter transient settling
            if self.filled == self.cfg.window_n and self.since_emit >= self.cfg.step_n \
                    and self.n_seen >= 2 * self.cfg.window_n:
                self.since_emit = 0
                out.append(self._features(self.buf))
        return out

    def _features(self, w: np.ndarray) -> dict:
        cfg = self.cfg
        bp = band_powers(w, cfg.fs, cfg.bands)
        q = channel_quality(w, cfg)
        good = np.array([l == "good" for l in q])
        feat = {"t": round(self.n_seen / cfg.fs, 3), "channels": cfg.channel_names, "quality": q}
        for k, v in bp.items():
            feat[k] = [round(float(a), 4) for a in v]
        # summary over good channels only (posterior for alpha = classic eyes-closed marker)
        post = np.array([c.upper().startswith(("O", "PO", "P")) for c in cfg.channel_names])
        sel = good & post if (good & post).any() else good
        feat["summary"] = {
            "n_good": int(good.sum()),
            "posterior_alpha_rel": float(np.mean(bp["alpha_rel"][sel])) if sel.any() else None,
            "beta_alpha_ratio": float(np.mean(bp["beta"][good] / np.maximum(bp["alpha"][good], 1e-12)))
            if good.any() else None,
        }
        return feat


def process_recording(rec, cfg: PipelineConfig | None = None, chunk=25) -> list[dict]:
    """Run a recording through the live pipeline (replay), return all feature frames."""
    pipe = EEGPipeline(cfg)
    frames = []
    for i in range(0, rec.data.shape[0], chunk):
        frames.extend(pipe.push(rec.data[i:i + chunk]))
    return frames


def run_live(inlet, cfg: PipelineConfig | None = None, emit=print):
    """Live loop: inlet.chunks() -> pipeline -> emit(json_string)."""
    pipe = EEGPipeline(cfg)
    for chunk, _ts in inlet.chunks():
        for feat in pipe.push(chunk):
            emit(json.dumps(feat))
