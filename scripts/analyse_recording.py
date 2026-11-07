"""Signal-quality + band-power report for Unicorn CSV recordings.

    python scripts/analyse_recording.py UnicornRecorder_*.csv --out reports/
    python scripts/analyse_recording.py --synthetic --out reports/
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_pipeline import (PipelineConfig, filter_offline, load_unicorn_csv, process_recording,
                          synthetic_eeg, welch_psd)


def report(rec, cfg, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    name = Path(rec.source).stem or "synthetic"
    trim = int(2 * cfg.fs)  # amp start-up transient
    y = filter_offline(rec.data[trim:], cfg)
    times = rec.times[trim:]
    frames = process_recording(rec, cfg)
    n_win = len(y) // cfg.window_n
    win_std = y[:n_win * cfg.window_n].reshape(n_win, cfg.window_n, -1).std(axis=1)
    std = np.median(win_std, axis=0)  # robust: typical 2 s window amplitude per channel
    good_frac = (np.mean([[q == "good" for q in f["quality"]] for f in frames], axis=0)
                 if frames else np.zeros(cfg.n_channels))
    f, p = welch_psd(y, cfg.fs, nperseg=int(2 * cfg.fs))

    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    mid = len(y) // 2
    seg = slice(max(0, mid - int(5 * cfg.fs)), min(len(y), mid + int(5 * cfg.fs)))
    off = np.arange(cfg.n_channels) * max(np.median(std) * 6, 1)
    ax[0].plot(times[seg], np.clip(y[seg], -3 * off[1], 3 * off[1]) + off, lw=0.6)
    ax[0].set_yticks(off, cfg.channel_names)
    ax[0].set_title(f"{name}: 10 s from middle, 1–40 Hz + 50 Hz notch (spacing {off[1]:.0f} µV)")
    ax[1].semilogy(f, p.T, lw=0.8)
    ax[1].set_xlim(0, 45); ax[1].set_xlabel("Hz"); ax[1].set_ylabel("µV²/Hz")
    ax[1].legend(cfg.channel_names, ncol=4, fontsize=7); ax[1].set_title("Welch PSD")
    if frames:
        t = [fr["t"] for fr in frames]
        ax[2].plot(t, [fr["summary"]["posterior_alpha_rel"] or np.nan for fr in frames], label="posterior α rel")
        ax[2].plot(t, [fr["summary"]["n_good"] / cfg.n_channels for fr in frames], label="fraction good ch")
        ax[2].set_ylim(0, 1); ax[2].set_xlabel("s"); ax[2].legend()
        ax[2].set_title("Live-pipeline output (2 s window, 1 s step)")
    fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=110); plt.close(fig)

    summary = {
        "file": name, "duration_s": round(rec.duration_s, 1),
        "median_window_std_uv": dict(zip(cfg.channel_names, np.round(std, 1).tolist())),
        "fraction_windows_good": dict(zip(cfg.channel_names, np.round(good_frac, 2).tolist())),
        "triggers": int(np.count_nonzero(np.diff(rec.trigger) != 0)),
        "verdict": ("good (typical EEG range)" if np.median(std) < 40 else
                    "marginal — some channels noisy, re-seat electrodes" if np.median(std) < cfg.max_std_uv * 3 else
                    "NOT physiological — check electrode contact / reference / units"),
    }
    with open(out / f"{name}_features.json", "w") as fh:
        json.dump(frames, fh)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    cfg = PipelineConfig()
    recs = [synthetic_eeg(60, cfg, blink_times=(10, 30))] if a.synthetic else []
    recs += [load_unicorn_csv(fn, cfg) for fn in a.files]
    summaries = [report(r, cfg, Path(a.out)) for r in recs]
    for s in summaries:
        print(json.dumps(s))
    with open(Path(a.out) / "summary.json", "w") as fh:
        json.dump(summaries, fh, indent=1)


if __name__ == "__main__":
    main()
