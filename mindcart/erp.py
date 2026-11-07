"""Event-related potentials: P300 and LPP.

Both are positive waves over centro-parietal scalp after a stimulus:
  P300 (P3b)  250-500 ms, max at Pz   - attention / "this is relevant"
  LPP         500-1000 ms, Pz + Cz    - sustained motivational / emotional significance (interest, liking)

ERP processing differs from the band-power pipeline: slow waves need a 0.1 Hz high-pass (not 1 Hz),
epochs are time-locked to the image-onset marker, baseline-corrected, artefact-rejected and averaged.

    python -m mindcart.erp sessions/P01                 # product task: liked vs disliked ERP plot
    python -m mindcart.erp sessions/OB01 --oddball      # oddball validation: target vs standard P300
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

FS = 250.0
CH = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
EPOCH = (-0.2, 1.0)          # s around onset
BASELINE = (-0.2, 0.0)
P300_WIN = (0.25, 0.50)
LPP_WIN = (0.50, 1.00)
REJECT_UV = 100.0            # |amplitude| limit in any ERP channel after baseline
ERP_CH = ["Fz", "Cz", "Pz"]  # channels used for rejection (blinks show at Fz)


def erp_filter(x, fs=FS):
    """0.1-30 Hz band-pass + 50 Hz notch, zero-phase (offline). Keeps slow ERP components intact."""
    sos = butter(2, [0.1, 30], btype="band", fs=fs, output="sos")
    b, a = iirnotch(50, 30, fs=fs)
    sos = np.vstack([sos, tf2sos(b, a)])
    return sosfiltfilt(sos, x - np.median(x, axis=0), axis=0)


def epochs(y, t, onsets, fs=FS, win=EPOCH, base=BASELINE):
    """y (n, ch) filtered, t (n,) timestamps, onsets list -> (n_trials, n_times, ch), times, ok mask."""
    n0, n1 = int(round(win[0] * fs)), int(round(win[1] * fs))
    times = np.arange(n0, n1) / fs
    b = (times >= base[0]) & (times < base[1])
    out, ok = [], []
    rej_idx = [CH.index(c) for c in ERP_CH]
    for on in onsets:
        i0 = int(np.searchsorted(t, on))
        if i0 + n0 < 0 or i0 + n1 > len(y):
            out.append(np.full((n1 - n0, y.shape[1]), np.nan)); ok.append(False); continue
        e = y[i0 + n0:i0 + n1].copy()
        e -= e[b].mean(axis=0)
        out.append(e)
        ok.append(bool(np.abs(e[:, rej_idx]).max() < REJECT_UV))
    return np.asarray(out), times, np.asarray(ok)


def window_mean(ep, times, win, ch_names):
    m = (times >= win[0]) & (times < win[1])
    idx = [CH.index(c) for c in ch_names]
    return ep[:, m][:, :, idx].mean(axis=(1, 2))


def erp_features(ep, times, ok):
    """Per-trial P300/LPP mean amplitudes (µV). NaN for rejected trials."""
    f = {
        "f_p300_pz": window_mean(ep, times, P300_WIN, ["Pz"]),
        "f_p300_cz": window_mean(ep, times, P300_WIN, ["Cz"]),
        "f_lpp_pz": window_mean(ep, times, LPP_WIN, ["Pz"]),
        "f_lpp_cpz": window_mean(ep, times, LPP_WIN, ["Cz", "Pz"]),
    }
    for k in f:
        f[k] = np.where(ok, f[k], np.nan)
    return f


def load(sdir):
    eeg = pd.read_csv(Path(sdir) / "eeg.csv")
    mk = pd.read_csv(Path(sdir) / "markers.csv", dtype={"marker": str})
    return eeg["lsl_time"].to_numpy(), eeg.drop(columns="lsl_time").to_numpy(float), mk


def plot_erp(times, groups, title, path, note=""):
    """groups: {label: (epochs, color)} -> Pz and Cz averages with ±SEM and shaded P300/LPP windows."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    ms = times * 1000
    for ax, ch in zip(axes, ["Pz", "Cz"]):
        ax.axvspan(P300_WIN[0] * 1000, P300_WIN[1] * 1000, color="#3e4fd8", alpha=.07)
        ax.axvspan(LPP_WIN[0] * 1000, LPP_WIN[1] * 1000, color="#12805c", alpha=.07)
        ax.axvline(0, color="#98a2b3", lw=1); ax.axhline(0, color="#98a2b3", lw=.8)
        for lab, (ep, col) in groups.items():
            if len(ep) == 0:
                continue
            d = ep[:, :, CH.index(ch)]
            mu, se = np.nanmean(d, 0), np.nanstd(d, 0) / np.sqrt(max(1, len(d)))
            ax.plot(ms, mu, color=col, lw=2, label=f"{lab} (n={len(d)})")
            ax.fill_between(ms, mu - se, mu + se, color=col, alpha=.18)
        ax.set_title(ch, loc="left", fontweight="bold"); ax.set_xlabel("ms after image onset")
        ax.text(P300_WIN[0] * 1000 + 5, ax.get_ylim()[1] * .92 if ax.get_ylim()[1] > 0 else 1, "P300", color="#3e4fd8", fontsize=9)
        ax.text(LPP_WIN[0] * 1000 + 5, ax.get_ylim()[1] * .92 if ax.get_ylim()[1] > 0 else 1, "LPP", color="#12805c", fontsize=9)
    axes[0].set_ylabel("µV (positive up)"); axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(title + (f"\n{note}" if note else ""), x=.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def product_erp(sdir, like="extreme"):
    from .features import make_labels
    t, x, mk = load(sdir)
    rows = {}
    for lt, m in zip(mk["lsl_time"], mk["marker"]):
        p = m.split("|")
        if p[0] == "img":
            rows.setdefault(int(p[1]), {})["onset"] = lt
        elif p[0] == "rate":
            rows.setdefault(int(p[1]), {})["rating"] = float(p[3]) if p[3] not in ("", "None") else np.nan
    tr = pd.DataFrame([{"trial": k, **v} for k, v in sorted(rows.items()) if "onset" in v])
    y = erp_filter(x)
    ep, times, ok = epochs(y, t, tr["onset"].to_numpy())
    lab = make_labels(tr["rating"].to_numpy(float), like)
    f = erp_features(ep, times, ok)
    like_m, dis_m = ok & (lab == 1), ok & (lab == 0)
    d = {k: (np.nanmean(v[like_m]) - np.nanmean(v[dis_m])) for k, v in f.items()}
    plot_erp(times, {"liked": (ep[like_m], "#12805c"), "not liked": (ep[dis_m], "#98a2b3")},
             f"{Path(sdir).name}: ERP by preference",
             Path(sdir) / "erp.png",
             f"{ok.sum()}/{len(ok)} trials clean · liked − not liked: P300(Pz) {d['f_p300_pz']:+.1f} µV, LPP(Cz/Pz) {d['f_lpp_cpz']:+.1f} µV")
    return d, ok


def oddball_erp(sdir):
    from scipy.stats import ttest_ind
    t, x, mk = load(sdir)
    ev = [(lt, m.split("|")[2]) for lt, m in zip(mk["lsl_time"], mk["marker"]) if m.startswith("odd|")]
    on = np.array([e[0] for e in ev]); kind = np.array([e[1] for e in ev])
    y = erp_filter(x)
    ep, times, ok = epochs(y, t, on, win=(-0.2, 0.8))
    p3 = window_mean(ep, times, (0.28, 0.48), ["Pz"])   # peak-centred P300 window for the check
    tg, st = ok & (kind == "target"), ok & (kind == "standard")
    tstat, p = ttest_ind(p3[tg], p3[st], equal_var=False, alternative="greater")  # predicted: target > standard
    diff = p3[tg].mean() - p3[st].mean()
    verdict = "PASS: clear P300, the setup records ERPs" if (diff > 1.5 and p < .05) else "WEAK / FAIL: check markers, contact, or run more trials"
    plot_erp(times, {"target (rare)": (ep[tg], "#3e4fd8"), "standard": (ep[st], "#98a2b3")},
             f"{Path(sdir).name}: oddball P300 check", Path(sdir) / "oddball_p300.png",
             f"P300 at Pz (280-480 ms): target − standard = {diff:+.1f} µV (one-sided Welch t = {tstat:.1f}, p = {p:.1g}) → {verdict}")
    return diff, p, verdict, int(tg.sum()), int(st.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dirs", nargs="+")
    ap.add_argument("--oddball", action="store_true")
    ap.add_argument("--like", choices=["extreme", "median"], default="extreme")
    a = ap.parse_args()
    for s in a.session_dirs:
        if a.oddball:
            diff, p, v, nt, ns = oddball_erp(s)
            print(f"{s}: targets {nt}, standards {ns} | P300(Pz) diff {diff:+.1f} µV, p={p:.2g} | {v}\n  -> {Path(s) / 'oddball_p300.png'}")
        else:
            d, ok = product_erp(s, a.like)
            print(f"{s}: {ok.sum()}/{len(ok)} clean trials | liked − not liked: "
                  + ", ".join(f"{k[2:]} {v:+.2f} µV" for k, v in d.items()) + f"\n  -> {Path(s) / 'erp.png'}")


if __name__ == "__main__":
    main()
