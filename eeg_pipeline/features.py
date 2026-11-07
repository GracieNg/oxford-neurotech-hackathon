import numpy as np
from scipy.signal import welch


def welch_psd(x: np.ndarray, fs: float, nperseg: int | None = None):
    """x: (n_samples, n_channels) -> freqs, psd (n_channels, n_freqs) in uV^2/Hz."""
    n = x.shape[0]
    nperseg = min(nperseg or int(fs), n)
    return welch(x.T, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)


def band_powers(x: np.ndarray, fs: float, bands: dict, relative=True):
    """Return dict band -> (n_channels,) absolute power, plus '<band>_rel' if relative."""
    f, p = welch_psd(x, fs)
    df = f[1] - f[0]
    lo = min(b[0] for b in bands.values())
    hi = max(b[1] for b in bands.values())
    total = p[:, (f >= lo) & (f < hi)].sum(axis=1) * df
    out = {}
    for name, (a, b) in bands.items():
        bp = p[:, (f >= a) & (f < b)].sum(axis=1) * df
        out[name] = bp
        if relative:
            out[f"{name}_rel"] = bp / np.maximum(total, 1e-12)
    return out
