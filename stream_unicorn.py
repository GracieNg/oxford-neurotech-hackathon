#!/usr/bin/env python3
"""stream_unicorn.py: ONE file to stream + record the g.tec Unicorn.

  * finds the Unicorn (LSL stream from the "Unicorn LSL" app, or UnicornPy directly on Windows)
  * records EEG + any experiment markers into ONE CSV:  recordings/unicorn_<time>.csv
        columns: lsl_time, EEG 1..8, marker
  * prints live signal quality every second
  * serves band-power features on ws://<this-laptop>:8765 for dashboard/index.html and mindcart.html

Install once:   pip install numpy scipy pylsl websockets
Run:            python stream_unicorn.py                 (auto: LSL first, then UnicornPy)
                python stream_unicorn.py --source unicornpy
                python stream_unicorn.py --fake          (no headset, for testing)
Stop:           Ctrl+C  (file is saved continuously)
"""
import argparse
import asyncio
import csv
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos, welch

FS = 250.0
N_EEG = 8
CH = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
WIN, STEP = int(2 * FS), int(1 * FS)


# ----------------------------------------------------------------- sources
class LSLSource:
    def __init__(self, timeout=8):
        from pylsl import StreamInlet, proc_clocksync, proc_dejitter, resolve_streams
        streams = resolve_streams(wait_time=timeout)
        eeg = [s for s in streams if s.channel_count() >= N_EEG and s.nominal_srate() > 0 and
               ("unicorn" in s.name().lower() or s.type().lower() in ("eeg", "data"))]
        if not eeg:
            names = [f"{s.name()} ({s.type()}, {s.channel_count()} ch)" for s in streams]
            raise RuntimeError("No Unicorn LSL stream. Open the 'Unicorn LSL' app, pick the headset, "
                               f"click Open then Start. Streams seen: {names or 'none'}")
        s = eeg[0]
        print(f"[source] LSL: {s.name()}  type={s.type()}  {s.channel_count()} ch @ {s.nominal_srate():.0f} Hz")
        self.inlet = StreamInlet(s, max_buflen=30, processing_flags=proc_clocksync | proc_dejitter)
        self.markers = None
        mk = [m for m in streams if m.type().lower() == "markers"]
        if mk:
            print(f"[source] markers: {mk[0].name()}")
            self.markers = StreamInlet(mk[0], processing_flags=proc_clocksync | proc_dejitter)
        self._find_markers_later = not mk

    def read(self):
        data, ts = self.inlet.pull_chunk(timeout=0.1, max_samples=256)
        mks = []
        if self._find_markers_later and int(time.time()) % 5 == 0:   # experiment may start later
            from pylsl import StreamInlet, resolve_byprop
            m = resolve_byprop("type", "Markers", timeout=0.05)
            if m:
                print(f"[source] markers stream appeared: {m[0].name()}")
                self.markers = StreamInlet(m[0]); self._find_markers_later = False
        if self.markers:
            md, mt = self.markers.pull_chunk(timeout=0.0)
            mks = [(t, str(d[0])) for d, t in zip(md, mt)]
        return (np.asarray(data, float)[:, :N_EEG] if data else np.empty((0, N_EEG))), np.asarray(ts), mks


class UnicornPySource:
    """Direct USB-dongle access via g.tec's UnicornPy (Windows, ships with Unicorn Suite)."""
    def __init__(self):
        import UnicornPy
        devs = UnicornPy.GetAvailableDevices(True)
        if not devs:
            raise RuntimeError("No Unicorn paired. Pair it in Unicorn Suite first.")
        print(f"[source] UnicornPy: {devs[0]}")
        self.dev = UnicornPy.Unicorn(devs[0])
        self.n_ch = self.dev.GetNumberOfAcquiredChannels()
        self.n = 25
        self.buf = bytearray(self.n * self.n_ch * 4)
        self.dev.StartAcquisition(False)
        self.t0 = time.time()
        self.count = 0

    def read(self):
        self.dev.GetData(self.n, self.buf, len(self.buf))
        x = np.frombuffer(self.buf, dtype=np.float32).reshape(self.n, self.n_ch)[:, :N_EEG].astype(float)
        ts = self.t0 + (self.count + np.arange(self.n)) / FS
        self.count += self.n
        return x, ts, []

    def close(self):
        self.dev.StopAcquisition()


class FakeSource:
    def __init__(self):
        print("[source] FAKE synthetic EEG (eyes closed every other 20 s)")
        self.k = 0; self.t0 = time.time(); self.rng = np.random.default_rng(0)

    def read(self):
        time.sleep(0.1)
        n = int((time.time() - self.t0) * FS) - self.k
        t = (self.k + np.arange(n)) / FS; self.k += n
        x = self.rng.normal(0, 8, (n, N_EEG))
        closed = (t // 20) % 2 == 1
        x[:, 4:] += (6 + 10 * closed[:, None]) * np.sin(2 * np.pi * 10 * t)[:, None]
        return x + 3000, self.t0 + t, []


# ----------------------------------------------------------------- processing
class Processor:
    def __init__(self):
        sos = butter(4, [1, 40], btype="band", fs=FS, output="sos")
        b, a = iirnotch(50, 30, fs=FS)
        self.sos = np.vstack([sos, tf2sos(b, a)])
        self.zi = None
        self.buf = np.zeros((WIN, N_EEG)); self.seen = 0; self.since = 0

    def push(self, x):
        if not len(x):
            return []
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        out = []
        for row in y:
            self.buf = np.roll(self.buf, -1, 0); self.buf[-1] = row
            self.seen += 1; self.since += 1
            if self.seen >= 2 * WIN and self.since >= STEP:
                self.since = 0; out.append(self.features())
        return out

    def features(self):
        w = self.buf
        f, p = welch(w.T, fs=FS, nperseg=int(FS), noverlap=int(FS) // 2)
        df = f[1] - f[0]
        tot = p[:, (f >= 1) & (f < 30)].sum(1) * df
        feat = {"t": round(self.seen / FS, 2), "channels": CH}
        for b, (lo, hi) in BANDS.items():
            bp = p[:, (f >= lo) & (f < hi)].sum(1) * df
            feat[b] = [round(float(v), 4) for v in bp]
            feat[b + "_rel"] = [round(float(v), 4) for v in bp / np.maximum(tot, 1e-12)]
        sd, ptp = w.std(0), np.ptp(w, 0)
        q = ["flat" if s < 0.5 else "noisy" if s > 100 else "artefact" if pp > 500 else "good" for s, pp in zip(sd, ptp)]
        feat["quality"] = q
        feat["std_uv"] = [round(float(s), 1) for s in sd]
        good = np.array([v == "good" for v in q]); post = np.array([c[0] in "PO" for c in CH])
        sel = good & post if (good & post).any() else good
        a, b_ = np.array(feat["alpha"]), np.array(feat["beta"])
        feat["summary"] = {"n_good": int(good.sum()),
                           "posterior_alpha_rel": float(np.mean(np.array(feat["alpha_rel"])[sel])) if sel.any() else None,
                           "beta_alpha_ratio": float(np.mean(b_[good] / np.maximum(a[good], 1e-12))) if good.any() else None}
        return feat


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "lsl", "unicornpy"], default="auto")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--name", default=None, help="file name (default unicorn_<date_time>.csv)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-ws", action="store_true")
    a = ap.parse_args()

    if a.fake:
        src = FakeSource()
    elif a.source == "unicornpy":
        src = UnicornPySource()
    else:
        try:
            src = LSLSource()
        except Exception as e:
            if a.source == "lsl":
                raise SystemExit(str(e))
            print(f"[source] LSL failed: {e}\n[source] trying UnicornPy ...")
            try:
                src = UnicornPySource()
            except Exception as e2:
                raise SystemExit(f"Could not connect to the Unicorn.\n  LSL: {e}\n  UnicornPy: {e2}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / (a.name or f"unicorn_{datetime.now():%Y%m%d_%H%M%S}.csv")
    fh = open(path, "w", newline=""); w = csv.writer(fh)
    w.writerow(["lsl_time"] + [f"EEG {i + 1}" for i in range(N_EEG)] + ["marker"])
    print(f"[record] writing ONE file: {path.resolve()}")

    proc, latest, clients = Processor(), {"f": None}, set()
    stop = threading.Event()

    def acquire():
        n, last_print, pending = 0, time.time(), []
        while not stop.is_set():
            x, ts, mks = src.read()
            pending += mks
            for row, t in zip(x, ts):
                m = ""
                while pending and pending[0][0] <= t:      # attach marker to nearest following sample
                    m = (m + ";" if m else "") + pending.pop(0)[1]
                w.writerow([f"{t:.6f}"] + [f"{v:.3f}" for v in row] + [m])
            n += len(x)
            for f in proc.push(x):
                latest["f"] = f
            if time.time() - last_print > 1 and latest["f"]:
                f = latest["f"]; last_print = time.time(); fh.flush()
                q = " ".join(f"{c}:{'ok' if s == 'good' else s[:5]:5s}" for c, s in zip(CH, f["quality"]))
                pa = f["summary"]["posterior_alpha_rel"]
                print(f"\r{n / FS:6.0f}s  good {f['summary']['n_good']}/8  postα {pa if pa is None else round(pa, 2)}  "
                      f"σ(µV) {f['std_uv']}  {q}", end="", flush=True)
        fh.flush()

    th = threading.Thread(target=acquire, daemon=True); th.start()

    async def serve():
        import websockets

        async def handler(ws):
            clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                clients.discard(ws)
        async with websockets.serve(handler, "0.0.0.0", a.port):
            print(f"[ws] dashboards: open dashboard/mindcart.html?ws=ws://localhost:{a.port}")
            last = None
            while True:
                await asyncio.sleep(0.2)
                f = latest["f"]
                if f is not None and f is not last:
                    last = f; msg = json.dumps(f)
                    for c in list(clients):
                        try:
                            await c.send(msg)
                        except Exception:
                            clients.discard(c)

    try:
        if a.no_ws:
            while th.is_alive():
                time.sleep(0.5)
        else:
            asyncio.run(serve())
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); th.join(timeout=2); fh.close()
        if hasattr(src, "close"):
            src.close()
        print(f"\n[record] saved {path.resolve()}")


if __name__ == "__main__":
    main()
