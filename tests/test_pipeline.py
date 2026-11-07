import numpy as np

from eeg_pipeline import (PipelineConfig, StreamingFilter, band_powers, channel_quality,
                          filter_offline, process_recording, synthetic_eeg)

CFG = PipelineConfig()


def test_band_power_finds_10hz():
    t = np.arange(int(4 * CFG.fs)) / CFG.fs
    x = np.tile(np.sin(2 * np.pi * 10 * t)[:, None], (1, 8)) * 10
    bp = band_powers(x, CFG.fs, CFG.bands)
    assert np.all(bp["alpha_rel"] > 0.9)


def test_notch_removes_mains():
    t = np.arange(int(10 * CFG.fs)) / CFG.fs
    x = np.tile((np.sin(2 * np.pi * 50 * t) * 100)[:, None], (1, 8))
    y = filter_offline(x, CFG)
    assert y[500:-500].std() < 2.0


def test_streaming_filter_chunk_invariant():
    rec = synthetic_eeg(10, CFG)
    a = StreamingFilter(CFG)(rec.data)
    f = StreamingFilter(CFG)
    b = np.vstack([f(rec.data[i:i + 17]) for i in range(0, len(rec.data), 17)])
    assert np.allclose(a, b)


def test_streaming_is_causal():
    rec = synthetic_eeg(10, CFG)
    full = StreamingFilter(CFG)(rec.data)
    x2 = rec.data.copy()
    x2[1500:] += 1000  # change the future
    part = StreamingFilter(CFG)(x2)
    assert np.allclose(full[:1500], part[:1500])


def test_alpha_block_detected_end_to_end():
    rec = synthetic_eeg(60, CFG, alpha_blocks=((20, 40),), seed=1)
    frames = process_recording(rec, CFG)
    t = np.array([f["t"] for f in frames])
    a = np.array([f["summary"]["posterior_alpha_rel"] for f in frames])
    inside = a[(t > 23) & (t < 40)].mean()
    outside = a[(t > 45) | (t < 18)].mean()
    assert inside > 2 * outside


def test_frame_rate_and_json():
    import json
    rec = synthetic_eeg(30, CFG)
    frames = process_recording(rec, CFG)
    # first emit after 2 windows (4 s) then 1 Hz
    assert 25 <= len(frames) <= 27
    json.dumps(frames[0])


def test_quality_flags():
    x = np.zeros((500, 8))
    x[:, 1] = np.random.default_rng(0).normal(0, 10, 500)
    x[:, 2] = np.random.default_rng(1).normal(0, 1000, 500)
    q = channel_quality(x, CFG)
    assert q[0] == "flat" and q[1] == "good" and q[2] == "noisy"
