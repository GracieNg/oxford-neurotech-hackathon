from dataclasses import dataclass, field

# Frequency bands in Hz, [low, high)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Default Unicorn Hybrid Black electrode positions (check against your cap!)
UNICORN_CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]


@dataclass
class PipelineConfig:
    fs: float = 250.0
    n_channels: int = 8
    channel_names: list = field(default_factory=lambda: list(UNICORN_CHANNELS))
    bandpass: tuple = (1.0, 40.0)
    filter_order: int = 4
    notch_hz: float | None = 50.0   # UK mains; None to disable
    notch_q: float = 30.0
    window_s: float = 2.0           # analysis window
    step_s: float = 1.0             # update rate
    bands: dict = field(default_factory=lambda: dict(BANDS))
    # quality thresholds (microvolts, after filtering)
    max_std_uv: float = 100.0       # above this -> likely bad contact / artefact
    min_std_uv: float = 0.5         # below this -> flat / disconnected
    max_ptp_uv: float = 500.0       # peak-to-peak spike threshold (blinks, movement)

    @property
    def window_n(self) -> int:
        return int(round(self.window_s * self.fs))

    @property
    def step_n(self) -> int:
        return int(round(self.step_s * self.fs))
