"""Simple machine-readable benchmark records independent of a database service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    experiment_id: str
    model_name: str
    seed: int
    metrics: dict[str, Any]
    integration_time_s: float | None = None
    wavelength_nm: float | None = None
    filter_config: dict[str, Any] | None = None
    noise_config: dict[str, Any] | None = None
    calibration_config: dict[str, Any] | None = None
    forward_model_config: dict[str, Any] | None = None
    reconstruction_config: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    benchmark_family: str = "solar_gravitational_lens"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        return output
