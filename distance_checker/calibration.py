"""Persist and load the camera focal-length calibration used for distance math."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "calibration.json"

# Average human iris diameter in millimeters. This is a well-established
# anthropometric constant (varies only slightly across adults, ~11.7mm +/- 0.5mm)
# which is why it's commonly used as the reference object for monocular
# distance estimation instead of asking the user to measure anything.
DEFAULT_IRIS_DIAMETER_MM = 11.7


@dataclass
class Calibration:
    focal_length_px: float
    known_distance_mm: float
    iris_diameter_px: float
    iris_diameter_mm: float = DEFAULT_IRIS_DIAMETER_MM
    created_at: float = 0.0

    @staticmethod
    def from_measurement(
        known_distance_mm: float,
        iris_diameter_px: float,
        iris_diameter_mm: float = DEFAULT_IRIS_DIAMETER_MM,
    ) -> "Calibration":
        focal_length_px = (iris_diameter_px * known_distance_mm) / iris_diameter_mm
        return Calibration(
            focal_length_px=focal_length_px,
            known_distance_mm=known_distance_mm,
            iris_diameter_px=iris_diameter_px,
            iris_diameter_mm=iris_diameter_mm,
            created_at=time.time(),
        )

    def distance_mm(self, iris_diameter_px: float) -> float:
        if iris_diameter_px <= 0:
            return float("nan")
        return (self.iris_diameter_mm * self.focal_length_px) / iris_diameter_px


def load(path: Path = DEFAULT_CALIBRATION_PATH) -> Calibration | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return Calibration(**data)
    except Exception:
        return None


def save(calibration: Calibration, path: Path = DEFAULT_CALIBRATION_PATH) -> None:
    path = Path(path)
    path.write_text(json.dumps(asdict(calibration), indent=2))
