"""Handles fetching and caching the MediaPipe Face Landmarker model asset."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    sys.stdout.write(f"\rDownloading face landmarker model... {pct}%")
    sys.stdout.flush()
    if downloaded >= total_size:
        sys.stdout.write("\n")


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the model asset if it isn't already cached locally."""
    model_path = Path(model_path)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Model not found at {model_path}, fetching from Google's model store...")
    tmp_path = model_path.with_suffix(".task.part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path, reporthook=_report_progress)
        tmp_path.rename(model_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return model_path
