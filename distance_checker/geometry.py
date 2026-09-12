"""Pixel-space geometry helpers derived from Face Landmarker output."""

from __future__ import annotations

import cv2
import numpy as np

# Landmark indices for the iris, present when the Face Landmarker's built-in
# attention-mesh refinement adds the 10 iris points (468-477) on top of the
# base 468-point face mesh.
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def _iris_diameter_px(landmarks, indices, frame_w: int, frame_h: int) -> tuple[float, tuple[float, float]]:
    pts = np.array(
        [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in indices],
        dtype=np.float32,
    )
    (cx, cy), radius = cv2.minEnclosingCircle(pts)
    return radius * 2.0, (cx, cy)


def average_iris_diameter_px(landmarks, frame_w: int, frame_h: int):
    """Returns (avg_diameter_px, left_center, right_center) for one detected face."""
    left_d, left_c = _iris_diameter_px(landmarks, LEFT_IRIS, frame_w, frame_h)
    right_d, right_c = _iris_diameter_px(landmarks, RIGHT_IRIS, frame_w, frame_h)
    return (left_d + right_d) / 2.0, left_c, right_c
