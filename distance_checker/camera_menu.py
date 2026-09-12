"""A small on-screen menu for picking which camera device to use.

Deliberately avoids opening (VideoCapture-ing) every device just to list
them: on macOS that would spin up a live session on each one, which is
exactly what triggers the "Allow this Mac to use your iPhone as a camera?"
handoff prompt for Continuity Camera devices. Instead we ask the OS for the
device list first (metadata only) and only open the single device the user
actually picks.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass

import cv2
import numpy as np

IPHONE_MODEL_ID_RE = re.compile(r"^iPhone\d+,\d+$")

BG = (26, 22, 20)
CARD_BG = (42, 36, 32)
CARD_BG_HOVER = (64, 54, 46)
CARD_BORDER = (70, 60, 54)
ACCENT = (60, 200, 255)
TEXT_PRIMARY = (240, 240, 240)
TEXT_MUTED = (150, 150, 150)
TAG_BUILTIN = (110, 220, 120)
TAG_WARN = (70, 150, 255)
TAG_VIRTUAL = (150, 150, 150)


@dataclass
class CameraOption:
    index: int
    name: str
    kind: str  # "builtin" | "iphone" | "virtual" | "unknown"

    @property
    def tag(self) -> str:
        return {
            "builtin": "Built-in",
            "iphone": "iPhone - may trigger a handoff prompt",
            "virtual": "Virtual camera",
            "unknown": "",
        }[self.kind]

    @property
    def tag_color(self):
        return {
            "builtin": TAG_BUILTIN,
            "iphone": TAG_WARN,
            "virtual": TAG_VIRTUAL,
            "unknown": TEXT_MUTED,
        }[self.kind]


def _classify(name: str, model_id: str) -> str:
    if IPHONE_MODEL_ID_RE.match(model_id or ""):
        return "iphone"
    if "virtual" in name.lower() or "obs" in name.lower():
        return "virtual"
    if "macbook" in name.lower() or "facetime" in name.lower() or "built-in" in name.lower():
        return "builtin"
    return "unknown"


def discover_cameras() -> list[CameraOption]:
    """List cameras without opening any of them, where possible."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["system_profiler", "SPCameraDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout
            data = json.loads(out).get("SPCameraDataType", [])
            options = []
            for i, entry in enumerate(data):
                name = entry.get("_name", f"Camera {i}")
                model_id = entry.get("spcamera_model-id", "")
                options.append(CameraOption(index=i, name=name, kind=_classify(name, model_id)))
            if options:
                return options
        except Exception:
            pass  # fall through to generic probing below

    # Non-macOS (or system_profiler failed): we can't get names without
    # opening devices, so just offer a handful of generic indices.
    return [CameraOption(index=i, name=f"Camera {i}", kind="unknown") for i in range(4)]


def _draw_rounded_rect(img, top_left, bottom_right, color, radius=14, thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(img, (cx, cy), radius, color, thickness)


def _draw_camera_icon(img, cx, cy, color):
    cv2.rectangle(img, (cx - 16, cy - 10), (cx + 10, cy + 10), color, -1, cv2.LINE_AA)
    pts = np.array([[cx + 10, cy - 5], [cx + 22, cy - 11], [cx + 22, cy + 11], [cx + 10, cy + 5]], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.circle(img, (cx - 3, cy), 5, BG, -1, cv2.LINE_AA)
    cv2.circle(img, (cx - 3, cy), 5, color, 1, cv2.LINE_AA)


def select_camera(options: list[CameraOption]) -> int | None:
    """Show a menu and return the chosen camera index, or None if cancelled."""
    if len(options) == 1:
        return options[0].index

    window = "Select Camera"
    card_h = 78
    card_gap = 14
    margin_x = 40
    top_y = 110
    width = 620
    height = top_y + len(options) * (card_h + card_gap) + 70

    state = {"hover": -1, "chosen": None}

    def on_mouse(event, x, y, flags, _):
        idx = _card_at(y, top_y, card_h, card_gap, len(options))
        state["hover"] = idx if margin_x <= x <= width - margin_x else -1
        if event == cv2.EVENT_LBUTTONDOWN and idx is not None and 0 <= idx < len(options):
            state["chosen"] = options[idx].index

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    default_idx = next((i for i, o in enumerate(options) if o.kind == "builtin"), 0)

    while state["chosen"] is None:
        frame = np.full((height, width, 3), BG, dtype=np.uint8)
        cv2.putText(frame, "Select a Camera", (margin_x, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, TEXT_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(frame, "Click a camera, or press its number", (margin_x, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_MUTED, 1, cv2.LINE_AA)

        for i, opt in enumerate(options):
            y1 = top_y + i * (card_h + card_gap)
            y2 = y1 + card_h
            hovered = state["hover"] == i
            recommended = i == default_idx and opt.kind == "builtin"
            fill = CARD_BG_HOVER if hovered else CARD_BG
            _draw_rounded_rect(frame, (margin_x, y1), (width - margin_x, y2), fill, radius=14)
            border_color = ACCENT if hovered or recommended else CARD_BORDER
            _draw_rounded_rect(frame, (margin_x, y1), (width - margin_x, y2), border_color, radius=14, thickness=2)

            _draw_camera_icon(frame, margin_x + 40, y1 + card_h // 2, TEXT_PRIMARY)

            cv2.putText(frame, f"{i + 1}", (margin_x + 14, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, opt.name, (margin_x + 80, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, TEXT_PRIMARY, 1, cv2.LINE_AA)
            if opt.tag:
                label = opt.tag + ("  (recommended)" if recommended else "")
                cv2.putText(frame, label, (margin_x + 80, y1 + 58), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, opt.tag_color, 1, cv2.LINE_AA)
            elif recommended:
                cv2.putText(frame, "recommended", (margin_x + 80, y1 + 58), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, ACCENT, 1, cv2.LINE_AA)

        cv2.putText(frame, "Esc to quit", (margin_x, height - 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, TEXT_MUTED, 1, cv2.LINE_AA)

        cv2.imshow(window, frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Esc
            cv2.destroyWindow(window)
            return None
        if ord("1") <= key <= ord("9"):
            n = key - ord("1")
            if n < len(options):
                state["chosen"] = options[n].index
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            return None

    cv2.destroyWindow(window)
    return state["chosen"]


def _card_at(y, top_y, card_h, card_gap, count):
    for i in range(count):
        y1 = top_y + i * (card_h + card_gap)
        y2 = y1 + card_h
        if y1 <= y <= y2:
            return i
    return None
