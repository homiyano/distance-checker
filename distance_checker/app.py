"""Real-time webcam distance checker.

Uses MediaPipe's Face Landmarker (a deep neural network that predicts 478
3D face + iris landmarks per frame) to locate the iris in each eye, then
estimates distance-to-screen with the classic pinhole-camera "known object
size" formula:

    distance = (real_world_size * focal_length_px) / size_in_pixels

The iris is used as the reference object because its diameter is one of the
most anatomically consistent measurements on the human body (~11.7mm across
adults), which makes it a reliable stand-in for a physical calibration
target. A one-time calibration step (press 'c' at a known distance) solves
for your specific webcam's focal length in pixels.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from . import calibration as calib
from .geometry import average_iris_diameter_px
from .model import DEFAULT_MODEL_PATH, ensure_model

WINDOW_NAME = "Distance Checker"

COLOR_OK = (60, 200, 60)
COLOR_WARN = (0, 200, 255)
COLOR_BAD = (0, 0, 255)
COLOR_TEXT_BG = (20, 20, 20)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time deep-learning webcam distance checker")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument(
        "--calib-distance-cm",
        type=float,
        default=50.0,
        help="Distance (cm) you'll be sitting at when you press 'c' to calibrate (default: 50)",
    )
    p.add_argument("--too-close-cm", type=float, default=40.0, help="Warn threshold for too close (cm)")
    p.add_argument("--too-far-cm", type=float, default=75.0, help="Warn threshold for too far (cm)")
    p.add_argument(
        "--iris-mm",
        type=float,
        default=calib.DEFAULT_IRIS_DIAMETER_MM,
        help="Assumed real-world iris diameter in mm (default: 11.7, the adult average)",
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to face_landmarker.task")
    p.add_argument(
        "--reset-calibration",
        action="store_true",
        help="Ignore/overwrite any saved calibration on startup",
    )
    p.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe camera indices 0-4, save a snapshot from each, then exit "
        "(useful when macOS Continuity Camera/OBS shifts which index is your real webcam)",
    )
    return p.parse_args()


def list_cameras(max_index: int = 5) -> None:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    out_dir = Path("camera_probe")
    out_dir.mkdir(exist_ok=True)
    print(f"Probing camera indices 0-{max_index - 1}, saving snapshots to {out_dir}/ ...")
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            print(f"  [{idx}] could not open")
            cap.release()
            continue
        ok, frame = False, None
        for _ in range(15):
            ok, frame = cap.read()
            if ok:
                break
            time.sleep(0.1)
        cap.release()
        if not ok:
            print(f"  [{idx}] opened but no frame")
            continue
        mean_val = frame.mean()
        h, w = frame.shape[:2]
        out_path = out_dir / f"camera_{idx}.jpg"
        cv2.imwrite(str(out_path), frame)
        note = "looks black/blank" if mean_val < 1.0 else "has image content"
        print(f"  [{idx}] {w}x{h}, mean pixel {mean_val:.2f} ({note}) -> saved {out_path}")
    print("Open the saved .jpg files and find the one showing your face; "
          "run again with --camera <that index>.")


def draw_label(frame, text, org, color=COLOR_TEXT_BG, text_color=(255, 255, 255), scale=0.7, thickness=2):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 6, y - th - 8), (x + tw + 6, y + baseline + 4), color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)


def status_for_distance(distance_cm: float, too_close_cm: float, too_far_cm: float):
    if distance_cm < too_close_cm:
        return f"TOO CLOSE  ({distance_cm:.0f} cm)", COLOR_BAD
    if distance_cm > too_far_cm:
        return f"TOO FAR  ({distance_cm:.0f} cm)", COLOR_WARN
    return f"GOOD DISTANCE  ({distance_cm:.0f} cm)", COLOR_OK


def main() -> None:
    args = parse_args()

    if args.list_cameras:
        list_cameras()
        return

    model_path = ensure_model(args.model)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    # On macOS the camera can report "opened" before the AVFoundation session
    # has actually started streaming, so the first handful of reads can fail
    # even when everything is fine. Retry briefly before giving up for real.
    ok = False
    for _ in range(30):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.1)
    if not ok:
        cap.release()
        raise SystemExit(
            "Could not read any frames from the camera.\n"
            "This is almost always a macOS Camera permission issue, not a code bug:\n"
            "  1. Open System Settings -> Privacy & Security -> Camera.\n"
            "  2. Make sure your terminal app (Terminal/iTerm/VS Code/etc.) is listed and enabled.\n"
            "     If it's not listed at all, run this script again from that terminal app so\n"
            "     macOS shows the permission prompt, then click Allow.\n"
            "  3. Also make sure no other app (Zoom, FaceTime, another Python process) is\n"
            "     currently holding the camera.\n"
            "  4. If you just changed the permission, fully quit and reopen the terminal app."
        )

    active_calibration = None if args.reset_calibration else calib.load()
    if active_calibration:
        print(f"Loaded saved calibration (focal length: {active_calibration.focal_length_px:.1f}px)")
    else:
        print("No calibration found. Sit at the distance given by --calib-distance-cm and press 'c'.")

    start_time = time.perf_counter()
    prev_frame_time = start_time
    fps = 0.0
    latest_diameter_px = None

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed; exiting.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            latest_diameter_px = None
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                diameter_px, left_c, right_c = average_iris_diameter_px(landmarks, w, h)
                latest_diameter_px = diameter_px

                for cx, cy in (left_c, right_c):
                    cv2.circle(frame, (int(cx), int(cy)), max(2, int(diameter_px / 2)), COLOR_OK, 1, cv2.LINE_AA)

                if active_calibration:
                    distance_mm = active_calibration.distance_mm(diameter_px)
                    distance_cm = distance_mm / 10.0
                    label, color = status_for_distance(distance_cm, args.too_close_cm, args.too_far_cm)
                    draw_label(frame, label, (20, 40), color=color)
                else:
                    draw_label(
                        frame,
                        f"Not calibrated - sit at {args.calib_distance_cm:.0f}cm and press 'c'",
                        (20, 40),
                        color=COLOR_WARN,
                    )
            else:
                draw_label(frame, "No face detected", (20, 40), color=COLOR_BAD)

            now = time.perf_counter()
            dt = now - prev_frame_time
            prev_frame_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            draw_label(frame, f"FPS: {fps:.0f}", (20, h - 20), scale=0.6, thickness=1)
            draw_label(frame, "c: calibrate  r: reset  q: quit", (20, h - 50), scale=0.5, thickness=1)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            elif key == ord("c"):
                if latest_diameter_px:
                    active_calibration = calib.Calibration.from_measurement(
                        known_distance_mm=args.calib_distance_cm * 10.0,
                        iris_diameter_px=latest_diameter_px,
                        iris_diameter_mm=args.iris_mm,
                    )
                    calib.save(active_calibration)
                    print(
                        f"Calibrated at {args.calib_distance_cm:.0f}cm "
                        f"(focal length: {active_calibration.focal_length_px:.1f}px). Saved to calibration.json."
                    )
                else:
                    print("Can't calibrate: no face/iris detected right now.")
            elif key == ord("r"):
                active_calibration = None
                print("Calibration cleared for this session (calibration.json left untouched).")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
