# Distance Checker

Real-time webcam tool that watches how far you're sitting from your laptop
screen and warns you if you're too close (or too far), using deep-learning
face/iris tracking — no manual measuring, no special hardware.

## How it works

- **MediaPipe Face Landmarker** (a neural network, Google's current
  production model, successor to the legacy FaceMesh solution) runs on every
  webcam frame and outputs 478 3D landmarks per face, including 10 iris
  points per eye.
- The **iris diameter in pixels** is measured each frame (`cv2.minEnclosingCircle`
  over the iris landmark cluster). The human iris is anatomically consistent
  in size (~11.7mm across adults), so it acts as a built-in "ruler" — no need
  for an ArUco marker, credit card, or checkerboard.
- A one-time **calibration** step (sit at a known distance, press `c`) solves
  for your webcam's focal length in pixels using the pinhole camera equation:

  ```
  distance = (real_iris_diameter_mm * focal_length_px) / iris_diameter_px
  ```

  The calibration is saved to `calibration.json` so you only need to do it once.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The face landmarker model (`models/face_landmarker.task`, ~3.7MB) is
downloaded automatically on first run and cached locally.

## Run

```bash
python main.py
```

On first launch, macOS will prompt for camera access for your terminal app —
allow it (System Settings → Privacy & Security → Camera if you missed the
prompt).

**Controls:**
- `c` — calibrate: sit at the distance given by `--calib-distance-cm`
  (default 50cm / ~20in) and press this once. Only needs to be done once ever.
- `r` — clear calibration for the current session
- `q` / `Esc` — quit

**Useful flags:**

```bash
python main.py --calib-distance-cm 50 --too-close-cm 40 --too-far-cm 75
python main.py --camera 1          # use a different camera index
python main.py --reset-calibration # ignore saved calibration.json on startup
```

## Notes

- Tested with `mediapipe==0.10.30`. Newer mediapipe (1.0.x) currently has a
  GPU/Metal service crash in the face-detection graph on Apple Silicon
  (`Check failed: service_ Service is unavailable` in
  `TensorsToDetectionsCalculator::Open()`) — stick to 0.10.30 until that's
  fixed upstream.
- Distance accuracy depends on the calibration step; the assumed iris size
  (`--iris-mm`, default 11.7mm) is an adult average and can be tuned per
  person for extra precision.
