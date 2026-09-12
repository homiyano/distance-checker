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

Before the video window opens, you'll see a **camera picker menu**. It lists
every camera macOS knows about (built-in webcam, iPhone via Continuity
Camera, virtual cams like OBS) purely from device metadata — it doesn't open
any of them yet, so it won't trigger the iPhone handoff prompt just for
showing the list. The built-in webcam is highlighted as recommended; iPhone
entries are flagged with a warning since selecting one will ask your phone
to confirm. Click a camera, or press its number key, to proceed.

**Controls (picker menu):**
- click a camera, or press its number key, to select it
- `Esc` — quit without opening the app

**Controls (main app):**
- `c` — calibrate: sit at the distance given by `--calib-distance-cm`
  (default 50cm / ~20in) and press this once. Only needs to be done once ever.
- `r` — clear calibration for the current session
- `q` / `Esc` — quit

**Useful flags:**

```bash
python main.py --calib-distance-cm 50 --too-close-cm 40 --too-far-cm 75
python main.py --camera 1          # skip the picker menu, use this index directly
python main.py --reset-calibration # ignore saved calibration.json on startup
```

## Continuity Camera / OBS sending a prompt to your phone

The picker menu (see above) only reads camera *metadata*, so just seeing the
menu never touches your iPhone. But if you deliberately click/select the
iPhone entry (or pass `--camera <its index>` directly), macOS will still ask
your phone to confirm the handoff — that's expected for that specific device,
not a bug. Just pick the entry marked **Built-in** instead.

To stop the iPhone from being offered as a webcam at all: on the iPhone, go
to **Settings → General → AirPlay & Handoff** and turn off **Continuity
Camera Webcam**.

If you ever need a lower-level fallback (e.g. the menu's device list doesn't
match reality), `python main.py --list-cameras` probes indices 0-4 directly
and saves a snapshot from each into `camera_probe/` — note this *does* open
every device, including the iPhone, so it can trigger that prompt.

## Notes

- Tested with `mediapipe==0.10.30`. Newer mediapipe (1.0.x) currently has a
  GPU/Metal service crash in the face-detection graph on Apple Silicon
  (`Check failed: service_ Service is unavailable` in
  `TensorsToDetectionsCalculator::Open()`) — stick to 0.10.30 until that's
  fixed upstream.
- Distance accuracy depends on the calibration step; the assumed iris size
  (`--iris-mm`, default 11.7mm) is an adult average and can be tuned per
  person for extra precision.
