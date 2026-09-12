import {
  FilesetResolver,
  FaceLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const LEFT_IRIS = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];
const CALIBRATION_KEY = "distance-checker:calibration";

const el = (id) => document.getElementById(id);
const startBtn = el("start-btn");
const deviceField = el("device-field");
const deviceSelect = el("device-select");
const stage = el("stage");
const video = el("video");
const canvas = el("canvas");
const ctx = canvas.getContext("2d");
const statusCard = el("status-card");
const statusText = el("status-text");
const calibDistanceInput = el("calib-distance");
const tooCloseInput = el("too-close");
const tooFarInput = el("too-far");
const irisMmInput = el("iris-mm");
const calibrateBtn = el("calibrate-btn");
const resetCalibBtn = el("reset-calib-btn");
const changeCameraBtn = el("change-camera-btn");
const fpsEl = el("fps");
const statusAnnouncer = el("status-announcer");
const pipBtn = el("pip-btn");
const videoWrap = document.querySelector(".video-wrap");

let currentStream = null;
let faceLandmarker = null;
let calibration = loadCalibration();
let latestIrisDiameterPx = null;
let fps = 0;
let lastFrameTime = performance.now();
let rafId = null;
let lastAnnouncedKind = undefined;

// --- Document Picture-in-Picture (floating window) ---
let pipWindow = null;
let pipStatusEl = null;
let pipPlaceholder = null;
const supportsDocumentPip = "documentPictureInPicture" in window;
if (pipBtn) pipBtn.hidden = !supportsDocumentPip;

function loadCalibration() {
  try {
    const raw = localStorage.getItem(CALIBRATION_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveCalibration(cal) {
  calibration = cal;
  localStorage.setItem(CALIBRATION_KEY, JSON.stringify(cal));
}

function clearCalibration() {
  calibration = null;
  localStorage.removeItem(CALIBRATION_KEY);
}

// --- Minimal enclosing circle (Welzl's algorithm) for a small point set ---
function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function circleFromTwo(a, b) {
  const center = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  return { center, radius: dist(a, b) / 2 };
}

function circleFromThree(a, b, c) {
  const ax = a[0], ay = a[1];
  const bx = b[0], by = b[1];
  const cx = c[0], cy = c[1];
  const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  if (Math.abs(d) < 1e-9) return circleFromTwo(a, b);
  const ux =
    ((ax * ax + ay * ay) * (by - cy) +
      (bx * bx + by * by) * (cy - ay) +
      (cx * cx + cy * cy) * (ay - by)) /
    d;
  const uy =
    ((ax * ax + ay * ay) * (cx - bx) +
      (bx * bx + by * by) * (ax - cx) +
      (cx * cx + cy * cy) * (bx - ax)) /
    d;
  const center = [ux, uy];
  return { center, radius: dist(center, a) };
}

function inCircle(circle, p, eps = 1e-6) {
  return dist(circle.center, p) <= circle.radius + eps;
}

function minEnclosingCircle(points) {
  let circle = null;
  for (let i = 0; i < points.length; i++) {
    if (!circle || !inCircle(circle, points[i])) {
      circle = { center: points[i], radius: 0 };
      for (let j = 0; j < i; j++) {
        if (!inCircle(circle, points[j])) {
          circle = circleFromTwo(points[i], points[j]);
          for (let k = 0; k < j; k++) {
            if (!inCircle(circle, points[k])) {
              circle = circleFromThree(points[i], points[j], points[k]);
            }
          }
        }
      }
    }
  }
  return circle;
}

function irisDiameterPx(landmarks, indices, w, h) {
  const pts = indices.map((i) => [landmarks[i].x * w, landmarks[i].y * h]);
  const circle = minEnclosingCircle(pts);
  return { diameter: circle.radius * 2, center: circle.center };
}

function averageIrisDiameterPx(landmarks, w, h) {
  const left = irisDiameterPx(landmarks, LEFT_IRIS, w, h);
  const right = irisDiameterPx(landmarks, RIGHT_IRIS, w, h);
  return {
    diameter: (left.diameter + right.diameter) / 2,
    leftCenter: left.center,
    rightCenter: right.center,
  };
}

function distanceMmFromCalibration(cal, diameterPx) {
  if (diameterPx <= 0) return NaN;
  return (cal.irisDiameterMm * cal.focalLengthPx) / diameterPx;
}

function calibrateFromMeasurement(knownDistanceMm, diameterPx, irisDiameterMm) {
  const focalLengthPx = (diameterPx * knownDistanceMm) / irisDiameterMm;
  return { focalLengthPx, knownDistanceMm, irisDiameterMm, diameterPxAtCalibration: diameterPx };
}

// --- Camera setup ---
async function populateDeviceList() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoInputs = devices.filter((d) => d.kind === "videoinput");
  deviceSelect.innerHTML = "";
  for (const d of videoInputs) {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `Camera (${d.deviceId.slice(0, 6)})`;
    deviceSelect.appendChild(opt);
  }
  return videoInputs;
}

async function startStreamForDevice(deviceId) {
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
  }
  const constraints = {
    video: deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  currentStream = stream;
  video.srcObject = stream;
  await video.play();
}

async function ensureFaceLandmarker() {
  if (faceLandmarker) return faceLandmarker;
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
  });
  return faceLandmarker;
}

function setStatus(text, kind) {
  statusText.textContent = text;
  statusCard.classList.remove("ok", "warn", "bad");
  if (kind) statusCard.classList.add(kind);

  // Screen readers: only announce when the status actually changes state
  // (not on every frame, since the cm value in `text` fluctuates constantly).
  if (kind !== lastAnnouncedKind) {
    lastAnnouncedKind = kind;
    if (statusAnnouncer) {
      statusAnnouncer.setAttribute("aria-live", kind === "bad" ? "assertive" : "polite");
      statusAnnouncer.textContent = text;
    }
  }

  // Mirror onto the compact floating-window readout, if open.
  if (pipStatusEl) {
    pipStatusEl.textContent = text;
    pipStatusEl.classList.remove("ok", "warn", "bad");
    if (kind) pipStatusEl.classList.add(kind);
  }
}

// --- Document Picture-in-Picture floating window ---
function copyStylesIntoWindow(win) {
  for (const styleSheet of document.styleSheets) {
    try {
      const cssRules = [...styleSheet.cssRules].map((rule) => rule.cssText).join("");
      const style = document.createElement("style");
      style.textContent = cssRules;
      win.document.head.appendChild(style);
    } catch {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.type = styleSheet.type;
      link.media = styleSheet.media;
      link.href = styleSheet.href;
      win.document.head.appendChild(link);
    }
  }
}

function onPipClosed() {
  if (pipPlaceholder && pipPlaceholder.parentElement) {
    pipPlaceholder.replaceWith(videoWrap);
  }
  pipPlaceholder = null;
  pipStatusEl = null;
  pipWindow = null;
  if (pipBtn) {
    pipBtn.textContent = "Float window (PiP)";
    pipBtn.setAttribute("aria-pressed", "false");
  }
}

async function openPip() {
  if (!supportsDocumentPip || pipWindow) return;
  try {
    pipWindow = await documentPictureInPicture.requestWindow({ width: 340, height: 320 });
  } catch (err) {
    console.error("Picture-in-Picture failed:", err);
    pipWindow = null;
    return;
  }

  copyStylesIntoWindow(pipWindow);
  pipWindow.document.title = "Distance Checker";
  pipWindow.document.body.classList.add("pip-body");

  pipPlaceholder = document.createElement("div");
  pipPlaceholder.className = "video-wrap-placeholder";
  pipPlaceholder.textContent = "Floating in the Picture-in-Picture window…";
  videoWrap.replaceWith(pipPlaceholder);

  pipStatusEl = document.createElement("div");
  pipStatusEl.className = "status-card pip-status";
  pipStatusEl.textContent = statusText.textContent;
  for (const kind of ["ok", "warn", "bad"]) {
    if (statusCard.classList.contains(kind)) pipStatusEl.classList.add(kind);
  }

  pipWindow.document.body.appendChild(videoWrap);
  pipWindow.document.body.appendChild(pipStatusEl);
  pipWindow.addEventListener("pagehide", onPipClosed, { once: true });

  if (pipBtn) {
    pipBtn.textContent = "Exit floating window";
    pipBtn.setAttribute("aria-pressed", "true");
  }
}

if (pipBtn) {
  pipBtn.addEventListener("click", () => {
    if (pipWindow) {
      pipWindow.close();
    } else {
      openPip();
    }
  });
}

function renderLoop() {
  rafId = requestAnimationFrame(renderLoop);
  if (video.readyState < 2) return;

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const now = performance.now();
  const result = faceLandmarker.detectForVideo(video, now);

  const dt = now - lastFrameTime;
  lastFrameTime = now;
  if (dt > 0) fps = fps * 0.9 + (1000 / dt) * 0.1;
  fpsEl.textContent = `FPS: ${fps.toFixed(0)}`;

  latestIrisDiameterPx = null;

  if (result.faceLandmarks && result.faceLandmarks.length > 0) {
    const landmarks = result.faceLandmarks[0];
    const { diameter, leftCenter, rightCenter } = averageIrisDiameterPx(
      landmarks,
      canvas.width,
      canvas.height
    );
    latestIrisDiameterPx = diameter;

    ctx.strokeStyle = "#5fd97f";
    ctx.lineWidth = 2;
    for (const c of [leftCenter, rightCenter]) {
      ctx.beginPath();
      ctx.arc(c[0], c[1], diameter / 2, 0, Math.PI * 2);
      ctx.stroke();
    }

    if (calibration) {
      const distanceCm = distanceMmFromCalibration(calibration, diameter) / 10;
      const tooClose = Number(tooCloseInput.value);
      const tooFar = Number(tooFarInput.value);
      if (distanceCm < tooClose) {
        setStatus(`Too close (${distanceCm.toFixed(0)} cm)`, "bad");
      } else if (distanceCm > tooFar) {
        setStatus(`Too far (${distanceCm.toFixed(0)} cm)`, "warn");
      } else {
        setStatus(`Good distance (${distanceCm.toFixed(0)} cm)`, "ok");
      }
    } else {
      setStatus(`Not calibrated — sit at ${calibDistanceInput.value}cm and press Calibrate`, "warn");
    }
  } else {
    setStatus("No face detected", "bad");
  }
}

function doCalibrate() {
  if (!latestIrisDiameterPx) {
    setStatus("Can't calibrate — no face detected right now", "bad");
    return;
  }
  const knownDistanceMm = Number(calibDistanceInput.value) * 10;
  const irisMm = Number(irisMmInput.value);
  const cal = calibrateFromMeasurement(knownDistanceMm, latestIrisDiameterPx, irisMm);
  saveCalibration(cal);
}

function doResetCalibration() {
  clearCalibration();
  setStatus("Calibration cleared", "warn");
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Requesting camera…";
  try {
    // First grant unlocks device labels for enumerateDevices().
    await startStreamForDevice(null);
    const devices = await populateDeviceList();
    if (devices.length > 1) {
      deviceField.hidden = false;
      // Preselect the device actually backing the current stream.
      const activeId = currentStream.getVideoTracks()[0]?.getSettings().deviceId;
      if (activeId) deviceSelect.value = activeId;
    }
    await ensureFaceLandmarker();
    stage.hidden = false;
    el("setup-panel").querySelector("#start-btn").hidden = true;
    if (rafId === null) renderLoop();
  } catch (err) {
    console.error(err);
    setStatus("Camera access failed: " + err.message, "bad");
    startBtn.disabled = false;
    startBtn.textContent = "Start Camera";
  }
});

deviceSelect.addEventListener("change", async () => {
  try {
    await startStreamForDevice(deviceSelect.value);
  } catch (err) {
    console.error(err);
    setStatus("Could not switch camera: " + err.message, "bad");
  }
});

changeCameraBtn.addEventListener("click", () => {
  deviceField.hidden = false;
  deviceField.scrollIntoView({ behavior: "smooth", block: "center" });
});

calibrateBtn.addEventListener("click", doCalibrate);
resetCalibBtn.addEventListener("click", doResetCalibration);

window.addEventListener("keydown", (e) => {
  if (stage.hidden) return;
  // Don't hijack keystrokes while the user is typing into a form control.
  const active = document.activeElement;
  const tag = active && active.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || (active && active.isContentEditable)) {
    return;
  }
  if (e.key === "c") doCalibrate();
  if (e.key === "r") doResetCalibration();
});
