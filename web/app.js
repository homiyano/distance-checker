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
const alertAfterInput = el("alert-after-seconds");
const enableNotificationsInput = el("enable-notifications");
const speakAlertsInput = el("speak-alerts");
const snoozeBtn = el("snooze-btn");
const snoozeRemainingEl = el("snooze-remaining");

let currentStream = null;
let faceLandmarker = null;
let calibration = loadCalibration();
let latestIrisDiameterPx = null;
let fps = 0;
let lastFrameTime = performance.now();
let rafId = null;

// --- Sustained-state smoothing + alerting (notifications / speech) ---
// This tracks the distance state ("close" / "far" / "good" / null) separately
// from the per-frame `setStatus` calls below, so the visible status card keeps
// reacting instantly (unchanged behavior) while alerts only react to a
// hysteresis-smoothed, sustained state.
const STATE_HYSTERESIS_MS = 1500; // ~1.5s of consistent state before it "counts" as changed
const SNOOZE_MS = 10 * 60 * 1000; // 10 minutes

let rawPendingState = null;
let rawPendingSince = 0;
let smoothedState = null; // "close" | "far" | "good" | null

let sustainedSince = null; // when smoothedState most recently became "close"/"far"
let episodeNotified = false;
let episodeSpoken = false;
let snoozeUntil = 0;

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
}

function isSnoozed(now) {
  return now < snoozeUntil;
}

// Hysteresis: only adopt a new raw state once it has been reported
// consistently for STATE_HYSTERESIS_MS, so brief flickers (e.g. leaning
// forward for a split second) don't trigger alert logic.
function updateSmoothedState(rawState, now) {
  if (rawState !== rawPendingState) {
    rawPendingState = rawState;
    rawPendingSince = now;
  }
  if (smoothedState === null) {
    smoothedState = rawState;
  } else if (rawState !== smoothedState && now - rawPendingSince >= STATE_HYSTERESIS_MS) {
    smoothedState = rawState;
  }
  return smoothedState;
}

function maybeNotify(title, body) {
  if (!enableNotificationsInput.checked) return;
  if (typeof Notification === "undefined") return;
  try {
    if (Notification.permission === "granted") {
      new Notification(title, { body });
    } else if (Notification.permission !== "denied") {
      Notification.requestPermission().then((perm) => {
        if (perm === "granted") new Notification(title, { body });
      });
    }
  } catch (err) {
    console.warn("Notification failed", err);
  }
}

function speak(text) {
  if (!speakAlertsInput.checked) return;
  if (typeof window.speechSynthesis === "undefined" || typeof SpeechSynthesisUtterance === "undefined") return;
  try {
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  } catch (err) {
    console.warn("Speech synthesis failed", err);
  }
}

// Drives the sustained-episode logic: fires (at most once per episode) a
// notification/speech alert once the smoothed state has been "close" or
// "far" continuously for the configured alert-after duration, and resets
// the episode once the state returns to "good"/unknown.
function handleSustainedAlerts(state, now) {
  const isBad = state === "close" || state === "far";
  if (!isBad) {
    sustainedSince = null;
    episodeNotified = false;
    episodeSpoken = false;
    statusCard.classList.remove("sustained");
    return;
  }

  if (sustainedSince === null) sustainedSince = now;
  const sustainedMs = now - sustainedSince;
  const alertAfterMs = Math.max(1, Number(alertAfterInput.value) || 15) * 1000;
  const snoozed = isSnoozed(now);

  statusCard.classList.toggle("sustained", sustainedMs >= alertAfterMs && !snoozed);

  if (sustainedMs < alertAfterMs || snoozed) return;

  const notifyMessage =
    state === "close"
      ? "You've been sitting too close for a while"
      : "You've been sitting too far for a while";
  const speechMessage = state === "close" ? "You're sitting too close" : "You're sitting too far away";

  if (!episodeNotified) {
    episodeNotified = true;
    maybeNotify("Distance Checker", notifyMessage);
  }
  if (!episodeSpoken) {
    episodeSpoken = true;
    speak(speechMessage);
  }
}

function updateSnoozeUI(now) {
  const remainingMs = snoozeUntil - now;
  if (remainingMs > 0) {
    const remainingSec = Math.ceil(remainingMs / 1000);
    const mm = Math.floor(remainingSec / 60);
    const ss = remainingSec % 60;
    snoozeRemainingEl.textContent = `Snoozed ${mm}:${String(ss).padStart(2, "0")}`;
    snoozeRemainingEl.hidden = false;
    snoozeBtn.textContent = "Snoozing…";
    snoozeBtn.disabled = true;
  } else {
    snoozeRemainingEl.hidden = true;
    snoozeBtn.textContent = "Snooze alerts (10 min)";
    snoozeBtn.disabled = false;
  }
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
  let rawState = null; // "close" | "far" | "good" | null — feeds sustained-alert smoothing only

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
        rawState = "close";
      } else if (distanceCm > tooFar) {
        setStatus(`Too far (${distanceCm.toFixed(0)} cm)`, "warn");
        rawState = "far";
      } else {
        setStatus(`Good distance (${distanceCm.toFixed(0)} cm)`, "ok");
        rawState = "good";
      }
    } else {
      setStatus(`Not calibrated — sit at ${calibDistanceInput.value}cm and press Calibrate`, "warn");
    }
  } else {
    setStatus("No face detected", "bad");
  }

  const smoothed = updateSmoothedState(rawState, now);
  handleSustainedAlerts(smoothed, now);
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
  speak("Calibrated");
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

snoozeBtn.addEventListener("click", () => {
  snoozeUntil = performance.now() + SNOOZE_MS;
  updateSnoozeUI(performance.now());
});

// Independent of the render loop so the countdown keeps ticking even before
// the camera stream is fully ready.
setInterval(() => updateSnoozeUI(performance.now()), 1000);
updateSnoozeUI(performance.now());

window.addEventListener("keydown", (e) => {
  if (stage.hidden) return;
  if (e.key === "c") doCalibrate();
  if (e.key === "r") doResetCalibration();
});
