import { FilesetResolver, FaceLandmarker } from "@mediapipe/tasks-vision";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import "./style.css";
import { averageIrisDiameterPx } from "./geometry";
import {
  type Calibration,
  loadCalibration,
  saveCalibration,
  clearCalibration,
  distanceMmFromCalibration,
  calibrateFromMeasurement,
} from "./calibration";
import { populateDeviceList, startStreamForDevice } from "./camera";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const THEME_KEY = "distance-checker:theme";

type StatusKind = "ok" | "warn" | "bad";
type SmoothedState = "close" | "far" | "good" | null;
type Theme = "light" | "dark";

function el<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing element #${id}`);
  return found as T;
}

function get2dContext(c: HTMLCanvasElement): CanvasRenderingContext2D {
  const context = c.getContext("2d");
  if (!context) throw new Error("Canvas 2D context not available");
  return context;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

const startBtn = el<HTMLButtonElement>("start-btn");
const deviceField = el<HTMLElement>("device-field");
const deviceSelect = el<HTMLSelectElement>("device-select");
const stage = el<HTMLElement>("stage");
const video = el<HTMLVideoElement>("video");
const canvas = el<HTMLCanvasElement>("canvas");
const ctx = get2dContext(canvas);
const statusCard = el<HTMLElement>("status-card");
const statusText = el<HTMLElement>("status-text");
const calibDistanceInput = el<HTMLInputElement>("calib-distance");
const tooCloseInput = el<HTMLInputElement>("too-close");
const tooFarInput = el<HTMLInputElement>("too-far");
const irisMmInput = el<HTMLInputElement>("iris-mm");
const calibrateBtn = el<HTMLButtonElement>("calibrate-btn");
const resetCalibBtn = el<HTMLButtonElement>("reset-calib-btn");
const changeCameraBtn = el<HTMLButtonElement>("change-camera-btn");
const fpsEl = el<HTMLElement>("fps");
const sparklineCanvas = el<HTMLCanvasElement>("distance-sparkline");
const sparklineCtx = get2dContext(sparklineCanvas);
const showMeshToggle = el<HTMLInputElement>("show-mesh-toggle");
const themeToggleBtn = el<HTMLButtonElement>("theme-toggle");
const alertAfterInput = el<HTMLInputElement>("alert-after-seconds");
const enableNotificationsInput = el<HTMLInputElement>("enable-notifications");
const speakAlertsInput = el<HTMLInputElement>("speak-alerts");
const snoozeBtn = el<HTMLButtonElement>("snooze-btn");
const snoozeRemainingEl = el<HTMLElement>("snooze-remaining");
const statusAnnouncer = el<HTMLElement>("status-announcer");
const pipBtn = el<HTMLButtonElement>("pip-btn");

const videoWrapEl = document.querySelector<HTMLDivElement>(".video-wrap");
if (!videoWrapEl) throw new Error("Missing .video-wrap");
const videoWrap: HTMLDivElement = videoWrapEl;

let currentStream: MediaStream | null = null;
let faceLandmarker: FaceLandmarker | null = null;
let calibration: Calibration | null = loadCalibration();
let latestIrisDiameterPx: number | null = null;
let fps = 0;
let lastFrameTime = performance.now();
let rafId: number | null = null;
let lastAnnouncedKind: StatusKind | undefined;

// --- Document Picture-in-Picture (floating window) ---
interface DocumentPictureInPictureAPI {
  requestWindow(options?: { width?: number; height?: number }): Promise<Window>;
}
const documentPictureInPicture = (
  window as unknown as { documentPictureInPicture?: DocumentPictureInPictureAPI }
).documentPictureInPicture;
const supportsDocumentPip = Boolean(documentPictureInPicture);

let pipWindow: Window | null = null;
let pipStatusEl: HTMLDivElement | null = null;
let pipPlaceholder: HTMLDivElement | null = null;

if (pipBtn) pipBtn.hidden = !supportsDocumentPip;

// --- Distance sparkline (rolling ~60s buffer, updated at ~10fps) ---
interface DistanceSample {
  t: number;
  distanceCm: number | null;
}
const SPARKLINE_WINDOW_MS = 60_000;
const SPARKLINE_UPDATE_INTERVAL_MS = 100;
const SPARKLINE_MAX_SAMPLES = Math.ceil(SPARKLINE_WINDOW_MS / SPARKLINE_UPDATE_INTERVAL_MS);
const distanceHistory: DistanceSample[] = [];
let lastSparklineSampleTime = 0;

// --- Sustained-state smoothing + alerting (notifications / speech) ---
// This tracks the distance state ("close" / "far" / "good" / null) separately
// from the per-frame `setStatus` calls below, so the visible status card keeps
// reacting instantly (unchanged behavior) while alerts only react to a
// hysteresis-smoothed, sustained state.
const STATE_HYSTERESIS_MS = 1500; // ~1.5s of consistent state before it "counts" as changed
const SNOOZE_MS = 10 * 60 * 1000; // 10 minutes

let rawPendingState: SmoothedState = null;
let rawPendingSince = 0;
let smoothedState: SmoothedState = null;

let sustainedSince: number | null = null; // when smoothedState most recently became "close"/"far"
let episodeNotified = false;
let episodeSpoken = false;
let snoozeUntil = 0;

// --- Theme (light/dark) ---
function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute("data-theme", theme);
  if (themeToggleBtn) {
    const isLight = theme === "light";
    themeToggleBtn.textContent = isLight ? "☀️" : "🌙";
    themeToggleBtn.setAttribute("aria-label", isLight ? "Switch to dark theme" : "Switch to light theme");
  }
}

function initTheme(): void {
  // index.html already sets data-theme on <html> before this script loads
  // (from localStorage, falling back to prefers-color-scheme), so just sync
  // the toggle button's icon/label to whatever is currently applied.
  const current = (document.documentElement.getAttribute("data-theme") as Theme | null) || "dark";
  applyTheme(current);
}

themeToggleBtn?.addEventListener("click", () => {
  const current: Theme = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  const next: Theme = current === "light" ? "dark" : "light";
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch {
    // ignore storage failures (e.g. private browsing)
  }
  applyTheme(next);
});

initTheme();

async function ensureFaceLandmarker(): Promise<FaceLandmarker> {
  if (faceLandmarker) return faceLandmarker;
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
  });
  return faceLandmarker;
}

function setStatus(text: string, kind?: StatusKind): void {
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
function copyStylesIntoWindow(win: Window): void {
  for (const styleSheet of Array.from(document.styleSheets)) {
    try {
      const cssRules = Array.from(styleSheet.cssRules)
        .map((rule) => rule.cssText)
        .join("");
      const style = document.createElement("style");
      style.textContent = cssRules;
      win.document.head.appendChild(style);
    } catch {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.type = styleSheet.type;
      link.media = styleSheet.media.mediaText;
      if (styleSheet.href) link.href = styleSheet.href;
      win.document.head.appendChild(link);
    }
  }
}

function onPipClosed(): void {
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

async function openPip(): Promise<void> {
  if (!documentPictureInPicture || pipWindow) return;
  try {
    pipWindow = await documentPictureInPicture.requestWindow({ width: 340, height: 320 });
  } catch (err) {
    console.error("Picture-in-Picture failed:", errorMessage(err));
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
  for (const kind of ["ok", "warn", "bad"] as const) {
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

// --- Distance sparkline ---
function recordDistanceSample(now: number, distanceCm: number | null): void {
  if (now - lastSparklineSampleTime < SPARKLINE_UPDATE_INTERVAL_MS) return;
  lastSparklineSampleTime = now;
  distanceHistory.push({ t: now, distanceCm });
  while (distanceHistory.length > SPARKLINE_MAX_SAMPLES) distanceHistory.shift();
  const cutoff = now - SPARKLINE_WINDOW_MS;
  while (distanceHistory.length > 0 && distanceHistory[0].t < cutoff) distanceHistory.shift();
  drawSparkline(now);
}

function drawSparkline(now: number): void {
  const w = sparklineCanvas.width;
  const h = sparklineCanvas.height;
  sparklineCtx.clearRect(0, 0, w, h);

  const valid = distanceHistory.filter(
    (s): s is DistanceSample & { distanceCm: number } => s.distanceCm != null && Number.isFinite(s.distanceCm)
  );
  if (valid.length < 2) return;

  const tooClose = Number(tooCloseInput.value);
  const tooFar = Number(tooFarInput.value);
  let min = Math.min(...valid.map((s) => s.distanceCm), tooClose);
  let max = Math.max(...valid.map((s) => s.distanceCm), tooFar);
  if (max - min < 1) max = min + 1;
  const pad = 6;

  const xFor = (t: number) => {
    const oldest = now - SPARKLINE_WINDOW_MS;
    const frac = (t - oldest) / SPARKLINE_WINDOW_MS;
    return pad + frac * (w - pad * 2);
  };
  const yFor = (d: number) => {
    const frac = (d - min) / (max - min);
    return h - pad - frac * (h - pad * 2);
  };

  // Guide lines for the too-close / too-far thresholds.
  sparklineCtx.strokeStyle = "rgba(255,255,255,0.15)";
  sparklineCtx.lineWidth = 1;
  for (const threshold of [tooClose, tooFar]) {
    const y = yFor(threshold);
    sparklineCtx.beginPath();
    sparklineCtx.moveTo(0, y);
    sparklineCtx.lineTo(w, y);
    sparklineCtx.stroke();
  }

  sparklineCtx.strokeStyle = "#ffb454";
  sparklineCtx.lineWidth = 2;
  sparklineCtx.beginPath();
  distanceHistory.forEach((s, i) => {
    if (s.distanceCm == null || !Number.isFinite(s.distanceCm)) return;
    const x = xFor(s.t);
    const y = yFor(s.distanceCm);
    const prev = distanceHistory[i - 1];
    if (i === 0 || prev.distanceCm == null) {
      sparklineCtx.moveTo(x, y);
    } else {
      sparklineCtx.lineTo(x, y);
    }
  });
  sparklineCtx.stroke();
}

// --- Face mesh wireframe overlay ---
function drawFaceMesh(landmarks: NormalizedLandmark[], w: number, h: number): void {
  const connectors = FaceLandmarker.FACE_LANDMARKS_TESSELATION;
  if (!connectors) return;
  ctx.strokeStyle = "rgba(95, 217, 127, 0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const { start, end } of connectors) {
    const a = landmarks[start];
    const b = landmarks[end];
    if (!a || !b) continue;
    ctx.moveTo(a.x * w, a.y * h);
    ctx.lineTo(b.x * w, b.y * h);
  }
  ctx.stroke();
}

function isSnoozed(now: number): boolean {
  return now < snoozeUntil;
}

// Hysteresis: only adopt a new raw state once it has been reported
// consistently for STATE_HYSTERESIS_MS, so brief flickers (e.g. leaning
// forward for a split second) don't trigger alert logic.
function updateSmoothedState(rawState: SmoothedState, now: number): SmoothedState {
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

function maybeNotify(title: string, body: string): void {
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
    console.warn("Notification failed", errorMessage(err));
  }
}

function speak(text: string): void {
  if (!speakAlertsInput.checked) return;
  if (typeof window.speechSynthesis === "undefined" || typeof SpeechSynthesisUtterance === "undefined") return;
  try {
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  } catch (err) {
    console.warn("Speech synthesis failed", errorMessage(err));
  }
}

// Drives the sustained-episode logic: fires (at most once per episode) a
// notification/speech alert once the smoothed state has been "close" or
// "far" continuously for the configured alert-after duration, and resets
// the episode once the state returns to "good"/unknown.
function handleSustainedAlerts(state: SmoothedState, now: number): void {
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

function updateSnoozeUI(now: number): void {
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

function renderLoop(): void {
  rafId = requestAnimationFrame(renderLoop);
  if (video.readyState < 2 || !faceLandmarker) return;

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
  let rawState: SmoothedState = null; // feeds sustained-alert smoothing only

  if (result.faceLandmarks && result.faceLandmarks.length > 0) {
    const landmarks = result.faceLandmarks[0];
    const { diameter, leftCenter, rightCenter } = averageIrisDiameterPx(
      landmarks,
      canvas.width,
      canvas.height
    );
    latestIrisDiameterPx = diameter;

    if (showMeshToggle.checked) {
      drawFaceMesh(landmarks, canvas.width, canvas.height);
    }

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
      recordDistanceSample(now, distanceCm);
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
      recordDistanceSample(now, null);
      setStatus(`Not calibrated — sit at ${calibDistanceInput.value}cm and press Calibrate`, "warn");
    }
  } else {
    recordDistanceSample(now, null);
    setStatus("No face detected", "bad");
  }

  const smoothed = updateSmoothedState(rawState, now);
  handleSustainedAlerts(smoothed, now);
}

function doCalibrate(): void {
  if (!latestIrisDiameterPx) {
    setStatus("Can't calibrate — no face detected right now", "bad");
    return;
  }
  const knownDistanceMm = Number(calibDistanceInput.value) * 10;
  const irisMm = Number(irisMmInput.value);
  const cal = calibrateFromMeasurement(knownDistanceMm, latestIrisDiameterPx, irisMm);
  calibration = cal;
  saveCalibration(cal);
  speak("Calibrated");
}

function doResetCalibration(): void {
  calibration = null;
  clearCalibration();
  setStatus("Calibration cleared", "warn");
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Requesting camera…";
  try {
    // First grant unlocks device labels for enumerateDevices().
    currentStream = await startStreamForDevice(video, currentStream, null);
    const activeStream = currentStream;
    const devices = await populateDeviceList(deviceSelect);
    if (devices.length > 1) {
      deviceField.hidden = false;
      // Preselect the device actually backing the current stream.
      const activeId = activeStream.getVideoTracks()[0]?.getSettings().deviceId;
      if (activeId) deviceSelect.value = activeId;
    }
    await ensureFaceLandmarker();

    const showStage = () => {
      stage.hidden = false;
      el<HTMLElement>("setup-panel").querySelector<HTMLButtonElement>("#start-btn")!.hidden = true;
    };
    type ViewTransitionDocument = Document & {
      startViewTransition?: (callback: () => void) => unknown;
    };
    const vtDocument = document as ViewTransitionDocument;
    if (vtDocument.startViewTransition) {
      vtDocument.startViewTransition(() => showStage());
    } else {
      showStage();
    }

    if (rafId === null) renderLoop();
  } catch (err) {
    console.error(err);
    setStatus("Camera access failed: " + errorMessage(err), "bad");
    startBtn.disabled = false;
    startBtn.textContent = "Start Camera";
  }
});

deviceSelect.addEventListener("change", async () => {
  try {
    currentStream = await startStreamForDevice(video, currentStream, deviceSelect.value);
  } catch (err) {
    console.error(err);
    setStatus("Could not switch camera: " + errorMessage(err), "bad");
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
  // Don't hijack keystrokes while the user is typing into a form control.
  const active = document.activeElement;
  const tag = active?.tagName;
  const isEditable = active instanceof HTMLElement && active.isContentEditable;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || isEditable) {
    return;
  }
  if (e.key === "c") doCalibrate();
  if (e.key === "r") doResetCalibration();
});
