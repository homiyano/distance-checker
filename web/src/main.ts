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
import { SustainedStateTracker, SustainedEpisode } from "./alerts";
import { AlarmSoundEngine, type AlarmSoundId } from "./sound";
import {
  PomodoroTimer,
  type PomodoroPhase,
  loadPomodoroSettings,
  savePomodoroSettings,
} from "./pomodoro";
import { populateDeviceList, startStreamForDevice } from "./camera";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const THEME_KEY = "distance-checker:theme";

type StatusKind = "ok" | "warn" | "bad";
type SmoothedDistanceState = "close" | "far" | "good" | null;
type SmoothedPresenceState = "present" | "away" | null;
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

// --- Settings dialog ---
const settingsBtn = el<HTMLButtonElement>("settings-btn");
const settingsDialog = el<HTMLDialogElement>("settings-dialog");

// --- Presence / away emergency alert ---
const presenceStatusCard = el<HTMLElement>("presence-status-card");
const presenceStatusText = el<HTMLElement>("presence-status-text");
const presenceAnnouncer = el<HTMLElement>("presence-status-announcer");
const enableAwayAlertInput = el<HTMLInputElement>("enable-away-alert");
const awayThresholdInput = el<HTMLInputElement>("away-threshold-seconds");
const emergencySoundSelect = el<HTMLSelectElement>("emergency-sound-select");
const emergencyVolumeInput = el<HTMLInputElement>("emergency-volume");
const emergencyRepeatSecondsInput = el<HTMLInputElement>("emergency-repeat-seconds");
const testEmergencyBtn = el<HTMLButtonElement>("test-emergency-btn");

// --- Customizable alarm sound (distance) ---
const enableAlarmSoundInput = el<HTMLInputElement>("enable-alarm-sound");
const alarmSoundSelect = el<HTMLSelectElement>("alarm-sound-select");
const alarmVolumeInput = el<HTMLInputElement>("alarm-volume");
const alarmRepeatSecondsInput = el<HTMLInputElement>("alarm-repeat-seconds");
const testSoundBtn = el<HTMLButtonElement>("test-sound-btn");

// --- Pomodoro focus timer ---
const pomodoroEl = el<HTMLElement>("pomodoro");
const pomodoroPhaseEl = el<HTMLElement>("pomodoro-phase");
const pomodoroTimeEl = el<HTMLElement>("pomodoro-time");
const pomodoroDotsEl = el<HTMLElement>("pomodoro-dots");
const pomodoroStartBtn = el<HTMLButtonElement>("pomodoro-start-btn");
const pomodoroSkipBtn = el<HTMLButtonElement>("pomodoro-skip-btn");
const pomodoroResetBtn = el<HTMLButtonElement>("pomodoro-reset-btn");
const focusMinutesInput = el<HTMLInputElement>("focus-minutes");
const shortBreakMinutesInput = el<HTMLInputElement>("short-break-minutes");
const longBreakMinutesInput = el<HTMLInputElement>("long-break-minutes");
const cyclesBeforeLongBreakInput = el<HTMLInputElement>("cycles-before-long-break");
const pomodoroAutoStartInput = el<HTMLInputElement>("pomodoro-auto-start");

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
let backgroundTimerId: number | null = null;
let lastAnnouncedKind: StatusKind | undefined;
let lastAnnouncedPresenceKind: StatusKind | undefined;
const ORIGINAL_TITLE = document.title;

// A background tab throttles/pauses requestAnimationFrame, but the webcam
// stream itself keeps delivering frames — so while hidden we fall back to a
// low-rate setTimeout loop instead, keeping distance/presence monitoring
// (and therefore alerts) alive while the user works in another tab or app.
const BACKGROUND_FRAME_INTERVAL_MS = 500;

const alarmSoundEngine = new AlarmSoundEngine();
const emergencyAlarmEngine = new AlarmSoundEngine();

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
let pipPresenceEl: HTMLDivElement | null = null;
let pipPomodoroEl: HTMLDivElement | null = null;
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

// --- Sustained-state smoothing + alerting (notifications / speech / sound) ---
// Each channel (distance, presence) tracks its raw per-frame state separately
// from the per-frame `setStatus`/`setPresenceStatus` calls below, so the
// visible status cards keep reacting instantly while alerts only react to a
// hysteresis-smoothed, sustained state.
const STATE_HYSTERESIS_MS = 1500; // ~1.5s of consistent state before it "counts" as changed
const SNOOZE_MS = 10 * 60 * 1000; // 10 minutes

const distanceStateTracker = new SustainedStateTracker(STATE_HYSTERESIS_MS);
const presenceStateTracker = new SustainedStateTracker(STATE_HYSTERESIS_MS);
const distanceEpisode = new SustainedEpisode();
const presenceEpisode = new SustainedEpisode();

let snoozeUntil = 0;
let distanceAlarmActive = false;
let emergencyAlarmActive = false;
let titleFlashOn = false;

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

// --- Settings dialog ---
settingsBtn.addEventListener("click", () => settingsDialog.showModal());
settingsDialog.addEventListener("click", (e) => {
  if (e.target === settingsDialog) settingsDialog.close();
});

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

function setPresenceStatus(text: string, kind?: StatusKind): void {
  presenceStatusText.textContent = text;
  presenceStatusCard.classList.remove("ok", "warn", "bad");
  if (kind) presenceStatusCard.classList.add(kind);

  if (kind !== lastAnnouncedPresenceKind) {
    lastAnnouncedPresenceKind = kind;
    if (presenceAnnouncer) {
      presenceAnnouncer.setAttribute("aria-live", kind === "bad" ? "assertive" : "polite");
      presenceAnnouncer.textContent = text;
    }
  }

  if (pipPresenceEl) {
    pipPresenceEl.textContent = text;
    pipPresenceEl.classList.remove("ok", "warn", "bad");
    if (kind) pipPresenceEl.classList.add(kind);
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
  pipPresenceEl = null;
  pipPomodoroEl = null;
  pipWindow = null;
  if (pipBtn) {
    pipBtn.textContent = "Float window (PiP)";
    pipBtn.setAttribute("aria-pressed", "false");
  }
}

async function openPip(): Promise<void> {
  if (!documentPictureInPicture || pipWindow) return;
  try {
    pipWindow = await documentPictureInPicture.requestWindow({ width: 340, height: 380 });
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

  pipPresenceEl = document.createElement("div");
  pipPresenceEl.className = "status-card pip-status";
  pipPresenceEl.textContent = presenceStatusText.textContent;
  for (const kind of ["ok", "warn", "bad"] as const) {
    if (presenceStatusCard.classList.contains(kind)) pipPresenceEl.classList.add(kind);
  }

  pipPomodoroEl = document.createElement("div");
  pipPomodoroEl.className = "status-card pip-status pip-pomodoro";
  pipPomodoroEl.textContent = `${pomodoroPhaseEl.textContent} ${pomodoroTimeEl.textContent}`;

  pipWindow.document.body.appendChild(videoWrap);
  pipWindow.document.body.appendChild(pipStatusEl);
  pipWindow.document.body.appendChild(pipPresenceEl);
  pipWindow.document.body.appendChild(pipPomodoroEl);
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
  sparklineCtx.strokeStyle = "rgba(128,128,128,0.35)";
  sparklineCtx.lineWidth = 1;
  for (const threshold of [tooClose, tooFar]) {
    const y = yFor(threshold);
    sparklineCtx.beginPath();
    sparklineCtx.moveTo(0, y);
    sparklineCtx.lineTo(w, y);
    sparklineCtx.stroke();
  }

  const accentColor = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#ffffff";
  sparklineCtx.strokeStyle = accentColor;
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

// Drives one channel's sustained-episode logic: fires (at most once per
// episode) a notification/speech alert once the smoothed state has been
// continuously bad for `alertAfterMs`, and resets the episode once the state
// returns to good/unknown. Returns whether this channel is currently
// "alarming" (past the sustained threshold, and not snoozed).
function driveChannelAlerts(
  card: HTMLElement,
  episode: SustainedEpisode,
  isBad: boolean,
  now: number,
  alertAfterMs: number,
  snoozed: boolean,
  notifyTitle: string,
  notifyBody: string,
  speechText: string
): boolean {
  const alarming = episode.update(isBad, now, alertAfterMs, snoozed, {
    onNotify: () => maybeNotify(notifyTitle, notifyBody),
    onSpeak: () => speak(speechText),
  });
  card.classList.toggle("sustained", alarming);
  return alarming;
}

// Starts/stops the repeating customizable alarm sound as the distance
// channel enters or leaves the "alarming" state, so it plays continuously
// while distance stays bad — including while the tab is backgrounded.
function updateDistanceAlarm(shouldAlarm: boolean): void {
  const enabled = enableAlarmSoundInput.checked;
  if (shouldAlarm && enabled) {
    if (!distanceAlarmActive) {
      distanceAlarmActive = true;
      const sound = alarmSoundSelect.value as AlarmSoundId;
      const volume = Number(alarmVolumeInput.value) / 100;
      const intervalMs = Math.max(1, Number(alarmRepeatSecondsInput.value) || 20) * 1000;
      alarmSoundEngine.startRepeating(sound, volume, intervalMs);
    }
  } else if (distanceAlarmActive) {
    distanceAlarmActive = false;
    alarmSoundEngine.stopRepeating();
  }
}

// Same idea for the away/emergency channel, using its own engine/settings so
// it can run independently (louder, faster-repeating) from the distance alarm.
function updateEmergencyAlarm(shouldAlarm: boolean): void {
  if (shouldAlarm) {
    if (!emergencyAlarmActive) {
      emergencyAlarmActive = true;
      const sound = emergencySoundSelect.value as AlarmSoundId;
      const volume = Number(emergencyVolumeInput.value) / 100;
      const intervalMs = Math.max(1, Number(emergencyRepeatSecondsInput.value) || 5) * 1000;
      emergencyAlarmEngine.startRepeating(sound, volume, intervalMs);
    }
  } else if (emergencyAlarmActive) {
    emergencyAlarmActive = false;
    emergencyAlarmEngine.stopRepeating();
  }
}

// Flashes the document title while any channel is alarming and the tab is
// hidden, so switching back to another app/tab still surfaces the alert.
function updateTitleFlash(anyAlarming: boolean): void {
  if (anyAlarming && document.hidden) {
    titleFlashOn = !titleFlashOn;
    document.title = titleFlashOn ? "⚠️ Distance Checker" : ORIGINAL_TITLE;
  } else if (document.title !== ORIGINAL_TITLE) {
    document.title = ORIGINAL_TITLE;
  }
}
let lastAnyAlarming = false;
setInterval(() => updateTitleFlash(lastAnyAlarming), 1000);

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

// --- Pomodoro focus timer ---
const pomodoroSettings = loadPomodoroSettings();
focusMinutesInput.value = String(pomodoroSettings.focusMinutes);
shortBreakMinutesInput.value = String(pomodoroSettings.shortBreakMinutes);
longBreakMinutesInput.value = String(pomodoroSettings.longBreakMinutes);
cyclesBeforeLongBreakInput.value = String(pomodoroSettings.cyclesBeforeLongBreak);
pomodoroAutoStartInput.checked = pomodoroSettings.autoStartNext;
const pomodoroTimer = new PomodoroTimer(pomodoroSettings);

function phaseLabel(phase: PomodoroPhase): string {
  return phase === "focus" ? "Focus" : phase === "shortBreak" ? "Short Break" : "Long Break";
}

function formatCountdown(ms: number): string {
  const totalSec = Math.max(0, Math.ceil(ms / 1000));
  const mm = Math.floor(totalSec / 60);
  const ss = totalSec % 60;
  return `${mm}:${String(ss).padStart(2, "0")}`;
}

function renderPomodoroUI(): void {
  pomodoroPhaseEl.textContent = phaseLabel(pomodoroTimer.phase);
  pomodoroTimeEl.textContent = formatCountdown(pomodoroTimer.remainingMs);
  pomodoroEl.classList.toggle("is-focus", pomodoroTimer.phase === "focus");
  pomodoroEl.classList.toggle("is-break", pomodoroTimer.phase !== "focus");
  pomodoroEl.classList.toggle("is-running", pomodoroTimer.running);
  pomodoroStartBtn.textContent = pomodoroTimer.running ? "Pause" : "Start";

  const cycles = Math.max(1, Number(cyclesBeforeLongBreakInput.value) || 4);
  const doneInSet = pomodoroTimer.completedFocusCount % cycles;
  pomodoroDotsEl.innerHTML = "";
  for (let i = 0; i < cycles; i++) {
    const dot = document.createElement("span");
    dot.className = "pomodoro-dot" + (i < doneInSet ? " filled" : "");
    pomodoroDotsEl.appendChild(dot);
  }

  if (pipPomodoroEl) {
    pipPomodoroEl.textContent = `${pomodoroPhaseEl.textContent} ${pomodoroTimeEl.textContent}`;
  }
}

function onPomodoroPhaseEnded(nextPhase: PomodoroPhase): void {
  const title = "Pomodoro Timer";
  if (nextPhase === "focus") {
    maybeNotify(title, "Break's over — back to focus.");
    speak("Break's over. Time to focus.");
  } else if (nextPhase === "longBreak") {
    maybeNotify(title, "Great work — take a long break.");
    speak("Take a long break.");
  } else {
    maybeNotify(title, "Nice work — take a short break.");
    speak("Take a short break.");
  }
  alarmSoundEngine.playOnce("chime", (Number(alarmVolumeInput.value) || 70) / 100);
}

pomodoroStartBtn.addEventListener("click", () => {
  // Re-unlock on every click (not just "Start Camera") so the alarm can
  // never end up silent because a browser suspended an idle AudioContext.
  alarmSoundEngine.unlock();
  emergencyAlarmEngine.unlock();
  if (pomodoroTimer.running) pomodoroTimer.pause();
  else pomodoroTimer.start();
  renderPomodoroUI();
});

pomodoroResetBtn.addEventListener("click", () => {
  pomodoroTimer.reset();
  renderPomodoroUI();
});

pomodoroSkipBtn.addEventListener("click", () => {
  const result = pomodoroTimer.skip();
  if (result.nextPhase) onPomodoroPhaseEnded(result.nextPhase);
  renderPomodoroUI();
});

function applyPomodoroSettingsFromInputs(): void {
  const settings = {
    focusMinutes: Math.max(1, Number(focusMinutesInput.value) || 25),
    shortBreakMinutes: Math.max(1, Number(shortBreakMinutesInput.value) || 5),
    longBreakMinutes: Math.max(1, Number(longBreakMinutesInput.value) || 15),
    cyclesBeforeLongBreak: Math.max(1, Number(cyclesBeforeLongBreakInput.value) || 4),
    autoStartNext: pomodoroAutoStartInput.checked,
  };
  pomodoroTimer.updateSettings(settings);
  savePomodoroSettings(settings);
  renderPomodoroUI();
}

for (const input of [
  focusMinutesInput,
  shortBreakMinutesInput,
  longBreakMinutesInput,
  cyclesBeforeLongBreakInput,
  pomodoroAutoStartInput,
]) {
  input.addEventListener("change", applyPomodoroSettingsFromInputs);
}

let lastPomodoroTickAt = Date.now();
setInterval(() => {
  const now = Date.now();
  const deltaMs = now - lastPomodoroTickAt;
  lastPomodoroTickAt = now;
  const result = pomodoroTimer.tick(deltaMs);
  if (result.phaseEnded && result.nextPhase) onPomodoroPhaseEnded(result.nextPhase);
  renderPomodoroUI();
}, 1000);

renderPomodoroUI();

function scheduleNextFrame(): void {
  if (document.hidden) {
    backgroundTimerId = window.setTimeout(renderLoop, BACKGROUND_FRAME_INTERVAL_MS);
  } else {
    rafId = requestAnimationFrame(renderLoop);
  }
}

function renderLoop(): void {
  rafId = null;
  backgroundTimerId = null;
  scheduleNextFrame();
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
  let rawDistanceState: SmoothedDistanceState = null; // feeds sustained-alert smoothing only
  let rawPresenceState: SmoothedPresenceState = null;

  if (result.faceLandmarks && result.faceLandmarks.length > 0) {
    const landmarks = result.faceLandmarks[0];
    const { diameter, leftCenter, rightCenter } = averageIrisDiameterPx(
      landmarks,
      canvas.width,
      canvas.height
    );
    latestIrisDiameterPx = diameter;
    rawPresenceState = "present";

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
        rawDistanceState = "close";
      } else if (distanceCm > tooFar) {
        setStatus(`Too far (${distanceCm.toFixed(0)} cm)`, "warn");
        rawDistanceState = "far";
      } else {
        setStatus(`Good distance (${distanceCm.toFixed(0)} cm)`, "ok");
        rawDistanceState = "good";
      }
    } else {
      recordDistanceSample(now, null);
      setStatus(`Not calibrated — sit at ${calibDistanceInput.value}cm and press Calibrate`, "warn");
    }
  } else {
    recordDistanceSample(now, null);
    setStatus("No face detected", "bad");
    rawPresenceState = "away";
  }

  const smoothedDistance = distanceStateTracker.update(rawDistanceState, now);
  const smoothedPresence = presenceStateTracker.update(rawPresenceState, now);

  const distanceIsBad = smoothedDistance === "close" || smoothedDistance === "far";
  const distanceAlarming = driveChannelAlerts(
    statusCard,
    distanceEpisode,
    distanceIsBad,
    now,
    Math.max(1, Number(alertAfterInput.value) || 15) * 1000,
    isSnoozed(now),
    "Distance Checker",
    smoothedDistance === "close"
      ? "You've been sitting too close for a while"
      : "You've been sitting too far for a while",
    smoothedDistance === "close" ? "You're sitting too close" : "You're sitting too far away"
  );
  updateDistanceAlarm(distanceAlarming);

  // Away/emergency tracking only applies during an active, running Focus
  // session — stepping away on a break, or with the timer paused, is fine.
  // It deliberately ignores the general "snooze" (unlike the distance
  // channel above): the whole point is to catch you when you've forgotten
  // you're mid-session, so the only way to silence it is to pause the timer
  // or turn it off in Settings.
  const trackingActive =
    enableAwayAlertInput.checked && pomodoroTimer.phase === "focus" && pomodoroTimer.running;
  const presenceIsBad = trackingActive && smoothedPresence === "away";
  const awayThresholdMs = Math.max(1, Number(awayThresholdInput.value) || 10) * 1000;
  const presenceAlarming = driveChannelAlerts(
    presenceStatusCard,
    presenceEpisode,
    presenceIsBad,
    now,
    awayThresholdMs,
    false,
    "Focus Timer",
    "You've been away from your desk during a focus session",
    "Emergency. Get back to your desk."
  );
  updateEmergencyAlarm(presenceAlarming);

  if (!trackingActive) {
    setPresenceStatus("Away tracking paused — start a Focus session", undefined);
  } else if (smoothedPresence === "present") {
    setPresenceStatus("Present", "ok");
  } else if (presenceAlarming) {
    setPresenceStatus("⚠️ Away during focus — emergency alarm", "bad");
  } else {
    setPresenceStatus("You stepped away…", "warn");
  }

  lastAnyAlarming = distanceAlarming || presenceAlarming;
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
    // Unlock both AudioContexts here (a user gesture) so alarms can play
    // later even when triggered from a background tab/timer.
    alarmSoundEngine.unlock();
    emergencyAlarmEngine.unlock();
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

    if (rafId === null && backgroundTimerId === null) renderLoop();
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
  settingsDialog.close();
  deviceField.scrollIntoView({ behavior: "smooth", block: "center" });
});

calibrateBtn.addEventListener("click", doCalibrate);
resetCalibBtn.addEventListener("click", doResetCalibration);

testSoundBtn.addEventListener("click", () => {
  alarmSoundEngine.unlock();
  const sound = alarmSoundSelect.value as AlarmSoundId;
  const volume = Number(alarmVolumeInput.value) / 100;
  alarmSoundEngine.playOnce(sound, volume);
});

testEmergencyBtn.addEventListener("click", () => {
  emergencyAlarmEngine.unlock();
  const sound = emergencySoundSelect.value as AlarmSoundId;
  const volume = Number(emergencyVolumeInput.value) / 100;
  emergencyAlarmEngine.playOnce(sound, volume);
});

snoozeBtn.addEventListener("click", () => {
  snoozeUntil = performance.now() + SNOOZE_MS;
  updateSnoozeUI(performance.now());
  // Silence the distance alarm immediately rather than waiting for the next
  // detection frame. The away/emergency alarm is NOT snoozed here — it
  // ignores snooze entirely (see the comment above its driveChannelAlerts
  // call), so pause the Pomodoro timer instead if you need to step away.
  distanceAlarmActive = false;
  alarmSoundEngine.stopRepeating();
});

// Independent of the render loop so the countdown keeps ticking even before
// the camera stream is fully ready.
setInterval(() => updateSnoozeUI(performance.now()), 1000);
updateSnoozeUI(performance.now());

window.addEventListener("keydown", (e) => {
  if (stage.hidden) return;
  if (settingsDialog.open) return;
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
