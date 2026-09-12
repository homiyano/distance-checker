const POMODORO_SETTINGS_KEY = "distance-checker:pomodoro-settings";

export type PomodoroPhase = "focus" | "shortBreak" | "longBreak";

export interface PomodoroSettings {
  focusMinutes: number;
  shortBreakMinutes: number;
  longBreakMinutes: number;
  cyclesBeforeLongBreak: number;
  autoStartNext: boolean;
}

export const DEFAULT_POMODORO_SETTINGS: PomodoroSettings = {
  focusMinutes: 25,
  shortBreakMinutes: 5,
  longBreakMinutes: 15,
  cyclesBeforeLongBreak: 4,
  autoStartNext: true,
};

export function loadPomodoroSettings(): PomodoroSettings {
  try {
    const raw = localStorage.getItem(POMODORO_SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_POMODORO_SETTINGS };
    return { ...DEFAULT_POMODORO_SETTINGS, ...(JSON.parse(raw) as Partial<PomodoroSettings>) };
  } catch {
    return { ...DEFAULT_POMODORO_SETTINGS };
  }
}

export function savePomodoroSettings(settings: PomodoroSettings): void {
  localStorage.setItem(POMODORO_SETTINGS_KEY, JSON.stringify(settings));
}

function phaseDurationMs(phase: PomodoroPhase, settings: PomodoroSettings): number {
  const minutes =
    phase === "focus"
      ? settings.focusMinutes
      : phase === "shortBreak"
        ? settings.shortBreakMinutes
        : settings.longBreakMinutes;
  return Math.max(1, minutes) * 60_000;
}

export interface PomodoroTickResult {
  phaseEnded: boolean;
  nextPhase: PomodoroPhase | null;
}

// A minimal, dependency-free Pomodoro state machine. Driven by an external
// ~1Hz interval that passes the actual elapsed wall-clock delta, so the
// countdown stays correct even if that interval gets throttled while the tab
// is backgrounded (it just catches up in bigger steps rather than losing time).
export class PomodoroTimer {
  phase: PomodoroPhase = "focus";
  remainingMs: number;
  running = false;
  completedFocusCount = 0;

  constructor(private settings: PomodoroSettings) {
    this.remainingMs = phaseDurationMs(this.phase, this.settings);
  }

  updateSettings(settings: PomodoroSettings): void {
    this.settings = settings;
    if (!this.running) {
      // Only snap the visible countdown to the new duration while idle —
      // don't yank time out from under an in-progress session.
      this.remainingMs = phaseDurationMs(this.phase, this.settings);
    }
  }

  start(): void {
    this.running = true;
  }

  pause(): void {
    this.running = false;
  }

  reset(): void {
    this.running = false;
    this.phase = "focus";
    this.completedFocusCount = 0;
    this.remainingMs = phaseDurationMs(this.phase, this.settings);
  }

  // Skips straight to the next phase (counts as ending the current one).
  skip(): PomodoroTickResult {
    return this.advance();
  }

  private advance(): PomodoroTickResult {
    let nextPhase: PomodoroPhase;
    if (this.phase === "focus") {
      this.completedFocusCount += 1;
      nextPhase =
        this.completedFocusCount % this.settings.cyclesBeforeLongBreak === 0 ? "longBreak" : "shortBreak";
    } else {
      nextPhase = "focus";
    }
    this.phase = nextPhase;
    this.remainingMs = phaseDurationMs(nextPhase, this.settings);
    this.running = this.settings.autoStartNext;
    return { phaseEnded: true, nextPhase };
  }

  // Call roughly once a second with the actual elapsed ms since the last
  // call. Returns phaseEnded=true (once) the moment a phase's countdown
  // reaches zero, so the caller can fire a sound/notification.
  tick(deltaMs: number): PomodoroTickResult {
    if (!this.running) return { phaseEnded: false, nextPhase: null };
    this.remainingMs -= deltaMs;
    if (this.remainingMs > 0) return { phaseEnded: false, nextPhase: null };
    return this.advance();
  }
}
