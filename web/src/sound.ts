// Customizable Web Audio alarm: a few synthesized tone patterns (no audio
// files to fetch/bundle), a volume control, and a repeat-while-bad mode so
// the alarm keeps sounding even if the tab is in the background or the user
// is working in another app.

export type AlarmSoundId = "chime" | "beep" | "urgent" | "siren";

interface Tone {
  freq: number;
  startMs: number;
  durationMs: number;
  type?: OscillatorType;
}

const PATTERNS: Record<AlarmSoundId, Tone[]> = {
  chime: [
    { freq: 660, startMs: 0, durationMs: 180, type: "sine" },
    { freq: 880, startMs: 160, durationMs: 320, type: "sine" },
  ],
  beep: [{ freq: 880, startMs: 0, durationMs: 220, type: "square" }],
  urgent: [
    { freq: 1046, startMs: 0, durationMs: 110, type: "square" },
    { freq: 1046, startMs: 150, durationMs: 110, type: "square" },
    { freq: 1046, startMs: 300, durationMs: 110, type: "square" },
  ],
  siren: [
    { freq: 800, startMs: 0, durationMs: 160, type: "sawtooth" },
    { freq: 1200, startMs: 160, durationMs: 160, type: "sawtooth" },
    { freq: 800, startMs: 320, durationMs: 160, type: "sawtooth" },
    { freq: 1200, startMs: 480, durationMs: 160, type: "sawtooth" },
  ],
};

export class AlarmSoundEngine {
  private ctx: AudioContext | null = null;
  private repeatTimerId: number | null = null;

  // Call from within a user-gesture handler (e.g. "Start Camera" or "Test
  // sound") at least once — autoplay policies block a freshly-created
  // AudioContext otherwise. Safe to call repeatedly.
  unlock(): void {
    this.ensureContext();
  }

  private ensureContext(): AudioContext | null {
    if (this.ctx) {
      if (this.ctx.state === "suspended") void this.ctx.resume();
      return this.ctx;
    }
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    this.ctx = new Ctor();
    return this.ctx;
  }

  playOnce(sound: AlarmSoundId, volume: number): void {
    const ctx = this.ensureContext();
    if (!ctx) return;
    const peak = Math.max(0, Math.min(1, volume));
    if (peak <= 0) return;
    const startAt = ctx.currentTime;
    for (const tone of PATTERNS[sound]) {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = tone.type ?? "sine";
      osc.frequency.value = tone.freq;
      const t0 = startAt + tone.startMs / 1000;
      const t1 = t0 + tone.durationMs / 1000;
      gain.gain.setValueAtTime(0, t0);
      gain.gain.linearRampToValueAtTime(peak, t0 + 0.015);
      gain.gain.linearRampToValueAtTime(0, t1);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(t0);
      osc.stop(t1 + 0.02);
    }
  }

  startRepeating(sound: AlarmSoundId, volume: number, intervalMs: number): void {
    this.stopRepeating();
    this.playOnce(sound, volume);
    this.repeatTimerId = window.setInterval(() => this.playOnce(sound, volume), Math.max(1000, intervalMs));
  }

  stopRepeating(): void {
    if (this.repeatTimerId !== null) {
      window.clearInterval(this.repeatTimerId);
      this.repeatTimerId = null;
    }
  }

  get isRepeating(): boolean {
    return this.repeatTimerId !== null;
  }
}
