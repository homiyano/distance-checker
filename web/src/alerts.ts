// Generic sustained-state alerting, shared by the distance and posture
// channels: a raw per-frame state is first hysteresis-smoothed (so brief
// flickers don't count), then tracked as an "episode" that only fires its
// callbacks once it has stayed bad continuously for a configured duration.

export type ChannelState = string | null;

export class SustainedStateTracker {
  private rawPendingState: ChannelState = null;
  private rawPendingSince = 0;
  private smoothedState: ChannelState = null;

  constructor(private readonly hysteresisMs: number) {}

  update(rawState: ChannelState, now: number): ChannelState {
    if (rawState !== this.rawPendingState) {
      this.rawPendingState = rawState;
      this.rawPendingSince = now;
    }
    if (this.smoothedState === null) {
      this.smoothedState = rawState;
    } else if (rawState !== this.smoothedState && now - this.rawPendingSince >= this.hysteresisMs) {
      this.smoothedState = rawState;
    }
    return this.smoothedState;
  }
}

export interface EpisodeCallbacks {
  onNotify: () => void;
  onSpeak: () => void;
}

export class SustainedEpisode {
  private sustainedSince: number | null = null;
  private notified = false;
  private spoken = false;

  // Returns true once the bad state has been sustained past `alertAfterMs`
  // and alerts aren't snoozed — i.e. this episode should currently be
  // "alarming". Fires onNotify/onSpeak at most once per episode.
  update(
    isBad: boolean,
    now: number,
    alertAfterMs: number,
    snoozed: boolean,
    callbacks: EpisodeCallbacks
  ): boolean {
    if (!isBad) {
      this.reset();
      return false;
    }
    if (this.sustainedSince === null) this.sustainedSince = now;
    const sustainedMs = now - this.sustainedSince;
    if (sustainedMs < alertAfterMs || snoozed) return false;

    if (!this.notified) {
      this.notified = true;
      callbacks.onNotify();
    }
    if (!this.spoken) {
      this.spoken = true;
      callbacks.onSpeak();
    }
    return true;
  }

  reset(): void {
    this.sustainedSince = null;
    this.notified = false;
    this.spoken = false;
  }
}
