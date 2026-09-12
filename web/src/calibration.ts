const CALIBRATION_KEY = "distance-checker:calibration";

export interface Calibration {
  focalLengthPx: number;
  knownDistanceMm: number;
  irisDiameterMm: number;
  diameterPxAtCalibration: number;
}

export function loadCalibration(): Calibration | null {
  try {
    const raw = localStorage.getItem(CALIBRATION_KEY);
    return raw ? (JSON.parse(raw) as Calibration) : null;
  } catch {
    return null;
  }
}

export function saveCalibration(cal: Calibration): void {
  localStorage.setItem(CALIBRATION_KEY, JSON.stringify(cal));
}

export function clearCalibration(): void {
  localStorage.removeItem(CALIBRATION_KEY);
}

export function distanceMmFromCalibration(cal: Calibration, diameterPx: number): number {
  if (diameterPx <= 0) return NaN;
  return (cal.irisDiameterMm * cal.focalLengthPx) / diameterPx;
}

export function calibrateFromMeasurement(
  knownDistanceMm: number,
  diameterPx: number,
  irisDiameterMm: number
): Calibration {
  const focalLengthPx = (diameterPx * knownDistanceMm) / irisDiameterMm;
  return { focalLengthPx, knownDistanceMm, irisDiameterMm, diameterPxAtCalibration: diameterPx };
}
