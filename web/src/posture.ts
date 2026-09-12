const POSTURE_CALIBRATION_KEY = "distance-checker:posture-calibration";

export interface HeadPose {
  pitchDeg: number;
  yawDeg: number;
  rollDeg: number;
}

export interface PostureCalibration {
  pitchDeg: number;
  rollDeg: number;
}

export type PostureIssue = "slouching" | "tilted" | null;

export interface PostureAssessment {
  pitchDeltaDeg: number;
  rollDeltaDeg: number;
  issue: PostureIssue;
}

// `data` is MediaPipe FaceLandmarker's `facialTransformationMatrixes[i].data`:
// a column-major 4x4 matrix mapping the canonical face model into camera
// space (the same layout consumed by e.g. three.js's Matrix4.fromArray). We
// only need the upper-left 3x3 rotation block.
export function headPoseFromTransformationMatrix(data: ArrayLike<number>): HeadPose {
  const at = (row: number, col: number) => data[col * 4 + row];
  const m13 = at(0, 2);
  const m21 = at(1, 0);
  const m22 = at(1, 1);
  const m23 = at(1, 2);
  const m31 = at(2, 0);
  const m33 = at(2, 2);
  const m11 = at(0, 0);

  // Tait-Bryan extraction for intrinsic rotation order Y-X-Z (yaw, then
  // pitch, then roll), the convention MediaPipe's own face-effect samples
  // use when feeding this matrix into a 3D scene.
  const clampedNeg23 = Math.max(-1, Math.min(1, -m23));
  const pitch = Math.asin(clampedNeg23);
  let yaw: number;
  let roll: number;
  if (Math.abs(m23) < 0.9999999) {
    yaw = Math.atan2(m13, m33);
    roll = Math.atan2(m21, m22);
  } else {
    // Gimbal lock: roll and yaw become ambiguous, so pin roll to 0.
    yaw = Math.atan2(-m31, m11);
    roll = 0;
  }

  const toDeg = (rad: number) => (rad * 180) / Math.PI;
  return { pitchDeg: toDeg(pitch), yawDeg: toDeg(yaw), rollDeg: toDeg(roll) };
}

export function loadPostureCalibration(): PostureCalibration | null {
  try {
    const raw = localStorage.getItem(POSTURE_CALIBRATION_KEY);
    return raw ? (JSON.parse(raw) as PostureCalibration) : null;
  } catch {
    return null;
  }
}

export function savePostureCalibration(cal: PostureCalibration): void {
  localStorage.setItem(POSTURE_CALIBRATION_KEY, JSON.stringify(cal));
}

export function clearPostureCalibration(): void {
  localStorage.removeItem(POSTURE_CALIBRATION_KEY);
}

export function calibratePostureFromPose(pose: HeadPose): PostureCalibration {
  return { pitchDeg: pose.pitchDeg, rollDeg: pose.rollDeg };
}

// Deviation is measured relative to the user's own calibrated "good posture"
// baseline rather than fixed absolute angles, since neutral head pose varies
// a lot by camera placement, monitor height, and person.
export function assessPosture(
  current: HeadPose,
  baseline: PostureCalibration,
  thresholdDeg: number
): PostureAssessment {
  const pitchDeltaDeg = current.pitchDeg - baseline.pitchDeg;
  const rollDeltaDeg = current.rollDeg - baseline.rollDeg;
  const pitchBad = Math.abs(pitchDeltaDeg) > thresholdDeg;
  const rollBad = Math.abs(rollDeltaDeg) > thresholdDeg;

  let issue: PostureIssue = null;
  if (pitchBad && rollBad) {
    // Report whichever has drifted further past its threshold so the status
    // text doesn't flicker between the two when both drift at once.
    issue = Math.abs(pitchDeltaDeg) >= Math.abs(rollDeltaDeg) ? "slouching" : "tilted";
  } else if (pitchBad) {
    issue = "slouching";
  } else if (rollBad) {
    issue = "tilted";
  }

  return { pitchDeltaDeg, rollDeltaDeg, issue };
}
