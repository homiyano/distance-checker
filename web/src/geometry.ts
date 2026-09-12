import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

// Iris landmark indices from MediaPipe's Face Landmarker (refine_landmarks / iris model).
export const LEFT_IRIS = [468, 469, 470, 471, 472];
export const RIGHT_IRIS = [473, 474, 475, 476, 477];

export type Point = [number, number];

export interface Circle {
  center: Point;
  radius: number;
}

export interface IrisMeasurement {
  diameter: number;
  center: Point;
}

export interface AverageIrisMeasurement {
  diameter: number;
  leftCenter: Point;
  rightCenter: Point;
}

// --- Minimal enclosing circle (Welzl's algorithm) for a small point set ---
function dist(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function circleFromTwo(a: Point, b: Point): Circle {
  const center: Point = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  return { center, radius: dist(a, b) / 2 };
}

function circleFromThree(a: Point, b: Point, c: Point): Circle {
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
  const center: Point = [ux, uy];
  return { center, radius: dist(center, a) };
}

function inCircle(circle: Circle, p: Point, eps = 1e-6): boolean {
  return dist(circle.center, p) <= circle.radius + eps;
}

export function minEnclosingCircle(points: Point[]): Circle {
  let circle: Circle | null = null;
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
  // points is always non-empty (LEFT_IRIS/RIGHT_IRIS have 5 entries), so circle is set.
  return circle as Circle;
}

export function irisDiameterPx(
  landmarks: NormalizedLandmark[],
  indices: number[],
  w: number,
  h: number
): IrisMeasurement {
  const pts: Point[] = indices.map((i) => [landmarks[i].x * w, landmarks[i].y * h]);
  const circle = minEnclosingCircle(pts);
  return { diameter: circle.radius * 2, center: circle.center };
}

export function averageIrisDiameterPx(
  landmarks: NormalizedLandmark[],
  w: number,
  h: number
): AverageIrisMeasurement {
  const left = irisDiameterPx(landmarks, LEFT_IRIS, w, h);
  const right = irisDiameterPx(landmarks, RIGHT_IRIS, w, h);
  return {
    diameter: (left.diameter + right.diameter) / 2,
    leftCenter: left.center,
    rightCenter: right.center,
  };
}
