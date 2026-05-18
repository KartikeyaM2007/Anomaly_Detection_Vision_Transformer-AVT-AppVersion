from __future__ import annotations

import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class Track:
    track_id: int
    label: str
    box: tuple[int, int, int, int]
    centroid: tuple[float, float]
    last_seen: float
    velocity: float = 0.0
    motion: float = 0.0
    emotion: str = "unknown"
    risk: float = 0.0
    history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=12))


class LiveIntelligence:
    """Fast per-frame person tracking and behavior scoring for desktop realtime mode."""

    def __init__(self) -> None:
        self.background = cv2.createBackgroundSubtractorMOG2(history=90, varThreshold=32, detectShadows=False)
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.frame_index = 0
        self.score_window: deque[float] = deque(maxlen=20)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.face_cascade = self._load_cascade("haarcascade_frontalface_default.xml")
        self.smile_cascade = self._load_cascade("haarcascade_smile.xml")
        self.eye_cascade = self._load_cascade("haarcascade_eye.xml")

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1
        self.frame_index = 0
        self.score_window.clear()
        self.background = cv2.createBackgroundSubtractorMOG2(history=90, varThreshold=32, detectShadows=False)

    def analyze(self, frame_rgb: np.ndarray, threshold: float) -> dict[str, Any]:
        now = time.time()
        self.frame_index += 1
        height, width = frame_rgb.shape[:2]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self._motion_boxes(frame_bgr)
        if self.frame_index % 8 == 1 and min(height, width) >= 240:
            boxes = self._merge_boxes(boxes + self._hog_boxes(frame_bgr))
        faces = self._faces(gray)
        boxes = self._merge_boxes(boxes + [self._expand_face_box(face, width, height) for face in faces])
        self._update_tracks(boxes, gray, now)
        self._expire_tracks(now)

        motion_score = min(1.0, sum(track.motion for track in self.tracks.values()) / 240.0)
        velocity_score = min(1.0, sum(track.velocity for track in self.tracks.values()) / 180.0)
        proximity_score = self._proximity_score()
        expression_score = max((self._emotion_risk(track.emotion) for track in self.tracks.values()), default=0.0)
        people_factor = min(1.0, len(self.tracks) / 3.0)
        instant_score = min(
            1.0,
            0.40 * motion_score
            + 0.24 * velocity_score
            + 0.18 * proximity_score
            + 0.12 * expression_score
            + 0.06 * people_factor,
        )
        self.score_window.append(instant_score)
        smoothed_score = max(instant_score, float(np.mean(self.score_window)) if self.score_window else instant_score)
        prediction = "ANOMALY" if smoothed_score >= threshold else "NORMAL"

        people = []
        for track in sorted(self.tracks.values(), key=lambda item: item.track_id):
            track.risk = min(
                1.0,
                0.45 * min(track.motion / 140.0, 1.0)
                + 0.35 * min(track.velocity / 110.0, 1.0)
                + 0.20 * self._emotion_risk(track.emotion),
            )
            people.append(
                {
                    "id": track.track_id,
                    "label": track.label,
                    "box": track.box,
                    "emotion": track.emotion,
                    "risk": track.risk,
                    "motion": track.motion,
                    "velocity": track.velocity,
                }
            )

        return {
            "ready": True,
            "status": "instant",
            "result": {
                "prob_anomaly": smoothed_score,
                "prob_normal": 1.0 - smoothed_score,
                "prediction": prediction,
                "confidence": max(smoothed_score, 1.0 - smoothed_score),
                "basis": "motion_person_expression",
            },
            "instant_score": instant_score,
            "motion_score": motion_score,
            "velocity_score": velocity_score,
            "proximity_score": proximity_score,
            "expression_score": expression_score,
            "person_count": len(people),
            "people": people,
            "events": [],
        }

    def draw_overlay(self, frame_rgb: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
        frame = frame_rgb.copy()
        result = payload.get("result") or {}
        prediction = str(result.get("prediction", "NORMAL"))
        score = float(result.get("prob_anomaly", 0.0))
        main_color = (244, 33, 46) if prediction == "ANOMALY" else (0, 186, 124)
        for person in payload.get("people") or []:
            x, y, w, h = [int(v) for v in person.get("box", (0, 0, 0, 0))]
            risk = float(person.get("risk", 0.0))
            color = (244, 33, 46) if risk >= 0.55 else (0, 186, 124)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{person.get('label', 'Person')} {risk * 100:.0f}% {person.get('emotion', 'unknown')}"
            cv2.rectangle(frame, (x, max(0, y - 24)), (min(frame.shape[1] - 1, x + 260), y), (0, 0, 0), -1)
            cv2.putText(frame, label, (x + 5, max(16, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (12, 12), (min(frame.shape[1] - 12, 480), 92), (0, 0, 0), -1)
        cv2.rectangle(frame, (12, 12), (min(frame.shape[1] - 12, 480), 92), main_color, 2)
        cv2.putText(
            frame,
            f"{prediction}  {score * 100:.1f}%",
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            main_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"people {int(payload.get('person_count', 0))} | cam {float(payload.get('camera_fps', 0.0)):.1f} | ai {float(payload.get('processed_fps', 0.0)):.1f}",
            (24, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (231, 233, 234),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _motion_boxes(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (max(160, width // 2), max(120, height // 2)))
        fg = self.background.apply(small)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.dilate(fg, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sx = width / small.shape[1]
        sy = height / small.shape[0]
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < small.shape[0] * small.shape[1] * 0.006:
                continue
            box = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
            if box[2] >= 18 and box[3] >= 28:
                boxes.append(box)
        return boxes[:8]

    def _hog_boxes(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        resized = frame_bgr
        scale = 1.0
        if frame_bgr.shape[1] > 640:
            scale = 640.0 / frame_bgr.shape[1]
            resized = cv2.resize(frame_bgr, (640, int(frame_bgr.shape[0] * scale)))
        boxes, _ = self.hog.detectMultiScale(resized, winStride=(8, 8), padding=(8, 8), scale=1.08)
        return [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for x, y, w, h in boxes[:6]]

    def _faces(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self.face_cascade.empty():
            return []
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(32, 32))
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces[:6]]

    def _expand_face_box(self, face: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
        x, y, w, h = face
        nx = max(0, int(x - w * 0.9))
        ny = max(0, int(y - h * 0.7))
        nw = min(width - nx, int(w * 2.8))
        nh = min(height - ny, int(h * 4.2))
        return nx, ny, nw, nh

    def _update_tracks(self, boxes: list[tuple[int, int, int, int]], gray: np.ndarray, now: float) -> None:
        assigned: set[int] = set()
        for box in boxes:
            centroid = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
            best_id = None
            best_dist = 80.0
            for track_id, track in self.tracks.items():
                if track_id in assigned:
                    continue
                dist = math.dist(centroid, track.centroid)
                if dist < best_dist:
                    best_id = track_id
                    best_dist = dist
            if best_id is None:
                label = f"Person {chr(64 + min(self.next_id, 26))}"
                self.tracks[self.next_id] = Track(self.next_id, label, box, centroid, now)
                self.tracks[self.next_id].history.append(centroid)
                assigned.add(self.next_id)
                self.next_id += 1
                continue
            track = self.tracks[best_id]
            dt = max(now - track.last_seen, 0.03)
            velocity = math.dist(centroid, track.centroid) / dt
            track.velocity = 0.65 * track.velocity + 0.35 * velocity
            track.motion = 0.70 * track.motion + 0.30 * min(255.0, velocity)
            track.box = box
            track.centroid = centroid
            track.last_seen = now
            track.history.append(centroid)
            track.emotion = self._estimate_emotion(gray, box)
            assigned.add(best_id)

    def _expire_tracks(self, now: float) -> None:
        stale = [track_id for track_id, track in self.tracks.items() if now - track.last_seen > 1.6]
        for track_id in stale:
            self.tracks.pop(track_id, None)

    def _estimate_emotion(self, gray: np.ndarray, box: tuple[int, int, int, int]) -> str:
        if self.smile_cascade.empty() or self.eye_cascade.empty():
            return "unknown"
        x, y, w, h = box
        roi = gray[max(0, y) : max(0, y) + max(1, h // 2), max(0, x) : max(0, x) + max(1, w)]
        if roi.size == 0:
            return "unknown"
        smiles = self.smile_cascade.detectMultiScale(roi, scaleFactor=1.7, minNeighbors=16, minSize=(18, 10))
        if len(smiles) > 0:
            return "smiling"
        eyes = self.eye_cascade.detectMultiScale(roi, scaleFactor=1.15, minNeighbors=4, minSize=(10, 10))
        if len(eyes) >= 2:
            return "focused"
        if len(eyes) == 1:
            return "tense"
        return "unknown"

    def _proximity_score(self) -> float:
        tracks = list(self.tracks.values())
        if len(tracks) < 2:
            return 0.0
        best = 0.0
        for i, left in enumerate(tracks):
            for right in tracks[i + 1 :]:
                dist = math.dist(left.centroid, right.centroid)
                avg_size = (left.box[2] + left.box[3] + right.box[2] + right.box[3]) / 4.0
                if avg_size <= 0:
                    continue
                best = max(best, max(0.0, 1.0 - dist / (avg_size * 2.2)))
        return best

    def _emotion_risk(self, emotion: str) -> float:
        return {"tense": 0.55, "focused": 0.25, "unknown": 0.15, "smiling": 0.0}.get(emotion, 0.15)

    def _load_cascade(self, filename: str) -> cv2.CascadeClassifier:
        candidates = []
        if getattr(cv2.data, "haarcascades", None):
            candidates.append(Path(cv2.data.haarcascades) / filename)
        if getattr(sys, "frozen", False):
            base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
            candidates.extend(
                [
                    base / "cv2" / "data" / filename,
                    base / "_internal" / "cv2" / "data" / filename,
                    Path(sys.executable).parent / "_internal" / "cv2" / "data" / filename,
                ]
            )
        for path in candidates:
            if path.exists():
                cascade = cv2.CascadeClassifier(str(path))
                if not cascade.empty():
                    return cascade
        return cv2.CascadeClassifier()

    def _merge_boxes(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        merged: list[tuple[int, int, int, int]] = []
        for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
            if box[2] <= 0 or box[3] <= 0:
                continue
            if any(self._iou(box, existing) > 0.35 for existing in merged):
                continue
            merged.append(box)
        return merged[:8]

    def _iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / max(union, 1)
