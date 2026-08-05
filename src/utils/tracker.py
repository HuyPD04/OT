from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Box, box_iou, normalized_center_distance


@dataclass(frozen=True, slots=True)
class TrackedObject:
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    score: float
    class_id: int | None


@dataclass(slots=True)
class _TrackState:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int | None
    track_id: int
    hits: int = 1
    missed_frames: int = 0

    def update(self, detection: np.ndarray) -> None:
        self.x1 = float(detection[0])
        self.y1 = float(detection[1])
        self.x2 = float(detection[2])
        self.y2 = float(detection[3])
        self.score = float(detection[4])
        self.class_id = _class_id_from_detection(detection)
        self.hits += 1
        self.missed_frames = 0

    def mark_missed(self) -> None:
        self.missed_frames += 1

    def snapshot(self) -> TrackedObject:
        return TrackedObject(
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            track_id=self.track_id,
            score=self.score,
            class_id=self.class_id,
        )


class Tracker:

    def __init__(
            self,
            max_age: int,
            min_hits: int,
            iou_threshold: float,
            center_threshold: float = 1.5,
            center_weight: float = 0.35,
    ) -> None:
        if max_age < 0:
            raise ValueError("max_age must be non-negative")
        if min_hits < 1:
            raise ValueError("min_hits must be at least 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if center_threshold <= 0.0:
            raise ValueError("center_threshold must be positive")
        if center_weight < 0.0:
            raise ValueError("center_weight must be non-negative")

        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._center_threshold = center_threshold
        self._center_weight = center_weight
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def update(self, detections: np.ndarray | None = None) -> tuple[TrackedObject, ...]:
        normalized_detections = _normalize_detections(detections)
        for track in self._tracks:
            track.mark_missed()

        matches, unmatched_detection_indexes = self._match(normalized_detections)
        matched_track_indexes = {track_index for _, track_index in matches}

        for detection_index, track_index in matches:
            self._tracks[track_index].update(normalized_detections[detection_index])

        if len(normalized_detections) > 0:
            for track_index, track in enumerate(self._tracks):
                if track_index in matched_track_indexes:
                    continue
                if self._conflicts_with_any_detection(track, normalized_detections):
                    track.missed_frames = self._max_age + 1

        for detection_index in unmatched_detection_indexes:
            detection = normalized_detections[detection_index]
            self._tracks.append(
                _TrackState(
                    x1=float(detection[0]),
                    y1=float(detection[1]),
                    x2=float(detection[2]),
                    y2=float(detection[3]),
                    score=float(detection[4]),
                    class_id=_class_id_from_detection(detection),
                    track_id=self._next_id,
                )
            )
            self._next_id += 1

        self._tracks = [
            track for track in self._tracks if track.missed_frames <= self._max_age
        ]
        return self.get_tracks()

    def get_tracks(self) -> tuple[TrackedObject, ...]:
        return tuple(
            track.snapshot()
            for track in self._tracks
            if track.hits >= self._min_hits
        )

    def _match(
            self,
            detections: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int]]:
        if len(detections) == 0 or len(self._tracks) == 0:
            return [], list(range(len(detections)))

        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for track_index, track in enumerate(self._tracks):
                score = self._match_score(detection, track)
                if score is not None:
                    candidates.append((score, detection_index, track_index))

        candidates.sort(reverse=True, key=lambda candidate: candidate[0])

        matches: list[tuple[int, int]] = []
        used_detection_indexes: set[int] = set()
        used_track_indexes: set[int] = set()
        for _, detection_index, track_index in candidates:
            if (
                    detection_index in used_detection_indexes
                    or track_index in used_track_indexes
            ):
                continue
            matches.append((detection_index, track_index))
            used_detection_indexes.add(detection_index)
            used_track_indexes.add(track_index)

        unmatched_detection_indexes = [
            detection_index
            for detection_index in range(len(detections))
            if detection_index not in used_detection_indexes
        ]
        return matches, unmatched_detection_indexes

    def _match_score(
            self,
            detection: np.ndarray,
            track: _TrackState,
    ) -> float | None:
        detection_class_id = _class_id_from_detection(detection)
        if (
                detection_class_id is not None
                and track.class_id is not None
                and detection_class_id != track.class_id
        ):
            return None

        track_box: Box = (track.x1, track.y1, track.x2, track.y2)
        detection_box = _box_from_detection(detection)
        iou = box_iou(track_box, detection_box)
        center_distance = normalized_center_distance(track_box, detection_box)
        if iou < self._iou_threshold and center_distance > self._center_threshold:
            return None

        center_score = max(0.0, 1.0 - (center_distance / self._center_threshold))
        return iou + (self._center_weight * center_score)

    def _conflicts_with_any_detection(
            self,
            track: _TrackState,
            detections: np.ndarray,
    ) -> bool:
        track_box: Box = (track.x1, track.y1, track.x2, track.y2)
        for detection in detections:
            detection_box = _box_from_detection(detection)
            if box_iou(track_box, detection_box) >= self._iou_threshold:
                return True
            if (
                    normalized_center_distance(track_box, detection_box)
                    <= self._center_threshold
            ):
                return True
        return False


def _normalize_detections(detections: np.ndarray | None) -> np.ndarray:
    if detections is None or detections.size == 0:
        return np.empty((0, 6), dtype=float)

    normalized = np.asarray(detections, dtype=float)
    if normalized.ndim == 1:
        normalized = normalized.reshape(1, -1)
    if normalized.shape[1] < 5:
        raise ValueError("detections must have at least 5 columns")
    if normalized.shape[1] == 5:
        class_ids = np.full((normalized.shape[0], 1), np.nan, dtype=float)
        normalized = np.concatenate((normalized[:, :5], class_ids), axis=1)
    return normalized[:, :6]


def _class_id_from_detection(detection: np.ndarray) -> int | None:
    if len(detection) < 6 or np.isnan(detection[5]):
        return None
    return int(detection[5])


def _box_from_detection(detection: np.ndarray) -> Box:
    return (
        float(detection[0]),
        float(detection[1]),
        float(detection[2]),
        float(detection[3]),
    )
