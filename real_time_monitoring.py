import cv2
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional
from concurrency.readWriteLock import ReadWriteLock
from faceProcessing import FaceProcessing, FaceResult
from weaviateClientManager import WeaviateClientManager
from weaviate.classes.query import MetadataQuery, QueryReference

# ── Config ────────────────────────────────────────────────────────────────────

IDENTIFY_INTERVAL = 1.0
IOU_THRESHOLD     = 0.3
MAX_MISSING       = 15
NO_MATCH_DISTANCE = 0.4

# ── Tracked face ──────────────────────────────────────────────────────────────

@dataclass
class TrackedFace:
    track_id:        int
    bbox:            np.ndarray
    embedding:       np.ndarray
    last_seen:       float = field(default_factory=time.time)
    last_identified: float = 0.0
    missing_frames:  int   = 0
    name:            Optional[str]   = None
    affiliation:     Optional[str]   = None
    status:          Optional[str]   = None
    confidence:      Optional[float] = None
    identifying:     bool            = False

# ── IoU helper ────────────────────────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

# ── Tracker ───────────────────────────────────────────────────────────────────

class FaceTracker:
    def __init__(self, face_processing: FaceProcessing, client_manager: WeaviateClientManager):
        self.fp             = face_processing
        self.client_manager = client_manager
        self._tracks        = ReadWriteLock(dict())
        self._next_id       = 0
        self._id_lock       = threading.Lock()
        self._identify_pool = ThreadPoolExecutor(max_workers=2)

    def _new_id(self) -> int:
        with self._id_lock:
            tid = self._next_id
            self._next_id += 1
            return tid

    # ── Identification ────────────────────────────────────────────────────────

    def _identify_async(self, track_id: int, embedding: np.ndarray) -> None:
        try:
            name, affiliation, status, confidence = self._query_weaviate(embedding)

            with self._tracks.write() as tracks:
                track = tracks.get(track_id)
                if track is None:
                    return
                track.name            = name
                track.affiliation     = affiliation
                track.status          = status
                track.confidence      = confidence
                track.identifying     = False
                track.last_identified = time.time()

        except Exception as e:
            print(f"[Identify] Error for track {track_id}: {e}")
            with self._tracks.write() as tracks:
                track = tracks.get(track_id)
                if track:
                    track.identifying = False

    def _query_weaviate(self, embedding: np.ndarray) -> tuple:
        """Returns (name, affiliation, status, confidence) or (None, None, None, None)."""
        with self.client_manager._call() as client:
            response = client.collections.get("FaceEmbedding").query.near_vector(
                near_vector=embedding.tolist(),
                limit=1,
                return_metadata=MetadataQuery(distance=True),
                return_references=QueryReference(
                    link_on="person",
                    return_properties=["name", "affiliation", "status"]
                )
            )

        if not response.objects:
            return None, None, None, None

        obj  = response.objects[0]
        dist = obj.metadata.distance

        if dist > NO_MATCH_DISTANCE:
            return None, None, None, None

        person_refs = obj.references.get("person")
        if not person_refs or not person_refs.objects:
            return None, None, None, None

        props = person_refs.objects[0].properties
        return (
            props.get("name"),
            props.get("affiliation"),
            props.get("status"),
            round((1 - dist) * 100, 1)
        )

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> list[TrackedFace]:
        detections = self.fp.run(frame)
        now        = time.time()

        with self._tracks.write() as tracks:
            self._match_detections(tracks, detections, now)
            self._register_new(tracks, detections, now)
            self._drop_stale(tracks)
            self._schedule_identification(tracks, now)
            return list(tracks.values())

    def _match_detections(self, tracks: dict, detections: list[FaceResult], now: float) -> set:
        matched_det_ids = set()

        for track in tracks.values():
            best_iou, best_idx = 0.0, -1
            for i, det in enumerate(detections):
                if i in matched_det_ids:
                    continue
                iou = _iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_iou >= IOU_THRESHOLD:
                det                  = detections[best_idx]
                track.bbox           = det.bbox
                track.embedding      = det.embedding
                track.last_seen      = now
                track.missing_frames = 0
                matched_det_ids.add(best_idx)
            else:
                track.missing_frames += 1

        return matched_det_ids

    def _register_new(self, tracks: dict, detections: list[FaceResult], now: float) -> None:
        # Re-compute matched set to find unmatched detections
        matched = set()
        for track in tracks.values():
            best_iou, best_idx = 0.0, -1
            for i, det in enumerate(detections):
                if i in matched:
                    continue
                iou = _iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= IOU_THRESHOLD:
                matched.add(best_idx)

        for i, det in enumerate(detections):
            if i not in matched:
                tid = self._new_id()
                tracks[tid] = TrackedFace(
                    track_id=tid,
                    bbox=det.bbox,
                    embedding=det.embedding,
                )

    def _drop_stale(self, tracks: dict) -> None:
        stale = [tid for tid, t in tracks.items() if t.missing_frames > MAX_MISSING]
        for tid in stale:
            del tracks[tid]

    def _schedule_identification(self, tracks: dict, now: float) -> None:
        for track in tracks.values():
            if (now - track.last_identified) >= IDENTIFY_INTERVAL and not track.identifying:
                track.identifying = True
                self._identify_pool.submit(self._identify_async, track.track_id, track.embedding.copy())

    def stop(self):
        self._identify_pool.shutdown(wait=False)

# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_tracks(frame: np.ndarray, tracks: list[TrackedFace]) -> np.ndarray:
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        color = (0, 255, 0) if track.name else (200, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        lines = [f"#{track.track_id}"]
        if track.name:
            lines += [f"{track.name} ({track.status})", f"{track.confidence}% conf"]
        else:
            lines.append("Identifying..." if track.identifying else "Unknown")

        for j, line in enumerate(lines):
            cv2.putText(frame, line, (x1, y1 - 10 - j * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame

# ── Main loop ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fp      = FaceProcessing()
    manager = WeaviateClientManager()
    tracker = FaceTracker(fp, manager)
    cap     = cv2.VideoCapture(0)

    print("Running — press Q to quit")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = draw_tracks(frame, tracker.update(frame))
            cv2.imshow("Face Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        tracker.stop()
        cap.release()
        cv2.destroyAllWindows()