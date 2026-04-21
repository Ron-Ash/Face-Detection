import cv2
import time
import weaviate
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from weaviate.classes.query import MetadataQuery, QueryReference

from faceProcessing import FaceProcessing
from concurrency.readWriteLock import ReadWriteLock


@dataclass
class TrackedFace:
    id: int
    bbox: np.ndarray
    embedding: np.ndarray
    last_identified: float = 0.0
    missing_frames: int = 0
    name: Optional[str] = None
    affiliation: Optional[str] = None
    status: Optional[str] = None
    confidence: Optional[float] = None
    identifying: bool = False


class FaceTracker:
    TRACK_SIMILARITY_THRESH = 0.4
    DB_DISTANCE_THRESH = 0.4
    IDENTIFY_INTERVAL = 1.0
    MAX_STALE_FRAMES = 15

    def __init__(self, face_processing: FaceProcessing, client: weaviate.WeaviateClient):
        self.fp = face_processing
        self.client = client
        self.tracks: ReadWriteLock[dict[int, TrackedFace]] = ReadWriteLock(dict())
        self.pool = ThreadPoolExecutor(max_workers=2)
        self._next_id = 0

    def stop(self, wait: bool = True) -> None:
        self.pool.shutdown(wait=wait)

    def _next_track_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def _best_match(self, embedding: np.ndarray, tracks: dict[int, TrackedFace]) -> Optional[int]:
        best_id: Optional[int] = None
        best_score = -1.0
        for fid, tracked in tracks.items():
            score = self._cosine_similarity(tracked.embedding, embedding)
            if score > best_score and score > self.TRACK_SIMILARITY_THRESH:
                best_id = fid
                best_score = score
        return best_id


    def _identify_async(self, track_id: int, embedding: np.ndarray) -> None:
        try:
            name, affiliation, status, confidence = self._query_weaviate(embedding)
            with self.tracks.write() as tracks:
                track = tracks.get(track_id)
                if track is None: return

                track.name = name
                track.affiliation = affiliation
                track.status = status
                track.confidence = confidence
                track.identifying = False
                track.last_identified = time.time()
        except Exception as e:
            print(f"[Identify] Error for track {track_id}: {e}")
            with self.tracks.write() as tracks:
                track = tracks.get(track_id)
                if track: track.identifying = False

    def _query_weaviate(self, embedding: np.ndarray) -> tuple[Optional[str], Optional[str], Optional[str], Optional[float]]:
        response = self.client.collections.get("FaceEmbedding").query.near_vector(
            near_vector=embedding.tolist(),
            limit=1,
            return_metadata=MetadataQuery(distance=True),
            return_references=QueryReference(
                link_on="person",
                return_properties=["name", "affiliation", "status"],
            ),
        )
        if not response.objects: return None, None, None, None

        obj = response.objects[0]
        dist = obj.metadata.distance
        if dist > self.DB_DISTANCE_THRESH: return None, None, None, None

        person_refs = obj.references.get("person")
        if not person_refs or not person_refs.objects: return None, None, None, None

        props = person_refs.objects[0].properties
        confidence = round((1 - dist) * 100, 1)
        return props.get("name"), props.get("affiliation"), props.get("status"), confidence

    def _dispatch_identification(self, track: TrackedFace, now: float) -> None:
        if not track.identifying and (now - track.last_identified) >= self.IDENTIFY_INTERVAL:
            track.identifying = True
            self.pool.submit(self._identify_async, track.id, track.embedding.copy())

    def update(self, frame: np.ndarray) -> set[int]:
        faces = self.fp.run(frame)
        updated_ids: set[int] = set()
        now = time.time()

        with self.tracks.write() as tracks:
            for face in faces:
                fid = self._best_match(face.embedding, tracks)
                if fid is None:
                    fid = self._next_track_id()
                    tracks[fid] = TrackedFace(fid, face.bbox, face.embedding)
                else:
                    tf = tracks[fid]
                    tf.bbox = face.bbox
                    tf.embedding = face.embedding
                    tf.missing_frames = 0

                updated_ids.add(fid)
                self._dispatch_identification(tracks[fid], now)

            stale = []
            for fid, tf in tracks.items():
                if fid not in updated_ids:
                    tf.missing_frames += 1
                    if tf.missing_frames > self.MAX_STALE_FRAMES:
                        stale.append(fid)
            for fid in stale: del tracks[fid]

        return updated_ids

    def update_frame(self, frame: np.ndarray) -> np.ndarray:
        updated_ids = self.update(frame)

        with self.tracks.read() as tracks:
            for fid, tf in tracks.items():
                if fid not in updated_ids:
                    continue
                self._draw_track(frame, fid, tf)

        return frame

    @staticmethod
    def _draw_track(frame: np.ndarray, fid: int, tf: TrackedFace) -> None:
        x1, y1, x2, y2 = map(int, tf.bbox)
        color = { "approved": (0, 255, 0), "unknown": (128, 128, 128), "unapproved": (0, 0, 255)}.get(tf.status or "unknown", (128, 128, 128))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        parts = [tf.name or f"ID {fid}"]
        if tf.affiliation: parts.append(tf.affiliation)
        if tf.status: parts.append(tf.status)
        if tf.confidence is not None: parts.append(f"{tf.confidence:.0f}%")
        label = "  |  ".join(parts)

        cv2.putText(frame, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    fp = FaceProcessing()
    client = weaviate.connect_to_local()
    face_tracker = FaceTracker(fp, client)

    try:
        while True:
            returned, frame = capture.read()
            if not returned: break
            frame = face_tracker.update_frame(frame)
            cv2.imshow("Face Tracker (Press Shift+Q to close)", frame)
            if cv2.waitKey(10) & 0xFF == ord("Q"): break
    finally:
        face_tracker.stop()
        capture.release()
        cv2.destroyAllWindows()
        client.close()