import cv2
import time
import tkinter
import threading
import numpy as np
from PIL import Image
from typing import Optional

import weaviate
from minio import Minio
from weaviate.classes.query import MetadataQuery, QueryReference

from concurrent.futures import ThreadPoolExecutor

from facialRecognition.faceProcessing import FaceProcessing
from facialRecognition.trackedFace import TrackedFace
from forms import create_form, update_form
from concurrency.readWriteLock import ReadWriteLock


class FaceTracker:
    TRACK_SIMILARITY_THRESH = 0.4
    DB_DISTANCE_THRESH = 0.4
    IDENTIFY_INTERVAL = 1.0
    MAX_STALE_FRAMES = 15

    def __init__(self, face_processing: FaceProcessing, root: tkinter.Tk, wv_client: weaviate.WeaviateClient, mn_client: Minio):
        self.fp = face_processing
        self.root = root
        self.wv_client = wv_client
        self.mn_client = mn_client
        self.tracks: ReadWriteLock[dict[int, TrackedFace]] = ReadWriteLock(dict())
        self.pool = ThreadPoolExecutor(max_workers=2)
        self._next_id = 0

        self._frame_lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None

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
                if track is None:
                    return
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
                if track:
                    track.identifying = False

    def _query_weaviate(self, embedding: np.ndarray) -> tuple[Optional[str], Optional[str], Optional[str], Optional[float]]:
        client = weaviate.connect_to_local()
        try:
            response = client.collections.get("FaceEmbedding").query.near_vector(
                near_vector=embedding.tolist(),
                limit=1,
                return_metadata=MetadataQuery(distance=True),
                return_references=QueryReference(
                    link_on="person",
                    return_properties=["name", "affiliation", "status"],
                ),
            )
            if not response.objects:
                return None, None, None, None

            obj = response.objects[0]
            dist = obj.metadata.distance
            if dist > self.DB_DISTANCE_THRESH:
                return None, None, None, None

            person_refs = obj.references.get("person")
            if not person_refs or not person_refs.objects:
                return None, None, None, None

            props = person_refs.objects[0].properties
            confidence = round((1 - dist) * 100, 1)
            return props.get("name"), props.get("affiliation"), props.get("status"), confidence
        finally:
            client.close()

    def _dispatch_identification(self, track: TrackedFace, now: float) -> None:
        if not track.identifying and (now - track.last_identified) >= self.IDENTIFY_INTERVAL:
            track.identifying = True
            self.pool.submit(self._identify_async, track.id, track.embedding.copy())

    def update(self, frame: np.ndarray) -> set[int]:
        with self._frame_lock:
            self._current_frame = frame.copy()
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

            stale = [fid for fid, tf in tracks.items() if fid not in updated_ids and tf.missing_frames > self.MAX_STALE_FRAMES]
            for fid in stale: del tracks[fid]
            for fid, tf in tracks.items():
                if fid not in updated_ids:
                    tf.missing_frames += 1

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
        color = {"approved": (0, 255, 0), "unknown": (128, 128, 128), "unapproved": (0, 0, 255)}.get(tf.status or "unknown", (128, 128, 128))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        parts = [tf.name or f"ID {fid}"]
        if tf.affiliation: parts.append(tf.affiliation)
        if tf.status: parts.append(tf.status)
        if tf.confidence is not None: parts.append(f"{tf.confidence:.0f}%")
        label = "  |  ".join(parts)

        cv2.putText(frame, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def mouse_callback(self, event: int, x: int, y: int, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        with self._frame_lock:
            frame_snap = self._current_frame  # already a copy from update()

        clicked_fid: Optional[int] = None
        clicked_bbox: Optional[tuple] = None
        clicked_status: Optional[str] = None
        clicked_snapshot: Optional[TrackedFace] = None

        with self.tracks.read() as tracks:
            for fid, tf in tracks.items():
                x1, y1, x2, y2 = map(int, tf.bbox)
                if (x1 <= x <= x2) and (y1 <= y <= y2):
                    clicked_fid = fid
                    clicked_bbox = (x1, y1, x2, y2)
                    clicked_status = tf.status

                    clicked_snapshot = TrackedFace(
                        id=tf.id,
                        bbox=tf.bbox,
                        embedding=tf.embedding.copy(),
                        last_identified=tf.last_identified,
                        missing_frames=tf.missing_frames,
                        name=tf.name,
                        affiliation=tf.affiliation,
                        status=tf.status,
                        confidence=tf.confidence,
                        identifying=tf.identifying,
                    )
                    break

        if clicked_fid is None:
            print(f"[Click] ({x},{y}) — no face hit")
            return

        print(f"[Click] Track {clicked_fid} [{clicked_snapshot.name or 'unknown'}]")

        pil_img = None
        if frame_snap is not None:
            x1, y1, x2, y2 = clicked_bbox
            crop = frame_snap[y1:y2, x1:x2]
            if crop.size > 0: 
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


        wv, mn = self.wv_client, self.mn_client
        if clicked_status is None:
            self.root.after(0, lambda: create_form(self.root, wv, mn, clicked_fid, clicked_snapshot, pil_img))
        else:
            self.root.after(0, lambda: update_form(self.root, wv, mn, clicked_fid, clicked_snapshot, pil_img))