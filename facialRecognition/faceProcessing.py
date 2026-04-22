import cv2
import numpy as np
from dataclasses import dataclass

from insightface.app import FaceAnalysis

@dataclass
class FaceResult:
    embedding: np.ndarray
    bbox: np.ndarray
    det_score: float
    kps: np.ndarray
    age: int | None = None
    gender: str | None = None


class FaceProcessing:
    def __init__(self, threshold: float = 0.6):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.threshold = threshold

    def _face_embedding(self, face):
        emb = face.embedding
        norm = np.linalg.norm(emb)
        if norm == 0: return emb
        return emb / norm

    def run(self, image: np.ndarray):
        faces = self.app.get(image)
        return [
            FaceResult(
                embedding=self._face_embedding(face),
                bbox=face.bbox.astype(int),
                det_score=float(face.det_score),
                kps=face.kps,
                age=getattr(face, 'age', None),
                gender=getattr(face, 'gender', None)
            )
            for face in faces
            if face.det_score >= self.threshold
        ]
    
# if __name__ == "__main__":
#     faceProcessing = FaceProcessing()
#     Console().print(len(faceProcessing.run(cv2.imread("image.jpg"))), style="bold red")
