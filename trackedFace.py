from typing import Optional
from dataclasses import dataclass

import numpy as np

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