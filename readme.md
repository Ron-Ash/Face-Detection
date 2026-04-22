# Face Detection & Recognition System

A real-time face recognition pipeline for a **live webcam monitor**. Faces are embedded with [InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l`) and stored in a [Weaviate](https://weaviate.io/) vector embedding database, and [MinIO](https://www.min.io/) for binary image storage.

---

## Project Structure

```
├── main.py                     # Real-time webcam monitoring entry point
├── facialRecognition/
│   ├── faceProcessing.py       # Face detection + embedding extraction
│   ├── faceTracker.py          # Frame tracking + identity association
│   └── trackedFace.py          # TrackedFace data structure
├── database/
│   ├── setup.py                # Idempotent Weaviate + MinIO setup
│   ├── weaviate_store.py       # Vector search + metadata operations
│   └── minio_store.py          # Image storage operations
├── concurrency/
│   ├── readWriteLock.py        # Versioned read/write synchronization
│   └── interruptTimer.py       # Interruptible timers for async flows
```