# Face Detection & Recognition System

A real-time face recognition pipeline combining a **Telegram enrollment bot** with a **live webcam monitor**. Faces are embedded with [InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l`) and stored in a [Weaviate](https://weaviate.io/) vector database, which is lazily started on demand via Docker Compose.

---

## Features

- **Telegram enrollment bot** — upload a photo, confirm or create a person record, and attach metadata (name, affiliation, status).
- **Realtime webcam monitor** — detects, tracks, and identifies faces live, drawing status-coloured boxes (green = approved, red = unapproved, grey = unknown).
- **Lazy Weaviate lifecycle** — the database container auto-starts on first use and shuts itself down after 5 minutes of inactivity to save resources.
- **Thread-safe concurrency primitives** — custom `ReadWriteLock` with versioning and `InterruptibleTimer` for coordinated state across async + threaded workloads.
- **Conversation state machine** — immutable, copy-on-transition state objects drive the multi-step Telegram flow cleanly.

---

## Architecture

```
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│ Telegram users   │──────▶│ telegram_bot.py  │──────▶ │ conversation     │
│ (images + text)  │        │ (async handler)  │        │ StateMachine.py  │
└──────────────────┘        └──────────────────┘        └────────┬─────────┘
                                                                 │
┌──────────────────┐        ┌──────────────────┐                 ▼
│ Webcam feed      │──────▶│ realtime_        │        ┌──────────────────┐
│ (OpenCV)         │        │ monitoring.py   │──────▶ │ faceProcessing   │
└──────────────────┘        └──────────────────┘        │ (InsightFace)    │
                                                        └────────┬─────────┘
                                                                 │
                                                                 ▼
                                                ┌────────────────────────────┐
                                                │ weaviateClientManager.py   │
                                                │  (lazy docker-compose)     │
                                                └────────────────┬───────────┘
                                                                 │
                                                                 ▼
                                                        ┌────────────────┐
                                                        │ Weaviate DB    │
                                                        │ Person +       │
                                                        │ FaceEmbedding  │
                                                        └────────────────┘
```

Two Weaviate collections back the system:

- **`Person`** — holds `name`, `affiliation`, `status` (`approved` / `unapproved` / `unknown`).
- **`FaceEmbedding`** — stores 512-d face vectors with a cross-reference to the `Person` they belong to. Multiple embeddings can point at the same person, so new photos of the same individual enrich the match without duplicating the record.

---

## Repository layout

```
face-detection/
├── concurrency/
│   ├── dockerComposeService.py   # Thin wrapper over `docker compose up/down`
│   ├── interruptTimer.py         # Cancellable countdown timer
│   └── readWriteLock.py          # Versioned reader-writer lock
├── conversationStateMachine.py   # Immutable Telegram conversation state + transitions
├── faceProcessing.py             # InsightFace wrapper → normalized 512-d embeddings
├── realtime_monitoring.py        # Webcam face tracker + async Weaviate lookups
├── telegram_bot.py               # Telegram gateway; dispatches to state machine
├── weaviateClientManager.py      # Lazy-start Weaviate client with idle shutdown
├── docker-compose.yml            # Weaviate service definition
├── analysis.ipynb                # Exploration / scratch notebook
├── requirements.txt
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.10+
- Docker + Docker Compose
- A webcam (for `realtime_monitoring.py`)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)

### 1. Clone & install

```bash
git clone <repo-url> face-detection
cd face-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets

The bot reads credentials from either environment variables **or** plaintext files (first line used):

| Purpose                   | Env var                   | File fallback                                  |
| ------------------------- | ------------------------- | ---------------------------------------------- |
| Telegram bot token        | `TELEGRAM_API`            | `telegram_token.txt`                           |
| Allowed Telegram user IDs | —                         | `telegram_users.txt` (one numeric ID per line) |
| Weaviate host             | `DB_HOST`                 | defaults to `localhost`                        |
| Weaviate user / password  | `DB_USER` / `DB_PASSWORD` | defaults to `user` / `password`                |

> **Heads-up:** `telegram_token.txt` and `telegram_users.txt` should be in `.gitignore`. Double-check before pushing.

### 3. Start Weaviate (optional — it auto-starts on demand)

```bash
docker compose up -d
```

The `WeaviateClientManager` will start the container itself on first use and stop it after 5 minutes of idle time. Running it manually is only useful for debugging or for the first schema creation.

---

## Running

### Telegram enrollment bot

```bash
python telegram_bot.py
```

Conversation flow:

1. User sends a photo.
2. Bot runs face detection and queries Weaviate for the closest match.
3. If a match is found → _"Is this the same person?"_ → `yes` attaches the new embedding; `no` offers to add a new person.
4. If no match → _"Add as new person?"_ → collect `name / affiliation / status` metadata → insert.
5. Type `exit` at any time to reset the conversation.

### Realtime webcam monitor

```bash
python realtime_monitoring.py
```

Boxes are drawn around each detected face. Labels include name, affiliation, status, and match confidence. Press **Shift+Q** to quit.

---

## How the concurrency works

The bot handles any number of simultaneous users without blocking the event loop:

- Each user gets a `ReadWriteLock`-wrapped `ConversationState`.
- A single-worker `ThreadPoolExecutor` serialises GPU-bound face processing, so only one inference runs on the GPU at a time.
- The worker blocks on `wait_for_update(version)`, which wakes only when that specific user's state has mutated — no polling, no busy loops.
- An `InterruptibleTimer` per user evicts stale conversations after 5 minutes of inactivity.

On the realtime side, `FaceTracker` maintains cosine-similarity-matched tracks across frames, dispatching Weaviate lookups asynchronously so frame rendering never stalls on a database round-trip.

---

## Weaviate schema

Created automatically by `setup_collections()` on first use:

```python
Person         (name, affiliation, status)
FaceEmbedding  (source_image, vector=512d) --person--> Person
```

Cosine distance is used for HNSW indexing. The match threshold is **0.05** for the Telegram flow (strict) and **0.4** for the realtime tracker (lenient, since tracking already filters noise).

---

## Troubleshooting

- **`Failed to connect to Weaviate after retries`** — ensure Docker is running and port `8080`/`50051` aren't already in use.
- **`Sorry — this bot is private.`** — add your Telegram user ID to `telegram_users.txt`.
- **No face detected** — the `buffalo_l` detector needs a reasonably clear, front-facing face; the default confidence threshold is `0.6`.
- **GPU not used** — `FaceAnalysis` falls back to CPU if no CUDA-capable GPU is found. Install `onnxruntime-gpu` for acceleration.

---

## License

MIT (or whatever you prefer — update this section).
