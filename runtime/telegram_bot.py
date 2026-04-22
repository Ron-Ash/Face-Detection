from functools import partial
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import weaviate

from concurrency.interruptTimer import InterruptibleTimer
from telegram import Message, Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

from concurrency.readWriteLock import ReadWriteLock
from conversationStateMachine import ConversationState, conversation_state_machine
from database.setup import setup_all
from faceProcessing import FaceProcessing
import os

from database.minio_store import create_client as minio_create_client

# ── Config ────────────────────────────────────────────────────────────────────

def _read_first_line(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.readline().strip()

def _read_users(path: str) -> set[int]:
    with open(path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f}

BOT_TOKEN        = os.getenv("TELEGRAM_API", _read_first_line("telegram_token.txt"))
ALLOWED_USER_IDS = _read_users("telegram_users.txt")
REPLY_TO_BLOCKED_USERS = True

# One GPU worker at a time; one FaceProcessing instance shared across all work
gpu_executor     = ThreadPoolExecutor(max_workers=1)
_face_processing = FaceProcessing()

clientMessageLocks: ReadWriteLock[dict[int, ReadWriteLock[ConversationState]]] = ReadWriteLock(dict())
clientTimerLocks:   ReadWriteLock[dict[int, InterruptibleTimer]]               = ReadWriteLock(dict())

# Per-user stop events so close_conversation can signal the worker to exit cleanly
clientStopEvents: ReadWriteLock[dict[int, asyncio.Event]] = ReadWriteLock(dict())


# ── Timeout ───────────────────────────────────────────────────────────────────

def close_conversation(userId: int) -> None:
    """
    Called by the inactivity timer on the timer thread.
    Removes all per-user state and sets the stop event so the worker exits
    and closes its own DB clients in its finally block.
    """
    with clientMessageLocks.write() as locks:
        locks.pop(userId, None)
    with clientTimerLocks.write() as userTimers:
        userTimers.pop(userId, None)
    with clientStopEvents.write() as stops:
        event = stops.pop(userId, None)
        if event:
            event.set()
    print(f"[TIMEOUT] Conversation closed for user {userId}")


# ── GPU worker ────────────────────────────────────────────────────────────────

def handle_payload(userId: int, lastVersion: int, stop_event: asyncio.Event) -> None:
    """
    Long-running worker for a single user.
    Opens one Weaviate and one MinIO connection for its lifetime and closes
    both on exit regardless of how the loop terminates.
    """
    wv_client = weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051)
    mn_client = minio_create_client()   # localhost:9000 by default

    try:
        # Idempotent: creates collections + bucket if they don't exist yet
        setup_all(wv_client)

        while not stop_event.is_set():
            # ── Wait for a state change ───────────────────────────────────────
            with clientMessageLocks.read() as messages:
                userLock = messages.get(userId)
            if userLock is None:
                break   # conversation was closed externally

            try:
                with userLock.wait_for_update(lastVersion) as currentVersion:
                    lastVersion = currentVersion
            except Exception:
                break   # lock destroyed or interrupted

            if stop_event.is_set():
                break

            # ── Reset inactivity timer ────────────────────────────────────────
            with clientTimerLocks.write() as userTimers:
                timer = userTimers.get(userId)
                if timer:
                    timer.interrupt()

            # ── Read current state and advance the machine ────────────────────
            with clientMessageLocks.read() as messages:
                userLock = messages.get(userId)
            if userLock is None:
                break

            with userLock.read() as currentState:
                newState, _ = conversation_state_machine(
                    currentState, wv_client, mn_client, _face_processing
                )

            # ── Write new state back ──────────────────────────────────────────
            with clientMessageLocks.write() as messages:
                if userId not in messages:
                    break   # conversation was closed while we were processing
                messages[userId].set_silent(newState)

            # ── (Re-)arm inactivity timer ─────────────────────────────────────
            with clientTimerLocks.write() as userTimers:
                timer = userTimers.get(userId)
                if timer:
                    timer.start()
                else:
                    userTimers[userId] = InterruptibleTimer(
                        5 * 60, partial(close_conversation, userId)
                    )
                    userTimers[userId].start()

    finally:
        wv_client.close()
        # MinIO client has no explicit close — connections are pooled internally
        print(f"[Worker] DB clients closed for user {userId}")


# ── Telegram handler ──────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg  = update.effective_message
    user = update.effective_user

    if not user or user.id not in ALLOWED_USER_IDS:
        if REPLY_TO_BLOCKED_USERS and msg:
            await msg.reply_text("Sorry — this bot is private.")
        return

    text  = (msg.text or msg.caption or "") if msg else ""
    image = None

    if msg and msg.photo:
        photo       = msg.photo[-1]
        file        = await photo.get_file()
        image_bytes = await file.download_as_bytearray()
        np_arr      = np.frombuffer(image_bytes, np.uint8)
        image       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    loop = asyncio.get_running_loop()

    spawnWorker  = False
    signalWorker = False

    with clientMessageLocks.write() as messages:
        clientStateLock = messages.get(user.id)

        if clientStateLock is None:
            # ── New conversation ──────────────────────────────────────────────
            initialState = ConversationState(user.id, msg, loop)
            if image is not None:
                initialState = initialState.with_image(image)
            messages[user.id] = ReadWriteLock(initialState)
            clientStateLock    = messages[user.id]

            stop_event = asyncio.Event()
            with clientStopEvents.write() as stops:
                stops[user.id] = stop_event

            spawnWorker  = True
            signalWorker = True

        else:
            # ── Existing conversation — apply transition ──────────────────────
            err      = None
            newState = None

            with clientStateLock.write() as state:
                state.message = msg   # always update so replies go to the latest message

                if text.strip().lower() == "exit":
                    newState = state.reset()
                elif image is not None:
                    newState = state.with_image(image)
                elif state.stage in ("awaiting_match_confirm", "awaiting_new_confirm"):
                    newState, err = state.with_confirmation(text)
                elif state.stage == "awaiting_metadata":
                    newState, err = state.with_metadata(text)
                else:
                    asyncio.run_coroutine_threadsafe(
                        msg.reply_text("I'm not expecting input right now."), loop
                    )

            if err:
                asyncio.run_coroutine_threadsafe(msg.reply_text(err), loop)
            elif newState is not None:
                clientStateLock.value = newState   # bumps version, wakes worker
                signalWorker = True

    # ── Spawn worker outside the lock ─────────────────────────────────────────
    if spawnWorker:
        with clientStopEvents.read() as stops:
            stop_event = stops[user.id]
        loop.run_in_executor(
            gpu_executor, handle_payload, user.id, 0, stop_event
        )


if __name__ == "__main__":
    print("⛩️  Loading BotGateway...")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.ALL, handle_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES)