from functools import partial
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from concurrency.interruptTimer import InterruptibleTimer
from telegram import Message, Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

from concurrency.readWriteLock import ReadWriteLock
from conversationStateMachine import ConversationState, conversation_state_machine
import os

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
gpu_executor = ThreadPoolExecutor(max_workers=1)

clientMessageLocks = ReadWriteLock(dict())
clientTimerLocks   = ReadWriteLock(dict())

# ── Timeout ───────────────────────────────────────────────────────────────────

def close_conversation(userId: int) -> None:
    with clientMessageLocks.write() as locks:
        locks.pop(userId, None)
    with clientTimerLocks.write() as userTimers:
        userTimers.pop(userId, None)
    print(f"[TIMEOUT] Conversation closed for user {userId}")

# ── GPU worker ────────────────────────────────────────────────────────────────

def handle_payload(userId: int, lastVersion: int) -> None:
    while True:
        with clientMessageLocks.read() as messages:
            userLock = messages.get(userId)
            if userLock is None:
                return

        try:
            with userLock.wait_for_update(lastVersion) as currentVersion:
                lastVersion = currentVersion
        except Exception:
            return

        with clientTimerLocks.read() as userTimers:
            timer = userTimers.get(userId)
            if timer:
                timer.interrupt()

        with clientMessageLocks.read() as messages:
            userLock = messages.get(userId)
            if userLock is None:
                return
            with userLock.read() as currentState:
                newState, _ = conversation_state_machine(currentState)

        with clientMessageLocks.write() as messages:
            if userId in messages:
                messages[userId].set_silent(newState)

        with clientTimerLocks.write() as userTimers:
            timer = userTimers.get(userId)
            if timer:
                timer.start()
            else:
                userTimers[userId] = InterruptibleTimer(
                    5 * 60, partial(close_conversation, userId)
                )

# ── Telegram handler ──────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg  = update.effective_message
    chat = update.effective_chat
    user = update.effective_user

    if not user or user.id not in ALLOWED_USER_IDS:
        if REPLY_TO_BLOCKED_USERS and msg:
            await msg.reply_text("Sorry — this bot is private.")
        return

    text  = (msg.text or msg.caption or "") if msg else ""
    image = msg.photo or None
    if image is not None:
        photo = image[-1]
        file = await photo.get_file()
        image_bytes = await file.download_as_bytearray()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    loop  = asyncio.get_running_loop()

    spawnWorker  = False
    signalWorker = False   # whether to bump version and wake existing worker

    with clientMessageLocks.write() as messages:
        clientStateLock = messages.get(user.id)

        if clientStateLock is None:
            # ── Brand new user — create state and spawn worker ─────────────
            initialState = ConversationState(user.id, msg, loop)
            if image is not None:
                initialState = initialState.with_image(image)
            messages[user.id] = ReadWriteLock(initialState)
            clientStateLock = messages[user.id]
            spawnWorker = True
            signalWorker = True

        else:
            # ── Existing conversation — compute next state ─────────────────
            err      = None
            newState = None

            with clientStateLock.write() as state:
                state.message = msg

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
        # For new users, manually bump version so wait_for_update(0) unblocks
        clientStateLock.value = clientStateLock.value  # re-set to bump version
        loop.run_in_executor(gpu_executor, handle_payload, user.id, 0)


if __name__ == "__main__":
    print("⛩️  Loading BotGateway...")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.ALL, handle_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES)