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
        # ── Wait for a new message from handle_message ─────────────────────
        with clientMessageLocks.read() as messages:
            userLock = messages.get(userId)
            if userLock is None:
                return

        try:
            with userLock.wait_for_update(lastVersion) as currentVersion:
                lastVersion = currentVersion
        except Exception:
            return  # lock removed or interrupted; exit cleanly

        # ── Cancel existing timeout ────────────────────────────────────────
        with clientTimerLocks.read() as userTimers:
            timer = userTimers.get(userId)
            if timer:
                timer.interrupt()

        # ── Read state and run state machine ──────────────────────────────
        with clientMessageLocks.read() as messages:
            userLock = messages.get(userId)
            if userLock is None:
                return
            with userLock.read() as currentState:
                newState, _ = conversation_state_machine(currentState)

        # ── Write new state back ───────────────────────────────────────────
        with clientMessageLocks.write() as messages:
            if userId in messages:
                messages[userId].set_silent(newState)

        # ── (Re)start timeout timer ────────────────────────────────────────
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

    newUser = False

    with clientMessageLocks.write() as messages:
        clientStateLock = messages.get(user.id)

        if clientStateLock is None:
            messages[user.id] = ReadWriteLock(ConversationState(user.id, msg, loop))
            newUser = True
            clientStateLock = messages.get(user.id)
        # ── Existing conversation: update state ────────────────────────
        err      = None
        newState = None

        with clientStateLock.write() as state:
            state.message=msg
            
            if text.strip().lower() == "exit":
                newState = state.reset()
            elif image is not None:
                newState = state.with_image(image)
            elif state.needs_command:
                newState, err = state.with_command(text)
            elif state.needs_metadata:
                newState, err = state.with_metadata(text)
            else:
                # Nothing expected right now — tell the user
                asyncio.run_coroutine_threadsafe(
                    msg.reply_text("I'm not expecting input right now."), loop
                )
                if not newUser: return

        if err:
            asyncio.run_coroutine_threadsafe(msg.reply_text(err), loop)
        elif newState is not None:
            clientStateLock.value = newState

    if newUser: 
        loop.run_in_executor(gpu_executor, handle_payload, user.id, 0)


# ── Entry point ───────────────────────────────────────────────────────────────

            


if __name__ == "__main__":
    print("⛩️  Loading BotGateway...")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.ALL, handle_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES)