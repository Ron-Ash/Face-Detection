from __future__ import annotations
from dataclasses import dataclass
import asyncio
import numpy as np
from telegram import Message
from weaviate import WeaviateClient
from minio import Minio

from faceProcessing import FaceProcessing
from weaviate.classes.query import MetadataQuery, QueryReference
import weaviate_store
import minio_store

# ── Prompts ───────────────────────────────────────────────────────────────────
_PROMPT_IMAGE    = "Please upload an image to be processed."
_PROMPT_METADATA = (
    "Please provide the following metadata:\n"
    "\t name:        <person's name>\n"
    "\t affiliation: <description of person's affiliation>\n"
    "\t status:      [Approved | Unapproved | Unknown]"
)
_ERR_METADATA = "Invalid metadata. Please use the given format."
_ERR_CONFIRM  = "Please reply 'yes' or 'no'."

_VALID_STATUSES = {"approved", "unapproved", "unknown"}

NO_MATCH_DISTANCE = 0.40


# ── State ─────────────────────────────────────────────────────────────────────
@dataclass
class ConversationState:
    userId:  str
    message: Message
    loop:    asyncio.AbstractEventLoop

    image:    np.ndarray | None = None
    # Stages (in order):
    #   None → "searching" → "confirm_match" | "confirm_new"
    #        → "awaiting_match_confirm" | "awaiting_new_confirm"
    #        → "add_to_existing" | "needs_metadata"
    #        → "awaiting_metadata" → "add_new_person" → reset
    stage:    str | None = None
    metadata: dict | None = None

    match_uuid:       str | None   = None
    match_name:       str | None   = None
    match_affiliation:str | None   = None
    match_status:     str | None   = None
    match_confidence: float | None = None
    match_embedding:  list | None  = None

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def needs_image(self) -> bool:
        return self.image is None

    def _metadata_valid(self) -> bool:
        required = {"name", "affiliation", "status"}
        return (
            self.metadata is not None
            and required.issubset(self.metadata)
            and self.metadata["status"].lower() in _VALID_STATUSES
        )

    # ── Transitions ───────────────────────────────────────────────────────────
    def _copy(self, **overrides) -> ConversationState:
        return ConversationState(
            userId=self.userId,
            message=self.message,
            loop=self.loop,
            image=overrides.get("image", self.image),
            stage=overrides.get("stage", self.stage),
            metadata=overrides.get("metadata", self.metadata),
            match_uuid=overrides.get("match_uuid", self.match_uuid),
            match_name=overrides.get("match_name", self.match_name),
            match_affiliation=overrides.get("match_affiliation", self.match_affiliation),
            match_status=overrides.get("match_status", self.match_status),
            match_confidence=overrides.get("match_confidence", self.match_confidence),
            match_embedding=overrides.get("match_embedding", self.match_embedding),
        )

    def with_image(self, image: np.ndarray) -> ConversationState:
        return self._copy(image=image, stage="searching")

    def with_confirmation(self, text: str) -> tuple[ConversationState, str | None]:
        answer = text.strip().lower()
        if answer not in {"yes", "no"}:
            return self, _ERR_CONFIRM

        if self.stage == "awaiting_match_confirm":
            if answer == "yes":
                return self._copy(stage="add_to_existing"), None
            else:
                # Rejected the match — ask whether to add as new person instead
                return self._copy(stage="confirm_new"), None

        if self.stage == "awaiting_new_confirm":
            if answer == "yes":
                return self._copy(stage="needs_metadata"), None
            else:
                return self.reset(), None

        return self, _ERR_CONFIRM

    def with_metadata(self, text: str) -> tuple[ConversationState, str | None]:
        fields: dict[str, str] = {}
        for line in text.strip().splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip().lower()] = value.strip()
        required = {"name", "affiliation", "status"}
        if not required.issubset(fields) or fields["status"].lower() not in _VALID_STATUSES:
            return self, _ERR_METADATA
        return self._copy(metadata=fields, stage="add_new_person"), None

    def reset(self) -> ConversationState:
        return ConversationState(userId=self.userId, message=self.message, loop=self.loop)


# ── Search ────────────────────────────────────────────────────────────────────

def search_and_stage(
    state: ConversationState,
    wv_client: WeaviateClient,
    fp: FaceProcessing,
) -> ConversationState:
    """
    Run face detection on state.image, query Weaviate via weaviate_store,
    and return a new state staged at "confirm_match" or "confirm_new".
    MinIO is not needed here — image storage only happens on confirmation.
    """
    results = fp.run(state.image)
    if not results:
        return state._copy(stage="no_face")

    query_vector = np.mean([r.embedding for r in results], axis=0).tolist()

    person_uuid, name, affiliation, status, confidence = weaviate_store.query_nearest_person(
        wv_client,
        np.array(query_vector),
        distance_threshold=NO_MATCH_DISTANCE,
    )

    if person_uuid is None:
        # No confident match — ask whether to register as new
        return state._copy(stage="confirm_new", match_embedding=query_vector)

    return state._copy(
        stage="confirm_match",
        match_uuid=person_uuid,
        match_name=name,
        match_affiliation=affiliation,
        match_status=status,
        match_confidence=confidence,
        match_embedding=query_vector,
    )


# ── State machine ─────────────────────────────────────────────────────────────

def conversation_state_machine(
    state: ConversationState,
    wv_client: WeaviateClient,
    mn_client: Minio,
    fp: FaceProcessing,
) -> tuple[ConversationState, bool]:
    """
    Drive the conversation one step forward.
    Returns (new_state, done).

    Both wv_client and mn_client are caller-owned; this function never
    opens or closes either connection.
    """

    def reply(text: str) -> None:
        asyncio.run_coroutine_threadsafe(
            state.message.reply_text(text), state.loop
        )

    def _pil_from_state() -> "Image.Image | None":
        """Convert the numpy BGR frame stored in state to a PIL RGB image."""
        if state.image is None:
            return None
        from PIL import Image
        import cv2
        rgb = cv2.cvtColor(state.image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    if state.needs_image:
        reply(_PROMPT_IMAGE)
        return state, False

    # ── searching ─────────────────────────────────────────────────────────────
    if state.stage == "searching":
        new_state = search_and_stage(state, wv_client, fp)
        # Recurse once so the match/no-match prompt is sent in the same call
        return conversation_state_machine(new_state, wv_client, mn_client, fp)

    # ── no face detected ──────────────────────────────────────────────────────
    if state.stage == "no_face":
        reply("No face detected in that image. Please try another.")
        return state.reset(), False

    # ── present match to user ─────────────────────────────────────────────────
    if state.stage == "confirm_match":
        reply(
            f"I found a possible match:\n"
            f"  {state.match_name}  |  {state.match_status}  ({state.match_affiliation})\n"
            f"  Confidence: {state.match_confidence}%\n\n"
            f"Is this the same person? (yes / no)"
        )
        return state._copy(stage="awaiting_match_confirm"), False

    # ── ask whether to add as new person ─────────────────────────────────────
    if state.stage == "confirm_new":
        reply("No confident match found. Would you like to add this as a new person? (yes / no)")
        return state._copy(stage="awaiting_new_confirm"), False

    # ── waiting for user input — nothing to do yet ────────────────────────────
    if state.stage in ("awaiting_match_confirm", "awaiting_new_confirm", "awaiting_metadata"):
        return state, False

    # ── confirmed existing match — upload image and link new embedding ────────
    if state.stage == "add_to_existing":
        pil_img = _pil_from_state()
        if pil_img is not None:
            object_key = minio_store.upload_image(mn_client, pil_img, state.match_uuid)
            weaviate_store.add_face_embedding(
                wv_client, state.match_uuid, np.array(state.match_embedding), object_key
            )
        reply(f"✅ New image added to {state.match_name}'s record.")
        return state.reset(), False

    # ── ask for metadata before creating new person ───────────────────────────
    if state.stage == "needs_metadata":
        reply(_PROMPT_METADATA)
        return state._copy(stage="awaiting_metadata"), False

    # ── metadata received — create person, upload image, store embedding ──────
    if state.stage == "add_new_person":
        name        = state.metadata["name"]
        affiliation = state.metadata["affiliation"]
        status      = state.metadata["status"]

        # 1. Create Person in Weaviate
        person_uuid = weaviate_store.create_person(wv_client, name, affiliation, status)

        # 2. Upload image to MinIO
        pil_img    = _pil_from_state()
        img_to_store = pil_img if pil_img is not None else __import__("PIL.Image", fromlist=["Image"]).Image.new("RGB", (64, 64))
        object_key = minio_store.upload_image(mn_client, img_to_store, person_uuid)

        # 3. Store embedding + MinIO key in Weaviate
        weaviate_store.add_face_embedding(
            wv_client, person_uuid, np.array(state.match_embedding), object_key
        )

        reply(f"✅ {name} added to the database.")
        return state.reset(), False

    # ── fallback ──────────────────────────────────────────────────────────────
    reply("Something went wrong. Please start again by sending an image.")
    return state.reset(), False