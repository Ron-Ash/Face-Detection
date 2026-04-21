from __future__ import annotations
from dataclasses import dataclass
import asyncio
import numpy as np
from telegram import Message

from faceProcessing import FaceProcessing
from weaviateClientManager import WeaviateClientManager
from weaviate.classes.config import Configure, Property, DataType, ReferenceProperty, VectorDistances
from weaviate.classes.query import MetadataQuery, QueryReference
import uuid

# ── Prompts ───────────────────────────────────────────────────────────────────
_PROMPT_IMAGE = "Please upload an image to be processed."
_PROMPT_METADATA = (
    "Please provide the following metadata:\n"
    "\t name:        <person's name>\n"
    "\t affiliation: <description of person's affiliation>\n"
    "\t status:      [Approved | Unapproved | Unknown]"
)
_ERR_METADATA = "Invalid metadata. Please use the given format."
_ERR_CONFIRM = "Please reply 'yes' or 'no'."

_VALID_STATUSES = {"approved", "unapproved", "unknown"}

# ── State ─────────────────────────────────────────────────────────────────────
@dataclass
class ConversationState:
    userId: str
    message: Message
    loop: asyncio.AbstractEventLoop

    image: np.ndarray | None = None
    # Flow control
    # Stages: None → "searching" → "confirm_match" → "confirm_new" → "needs_metadata" → done
    stage: str | None = None
    metadata: dict | None = None

    # Populated after search
    match_uuid: str | None = None   # UUID of best Weaviate Person match
    match_name: str | None = None
    match_affiliation:str | None = None
    match_status: str | None = None
    match_confidence: float | None = None
    match_embedding: list | None  = None   # the averaged query vector

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def needs_image(self) -> bool:
        return self.image is None

    @property
    def needs_metadata(self) -> bool:
        return self.stage == "needs_metadata" and not self._metadata_valid()

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

        if self.stage == "awaiting_match_confirm":   # ← was "confirm_match"
            if answer == "yes":
                return self._copy(stage="add_to_existing"), None
            else:
                return self._copy(stage="confirm_new"), None

        if self.stage == "awaiting_new_confirm":     # ← was "confirm_new"
            if answer == "yes": return self._copy(stage="needs_metadata"), None
            else:
                return self.reset(), None

        return self, _ERR_CONFIRM

    def with_metadata(self, text: str) -> tuple[ConversationState, str | None]:
        fields = {}
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


# ── Weaviate ──────────────────────────────────────────────────────────────────
weaviateClientManager = WeaviateClientManager()
NO_MATCH_DISTANCE = 0.05   # distances above this are treated as no confident match

def setup_collections(client) -> None:
    if not client.collections.exists("Person"):
        client.collections.create(
            name="Person",
            properties=[
                Property(name="name",        data_type=DataType.TEXT),
                Property(name="affiliation", data_type=DataType.TEXT),
                Property(name="status",      data_type=DataType.TEXT),
            ]
        )
    if not client.collections.exists("FaceEmbedding"):
        client.collections.create(
            name="FaceEmbedding",
            properties=[Property(name="source_image", data_type=DataType.TEXT)],
            references=[ReferenceProperty(name="person", target_collection="Person")],
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            )
        )

def _create_person(client, name: str, affiliation: str, status: str) -> str:
    person_uuid = str(uuid.uuid4())
    client.collections.get("Person").data.insert(
        properties={"name": name, "affiliation": affiliation, "status": status},
        uuid=person_uuid
    )
    return person_uuid

def _add_embedding(client, embedding: list, person_uuid: str) -> None:
    client.collections.get("FaceEmbedding").data.insert(
        properties={"source_image": ""},
        vector=embedding,
        references={"person": person_uuid}
    )

def search_and_stage(state: ConversationState) -> ConversationState:
    """
    Run face detection + Weaviate search on state.image.
    Returns a new state staged at 'confirm_match' or 'confirm_new'
    depending on whether a confident match was found.
    """
    results = FaceProcessing().run(state.image)
    if not results:
        return state._copy(stage="no_face")

    query_vector = np.mean([r.embedding for r in results], axis=0).tolist()

    with weaviateClientManager._call() as client:
        setup_collections(client)

        embeddings = client.collections.get("FaceEmbedding")
        response = embeddings.query.near_vector(
            near_vector=query_vector,
            limit=1,
            return_metadata=MetadataQuery(distance=True),
            return_references=QueryReference(
                link_on="person",
                return_properties=["name", "affiliation", "status"]
            )
        )

    # No records in DB yet or no match
    if not response.objects:
        return state._copy(
            stage="confirm_new",
            match_embedding=query_vector
        )

    obj  = response.objects[0]
    dist = obj.metadata.distance

    if dist > NO_MATCH_DISTANCE:
        return state._copy(
            stage="confirm_new",
            match_embedding=query_vector
        )

    # Confident match found — extract person details
    person_refs = obj.references.get("person")
    if not person_refs or not person_refs.objects:
        return state._copy(stage="confirm_new", match_embedding=query_vector)

    props = person_refs.objects[0].properties
    person_uuid = str(person_refs.objects[0].uuid)

    return state._copy(
        stage="confirm_match",
        match_uuid=person_uuid,
        match_name=props.get("name"),
        match_affiliation=props.get("affiliation"),
        match_status=props.get("status"),
        match_confidence=round((1 - dist) * 100, 1),
        match_embedding=query_vector,
    )


# ── State machine ─────────────────────────────────────────────────────────────

def conversation_state_machine(state: ConversationState) -> tuple[ConversationState, bool]:

    def reply(text: str) -> None:
        asyncio.run_coroutine_threadsafe(
            state.message.reply_text(text), state.loop
        )

    if state.needs_image:
        reply(_PROMPT_IMAGE)
        return state, False

    if state.stage == "searching":
        new_state = search_and_stage(state)
        return conversation_state_machine(new_state)  # ← key fix

    if state.stage == "no_face":
        reply("No face detected in that image. Please try another.")
        return state.reset(), False

    if state.stage == "confirm_match":
        reply(
            f"I found a possible match:\n"
            f"\t{state.match_name} | {state.match_status} ({state.match_affiliation})\n"
            f"\tConfidence: {state.match_confidence}%\n\n"
            f"Is this the same person? (yes / no)"
        )
        return state._copy(stage="awaiting_match_confirm"), False

    if state.stage == "confirm_new":
        reply("No confident match found. Would you like to add this as a new person? (yes / no)")
        return state._copy(stage="awaiting_new_confirm"), False

    if state.stage in ("awaiting_match_confirm", "awaiting_new_confirm", "awaiting_metadata"):
        return state, False

    if state.stage == "add_to_existing":
        with weaviateClientManager._call() as client:
            setup_collections(client)
            _add_embedding(client, state.match_embedding, state.match_uuid)
        reply(f"✅ New image added to {state.match_name}'s record.")
        return state.reset(), False

    if state.stage == "needs_metadata":
        reply(_PROMPT_METADATA)
        return state._copy(stage="awaiting_metadata"), False

    if state.stage == "add_new_person":
        with weaviateClientManager._call() as client:
            setup_collections(client)
            person_uuid = _create_person(
                client,
                state.metadata["name"],
                state.metadata["affiliation"],
                state.metadata["status"],
            )
            _add_embedding(client, state.match_embedding, person_uuid)
        reply(f"✅ {state.metadata['name']} added to the database.")
        return state.reset(), False

    reply("Something went wrong. Please start again by sending an image.")
    return state.reset(), False