from __future__ import annotations
from dataclasses import dataclass
import asyncio
import numpy as np
from telegram import Message

from faceProcessing import FaceProcessing
from weaviate.classes.query import MetadataQuery
from weaviateClientManager import WeaviateClientManager
from weaviate.classes.config import Configure, Property, DataType, VectorDistances


# ── Prompts ───────────────────────────────────────────────────────────────────

_PROMPT_IMAGE    = "Please upload an image to be processed."
_PROMPT_COMMAND  = (
    "What would you like to do with this image?\n"
    "\t add:    add face to database\n"
    "\t search: identify person from current database\n"
    "\t exit:   cancel"
)
_PROMPT_METADATA = (
    "Please provide the following metadata:\n"
    "\t name:        <person's name>\n"
    "\t affiliation: <description of person's affiliation to CSG>\n"
    "\t status:      [Approved | Unapproved | Unknown]"
)
_ERR_COMMAND = "Unknown command."
_ERR_METADATA = "Invalid metadata. Please use the given format."

# ── State ─────────────────────────────────────────────────────────────────────

_VALID_COMMANDS = {"add", "search", "exit"}
_VALID_STATUSES = {"approved", "unapproved", "unknown"}


@dataclass
class ConversationState:
    userId: str
    message: Message
    loop: asyncio.AbstractEventLoop

    image: np.ndarray | None = None   # Telegram PhotoSize list
    command: str | None = None
    metadata: dict | None = None

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def needs_image(self) -> bool:
        return self.image is None

    @property
    def needs_command(self) -> bool:
        return self.image is not None and self.command is None

    @property
    def needs_metadata(self) -> bool:
        return self.command == "add" and not self._metadata_valid()

    @property
    def is_complete(self) -> bool:
        if self.command == "add":
            return self._metadata_valid()
        return self.command in {"search", "exit"}

    def _metadata_valid(self) -> bool:
        required = {"name", "affiliation", "status"}
        return (
            self.metadata is not None
            and required.issubset(self.metadata)
            and self.metadata["status"].lower() in _VALID_STATUSES
        )

    # ── Transitions ───────────────────────────────────────────────────────────

    def _copy(self, **overrides) -> ConversationState:
        """Return a new state with the same base fields, applying any overrides."""
        return ConversationState(
            userId=self.userId,
            message=self.message,
            loop=self.loop,
            image=overrides.get("image", self.image),
            command=overrides.get("command", self.command),
            metadata=overrides.get("metadata", self.metadata),
        )

    def with_message(self, message: Message) -> ConversationState:
        return self._copy(message=message)  # not in overrides pattern, handle manually
        # ^ handled below to keep _copy simple

    def with_image(self, image: object) -> ConversationState:
        return self._copy(image=image)

    def with_command(self, text: str) -> tuple[ConversationState, str | None]:
        cmd = text.strip().lower()
        if cmd not in _VALID_COMMANDS:
            return self, _ERR_COMMAND
        return self._copy(command=cmd), None

    def with_metadata(self, text: str) -> tuple[ConversationState, str | None]:
        fields = {}
        for line in text.strip().splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip().lower()] = value.strip()
        required = {"name", "affiliation", "status"}
        if not required.issubset(fields) or fields["status"].lower() not in _VALID_STATUSES:
            return self, _ERR_METADATA
        return self._copy(metadata=fields), None

    def reset(self) -> ConversationState:
        """Return a blank state, preserving only identity/loop fields."""
        return ConversationState(userId=self.userId, message=self.message, loop=self.loop)

# ── Business Logic ─────────────────────────────────────────────────────────────

weaviateClientManager = WeaviateClientManager()

CLASS_NAME = "Faces"
def upload_face(image: np.ndarray, name: str, affiliation: str, status: str):
    results = FaceProcessing().run(image)
    if not results: return
    with weaviateClientManager._call() as client:
        if not client.collections.exists(CLASS_NAME):
            client.collections.create(
                name=CLASS_NAME,
                properties=[
                    Property(name="name", data_type=DataType.TEXT),
                    Property(name="affiliation", data_type=DataType.TEXT),
                    Property(name="status", data_type=DataType.TEXT)
                ],
                vector_config=Configure.Vectors.none(
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE
                    )
                )
            )
        collection = client.collections.get(CLASS_NAME)
        for result in results:
            collection.data.insert(
                properties={"name": str(name), "affiliation": str(affiliation), "status": str(status)},
                vector=result.embedding.tolist()
            )


def search_face(image: object) -> str:
    results = FaceProcessing().run(image)
    if not results:
        return "No face detected in image."
    
    queryVector = np.mean([r.embedding for r in results], axis=0)
    
    with weaviateClientManager._call() as client:
        collection = client.collections.get(CLASS_NAME)
        response = collection.query.near_vector(
            near_vector=queryVector.tolist(),
            limit=1,
            return_metadata=MetadataQuery(distance=True, certainty=True)  
        )
        
        if not response.objects:
            return "No match found in database."
        
        obj        = response.objects[0]
        properties = obj.properties
        name        = properties.get("name")
        affiliation = properties.get("affiliation")
        status      = properties.get("status")
        
        distance  = obj.metadata.distance   # 0.0 = identical, 1.0 = completely different
        certainty = obj.metadata.certainty  # 1.0 = identical, 0.0 = completely different
        confidence_pct = round((1 - distance) * 100, 1)
        
        return (
            f"Possible Identification:\n"
            f"\t{name}: {status} ({affiliation})\n"
            f"\tConfidence: {confidence_pct}% (certainty: {certainty:.3f})"
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

    if state.needs_command:
        reply(_PROMPT_COMMAND)
        return state, False

    if state.needs_metadata:
        reply(_PROMPT_METADATA)
        return state, False

    if state.command == "exit":
        reply("Cancelled. Send an image any time to start again.")
        return state.reset(), False

    if state.command == "search":
        results = search_face(state.image)
        reply(results)
        return state.reset(), False

    if state.command == "add":
        upload_face(
            state.image,
            state.metadata["name"],
            state.metadata["affiliation"],
            state.metadata["status"],
        )
        reply("✅ Person uploaded to database.")
        return state.reset(), False

    # Unreachable given guards above
    reply(_ERR_COMMAND)
    return state, False