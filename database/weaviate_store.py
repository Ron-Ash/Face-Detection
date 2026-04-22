from __future__ import annotations

from typing import Optional
from uuid import UUID

import numpy as np
import weaviate
from weaviate.classes.query import MetadataQuery, QueryReference


def create_person(client: weaviate.WeaviateClient, name: str, affiliation: str, status: str) -> str:
    uuid = client.collections.get("Person").data.insert(
        properties={"name": name, "affiliation": affiliation, "status": status}
    )
    return str(uuid)

def get_person(client: weaviate.WeaviateClient, person_uuid: str) -> Optional[dict]:
    try:
        obj = client.collections.get("Person").query.fetch_object_by_id(
            uuid=person_uuid,
            return_properties=["name", "affiliation", "status"],
        )
        if obj is None:
            return None
        return dict(obj.properties)
    except Exception as e:
        print(f"[weaviate_store] get_person error: {e}")
        return None

def update_person(client: weaviate.WeaviateClient, person_uuid: str, name: Optional[str] = None, affiliation: Optional[str] = None, status: Optional[str] = None) -> bool:
    try:
        props = {}
        if name is not None: props["name"] = name
        if affiliation is not None: props["affiliation"] = affiliation
        if status is not None: props["status"] = status
        client.collections.get("Person").data.update(uuid=person_uuid, properties=props)
        return True
    except Exception as e:
        print(f"[weaviate_store] update_person error: {e}")
        return False

def delete_person(client: weaviate.WeaviateClient, person_uuid: str) -> bool:
    try:
        client.collections.get("Person").data.delete_by_id(uuid=person_uuid)
        return True
    except Exception as e:
        print(f"[weaviate_store] delete_person error: {e}")
        return False




def add_face_embedding(client: weaviate.WeaviateClient, person_uuid: str, embedding: np.ndarray, minio_object_key: str) -> str:
    uuid = client.collections.get("FaceEmbedding").data.insert(
        properties={"source_image": minio_object_key},
        vector=embedding.tolist(),
        references={"person": person_uuid},
    )
    return str(uuid)

def delete_face_embedding(client: weaviate.WeaviateClient, embedding_uuid: str) -> bool:
    try:
        client.collections.get("FaceEmbedding").data.delete_by_id(uuid=embedding_uuid)
        return True
    except Exception as e:
        print(f"[weaviate_store] delete_face_embedding error: {e}")
        return False




def query_nearest_person(client: weaviate.WeaviateClient, embedding: np.ndarray, distance_threshold: float = 0.4) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[float]]:
    try:
        response = client.collections.get("FaceEmbedding").query.near_vector(
            near_vector=embedding.tolist(),
            limit=1,
            return_metadata=MetadataQuery(distance=True),
            return_references=QueryReference(
                link_on="person",
                return_properties=["name", "affiliation", "status"],
            ),
        )
        if not response.objects:
            return None, None, None, None, None

        obj = response.objects[0]
        dist = obj.metadata.distance
        if dist > distance_threshold:
            return None, None, None, None, None

        refs = obj.references.get("person")
        if not refs or not refs.objects:
            return None, None, None, None, None

        person = refs.objects[0]
        props = person.properties
        confidence = round((1 - dist) * 100, 1)
        return (
            str(person.uuid),
            props.get("name"),
            props.get("affiliation"),
            props.get("status"),
            confidence,
        )
    except Exception as e:
        print(f"[weaviate_store] query_nearest_person error: {e}")
        return None, None, None, None, None

def query_embeddings_for_person(client: weaviate.WeaviateClient, embedding: np.ndarray, limit: int = 8, distance_threshold: float = 0.45) -> list[tuple[str, float]]:
    try:
        response = client.collections.get("FaceEmbedding").query.near_vector(
            near_vector=embedding.tolist(),
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
            return_properties=["source_image"],
        )
        results = []
        for obj in response.objects:
            dist = obj.metadata.distance
            if dist > distance_threshold:
                continue
            key = obj.properties.get("source_image", "")
            if key:
                results.append((key, dist))
        results.sort(key=lambda t: t[1])
        return results
    except Exception as e:
        print(f"[weaviate_store] query_embeddings_for_person error: {e}")
        return []

def get_person_uuid_for_embedding(client: weaviate.WeaviateClient, embedding: np.ndarray, distance_threshold: float = 0.4) -> Optional[str]:
    person_uuid, *_ = query_nearest_person(client, embedding, distance_threshold)
    return person_uuid