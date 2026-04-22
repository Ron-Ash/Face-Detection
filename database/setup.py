from __future__ import annotations

import weaviate
from weaviate.classes.config import Configure, DataType, Property, ReferenceProperty, VectorDistances

from .minio_store import (
    create_client as minio_create_client,
    ensure_bucket,
    BUCKET_NAME,
)

def setup_weaviate(client: weaviate.WeaviateClient) -> None:
    if not client.collections.exists("Person"):
        client.collections.create(
            name="Person",
            properties=[
                Property(name="name",        data_type=DataType.TEXT),
                Property(name="affiliation", data_type=DataType.TEXT),
                Property(name="status",      data_type=DataType.TEXT),
            ],
        )
        print("[db_setup] Created Weaviate collection: Person")

    if not client.collections.exists("FaceEmbedding"):
        client.collections.create(
            name="FaceEmbedding",
            properties=[
                # Stores the MinIO object key, e.g. "persons/<uuid>/<img_uuid>.jpg"
                Property(name="source_image", data_type=DataType.TEXT),
            ],
            references=[
                ReferenceProperty(name="person", target_collection="Person"),
            ],
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            ),
        )
        print("[db_setup] Created Weaviate collection: FaceEmbedding")

def setup_minio(
    endpoint: str = "localhost:9000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
    secure: bool = False,
    bucket: str = BUCKET_NAME,
) -> None:
    client = minio_create_client(endpoint, access_key, secret_key, secure)
    ensure_bucket(client, bucket)
    print(f"[db_setup] MinIO bucket ready: '{bucket}'")


def setup_all(
    weaviate_client: weaviate.WeaviateClient,
    minio_endpoint: str = "localhost:9000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
    minio_secure: bool = False,
    minio_bucket: str = BUCKET_NAME,
) -> None:
    setup_weaviate(weaviate_client)
    setup_minio(minio_endpoint, minio_access_key, minio_secret_key, minio_secure, minio_bucket)
    print("[db_setup] All stores initialised.")

if __name__ == "__main__":
    wv_client = weaviate.connect_to_local()
    try:
        setup_all(wv_client)
    finally:
        wv_client.close()