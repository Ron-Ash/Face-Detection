from __future__ import annotations

import io
import uuid
from typing import Optional

from minio import Minio
from minio.error import S3Error
from PIL import Image


BUCKET_NAME = "face-images"
_DEFAULT_REGION = "us-east-1"


def create_client(endpoint: str = "localhost:9000", access_key: str = "minioadmin", secret_key: str = "minioadmin", secure: bool = False) -> Minio:
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

def ensure_bucket(client: Minio, bucket: str = BUCKET_NAME) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket, location=_DEFAULT_REGION)
        print(f"[minio_store] Created bucket '{bucket}'")




def upload_image(client: Minio, img: Image.Image, person_uuid: str, fmt: str = "JPEG", bucket: str = BUCKET_NAME) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    buf.seek(0)
    size = buf.getbuffer().nbytes

    ext = fmt.lower().replace("jpeg", "jpg")
    object_key = f"persons/{person_uuid}/{uuid.uuid4()}.{ext}"
    content_type = f"image/{ext}"

    client.put_object(
        bucket_name=bucket,
        object_name=object_key,
        data=buf,
        length=size,
        content_type=content_type,
    )
    print(f"[minio_store] Uploaded {object_key} ({size} bytes)")
    return object_key

def download_image(client: Minio, object_key: str, bucket: str = BUCKET_NAME) -> Optional[Image.Image]:
    try:
        response = client.get_object(bucket_name=bucket, object_name=object_key)
        data = response.read()
        response.close()
        response.release_conn()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except S3Error as e:
        print(f"[minio_store] download_image S3 error for '{object_key}': {e}")
        return None
    except Exception as e:
        print(f"[minio_store] download_image error for '{object_key}': {e}")
        return None

def download_images_for_person(client: Minio, object_keys: list[str], bucket: str = BUCKET_NAME) -> list[tuple[Image.Image, str]]:
    results = []
    for key in object_keys:
        img = download_image(client, key, bucket)
        if img is not None:
            results.append((img, key))
    return results




def get_presigned_url(client: Minio, object_key: str, expires_seconds: int = 3600, bucket: str = BUCKET_NAME) -> Optional[str]:
    from datetime import timedelta
    try:
        url = client.presigned_get_object(
            bucket_name=bucket,
            object_name=object_key,
            expires=timedelta(seconds=expires_seconds),
        )
        return url
    except S3Error as e:
        print(f"[minio_store] presigned_url error for '{object_key}': {e}")
        return None




def delete_image(client: Minio, object_key: str, bucket: str = BUCKET_NAME) -> bool:
    try:
        client.remove_object(bucket_name=bucket, object_name=object_key)
        print(f"[minio_store] Deleted {object_key}")
        return True
    except S3Error as e:
        print(f"[minio_store] delete_image error for '{object_key}': {e}")
        return False


def delete_all_images_for_person(client: Minio, person_uuid: str, bucket: str = BUCKET_NAME) -> int:
    prefix = f"persons/{person_uuid}/"
    objects = client.list_objects(bucket_name=bucket, prefix=prefix, recursive=True)
    count = 0
    for obj in objects:
        if delete_image(client, obj.object_name, bucket):
            count += 1
    return count