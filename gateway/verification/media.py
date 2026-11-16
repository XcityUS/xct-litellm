from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from io import BytesIO
from typing import Final, Protocol, cast
from urllib.parse import quote, urlparse

import httpx
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import TypeAdapter

from litellm.proxy._types import hash_token
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.management_endpoints.key_management_endpoints import (
    generate_key_helper_fn,  # pyright: ignore[reportUnknownVariableType]
)

logger = logging.getLogger(__name__)


class KeyTable(Protocol):
    async def delete(self, *, where: dict[str, str]) -> object: ...


class KeyWriter(Protocol):
    litellm_verificationtoken: KeyTable


IssueKey = Callable[..., Awaitable[Mapping[str, object]]]


def media_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(35.0))


async def delete_temporary_key(key: str, owner_id: str) -> None:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return
    try:
        db: object = prisma_client.db
        writer: Final[KeyWriter] = cast(KeyWriter, db.writer if isinstance(db, RoutingPrismaWrapper) else db)
        await writer.litellm_verificationtoken.delete(where={"token": hash_token(key)})
    except Exception:
        # Expiry still caps the key at two minutes; do not log its value.
        logger.warning("Invitation upload key cleanup failed for inviter %s", owner_id, exc_info=True)


def validate_photo(content: bytes, content_type: str) -> None:
    formats: Final = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
    if not 1 <= len(content) <= 10 * 1024 * 1024 or content_type not in formats:
        raise HTTPException(400, "PHOTO_INVALID")
    try:
        with Image.open(BytesIO(content)) as photo:
            width, height = photo.size
            if (
                photo.format != formats[content_type]
                or min(width, height) < 300
                or max(width, height) > 6000
                or not 0.4 < width / height < 2.5
            ):
                raise HTTPException(400, "PHOTO_INVALID")
            photo.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "PHOTO_INVALID") from exc


async def upload_for_inviter(owner_id: str, content: bytes, content_type: str, name: str) -> str:
    """File guest media under the inviter using a short-lived, info-only virtual key."""
    worker: Final = os.getenv("MEDIA_WORKER_URL", "https://media.xcity.ai").rstrip("/")
    worker_origin: Final = urlparse(worker)
    if worker_origin.scheme != "https" and not (worker_origin.scheme == "http" and worker_origin.hostname in {"localhost", "127.0.0.1"}):
        raise HTTPException(503, "INVITATIONS_UNAVAILABLE")
    issue_key: Final[IssueKey] = cast(IssueKey, generate_key_helper_fn)
    key_data: Final = await issue_key(
        request_type="key",
        user_id=owner_id,
        duration="2m",
        models=[],
        allowed_routes=["/key/info"],
        key_alias="portrait-invitation-upload",
        metadata={"purpose": "portrait-invitation-upload"},
    )
    key: Final = key_data.get("token")
    if not isinstance(key, str) or not key:
        raise HTTPException(503, "INVITATIONS_UNAVAILABLE")
    try:
        async with media_client() as client:
            async with asyncio.timeout(40):
                result: Final = await client.post(
                    f"{worker}/upload",
                    content=content,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": content_type, "X-Asset-Name": quote(name[:80])},
                )
        if result.status_code != 200:
            raise HTTPException(503, "INVITATIONS_UNAVAILABLE")
        payload: Final = TypeAdapter(dict[str, object]).validate_json(result.content)
        url: Final = payload.get("url")
        parsed: Final = urlparse(url) if isinstance(url, str) else None
        if (
            not isinstance(url, str)
            or parsed is None
            or parsed.scheme != worker_origin.scheme
            or parsed.netloc != worker_origin.netloc
            or not parsed.path.startswith(f"/media/u/{owner_id}/")
        ):
            raise HTTPException(503, "INVITATIONS_UNAVAILABLE")
        return url
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        raise HTTPException(503, "INVITATIONS_UNAVAILABLE") from exc
    finally:
        await delete_temporary_key(key, owner_id)
