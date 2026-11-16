from __future__ import annotations

from io import BytesIO
from typing import Final
from urllib.parse import quote

import httpx
import pytest
from fastapi import HTTPException
from PIL import Image

from gateway.verification.media import upload_for_inviter, validate_photo


def test_guest_photo_rejects_corrupt_or_undersized_image() -> None:
    with pytest.raises(HTTPException):
        validate_photo(b"not a photo", "image/png")
    small = BytesIO()
    Image.new("RGB", (100, 100), "white").save(small, format="PNG")
    with pytest.raises(HTTPException):
        validate_photo(small.getvalue(), "image/png")


@pytest.mark.asyncio
async def test_guest_upload_uses_inviter_scoped_temporary_key_and_removes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    generated: list[dict[str, object]] = []
    deleted: list[tuple[str, str]] = []
    real_client: Final = httpx.AsyncClient

    async def issue_key(**kwargs: object) -> dict[str, object]:
        generated.append(kwargs)
        return {"token": "sk-test-invitation-upload"}

    async def delete(key: str, owner: str) -> None:
        deleted.append((key, owner))

    def transport(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://media.xcity.ai/upload"
        assert request.headers["authorization"] == "Bearer sk-test-invitation-upload"
        assert request.headers["x-asset-name"] == quote("本人照片.png")
        return httpx.Response(200, json={"url": "https://media.xcity.ai/media/u/creator/refs/photo.png"})

    def client_factory() -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(transport), timeout=35)

    monkeypatch.setattr("gateway.verification.media.generate_key_helper_fn", issue_key)
    monkeypatch.setattr("gateway.verification.media.media_client", client_factory)
    monkeypatch.setattr("gateway.verification.media.delete_temporary_key", delete)
    result: Final = await upload_for_inviter("creator", b"image", "image/png", "本人照片.png")
    assert result == "https://media.xcity.ai/media/u/creator/refs/photo.png"
    assert generated[0]["user_id"] == "creator"
    assert generated[0]["allowed_routes"] == ["/key/info"]
    assert generated[0]["duration"] == "2m"
    assert deleted == [("sk-test-invitation-upload", "creator")]


@pytest.mark.asyncio
async def test_guest_upload_rejects_worker_url_outside_inviter_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    real_client: Final = httpx.AsyncClient

    async def issue_key(**kwargs: object) -> dict[str, object]:
        return {"token": "sk-test-invitation-upload"}

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"url": "https://media.xcity.ai/media/u/other/refs/photo.png"})

    def client_factory() -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(transport), timeout=35)

    async def delete(key: str, owner: str) -> None:
        return None

    monkeypatch.setattr("gateway.verification.media.generate_key_helper_fn", issue_key)
    monkeypatch.setattr("gateway.verification.media.media_client", client_factory)
    monkeypatch.setattr("gateway.verification.media.delete_temporary_key", delete)
    with pytest.raises(HTTPException) as caught:
        await upload_for_inviter("creator", b"image", "image/png", "photo.png")
    assert caught.value.status_code == 503
