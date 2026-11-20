"""
Unit tests for the BytePlus image-generation and video-generation handlers.

Transformation-layer tests (no network) plus env-gated live smoke tests.
"""

import base64
import json
import os
import sys
from typing import Final

import httpx
import pytest

workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


# ----------------------------- image -----------------------------
class TestBytePlusImageGeneration:
    def _cfg(self):
        return ProviderConfigManager.get_provider_image_generation_config(
            model="seedream-4-5-251128", provider=LlmProviders.BYTEPLUS
        )

    def test_config_resolves(self):
        from litellm.llms.byteplus.image_generation.transformation import (
            BytePlusImageGenerationConfig,
        )

        assert isinstance(self._cfg(), BytePlusImageGenerationConfig)

    def test_complete_url(self):
        url = self._cfg().get_complete_url(
            api_base=None,
            api_key="x",
            model="seedream-4-5-251128",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"

    def test_size_defaults_to_valid_min_area(self):
        # BytePlus rejects images under 3,686,400 px; default must be applied.
        mapped = self._cfg().map_openai_params(
            non_default_params={}, optional_params={}, model="m", drop_params=False
        )
        assert mapped["size"] == "2048x2048"

    def test_params_pass_through(self):
        mapped = self._cfg().map_openai_params(
            non_default_params={"size": "2848x1600", "response_format": "url", "n": 2},
            optional_params={},
            model="m",
            drop_params=False,
        )
        assert mapped == {"size": "2848x1600", "response_format": "url", "n": 2}

    def test_validate_environment_requires_key(self, monkeypatch):
        monkeypatch.delenv("BYTEPLUS_API_KEY", raising=False)
        with pytest.raises(ValueError, match='BYTEPLUS_API_KEY is not set'):
            self._cfg().validate_environment(
                headers={},
                model="m",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )
        h = self._cfg().validate_environment(
            headers={}, model="m", messages=[], optional_params={}, litellm_params={},
            api_key="ark-xyz",
        )
        assert h["Authorization"] == "Bearer ark-xyz"

    def test_response_parsing(self):
        raw = httpx.Response(
            200,
            json={
                "model": "seedream-4-5-251128",
                "created": 1780576790,
                "data": [{"url": "https://example.com/img.jpeg"}],
                "usage": {"generated_images": 1},
            },
        )
        out = self._cfg().transform_image_generation_response(
            model="seedream-4-5-251128",
            raw_response=raw,
            model_response=ImageResponse(),
            logging_obj=None,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert out.data[0].url == "https://example.com/img.jpeg"

    def test_cost_calculator(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")
        from litellm.llms.byteplus.image_generation.cost_calculator import (
            cost_calculator,
        )

        resp = ImageResponse()
        resp.data = [object()]  # one image
        cost = cost_calculator(
            model="byteplus/seedream-4-5-251128", image_response=resp
        )
        assert cost == 0.04


_SEEDREAM: Final = "byteplus/seedream-5-0-260128"
# "Seedream 5.0 lite, 4.5, and 4.0 support up to 14 reference images": https://docs.byteplus.com/en/docs/ModelArk/1541523
# (Image generation API, `image`), read 2026-11-19
MAX_REFERENCE_IMAGES: Final = 14
_URL_IMAGE: Final = "https://media.example.test/portrait.jpg"
_BASE64_IMAGE: Final = "data:image/png;base64," + base64.b64encode(b"\x89PNG portrait bytes").decode()


def _generate(sent: list[dict[str, object]], **params: object) -> ImageResponse:
    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "aW1n"}, {"b64_json": "aW1n"}]})

    return litellm.image_generation(
        model=_SEEDREAM,
        prompt="the same person on a beach",
        api_key="test-key",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
        **params,
    )


class TestBytePlusReferenceImages:
    @pytest.mark.parametrize(
        "image",
        [
            _URL_IMAGE,
            _BASE64_IMAGE,
            [_URL_IMAGE, _BASE64_IMAGE],
            [_BASE64_IMAGE] * MAX_REFERENCE_IMAGES,
            ["data:image/jpeg;base64,AAAA", "data:image/jpg;base64,AA==", "data:image/webp;base64,QQ="],
        ],
    )
    def test_reference_images_reach_the_provider_unchanged(self, image: str | list[str]) -> None:
        sent: Final[list[dict[str, object]]] = []
        _generate(sent, image=image)
        assert len(sent) == 1
        assert sent[0]["image"] == image

    @pytest.mark.parametrize(
        "image",
        [
            "http://media.example.test/portrait.jpg",
            "https://",
            "https://media.example.test/a portrait.jpg",
            "ftp://media.example.test/portrait.jpg",
            "data:image/gif;base64,R0lGODlh",
            "data:image/PNG;base64,AAAA",
            "data:image/png;base64,",
            "data:image/png;base64,AA AA",
            "data:image/png,AAAA",
            "",
            [],
            [_BASE64_IMAGE] * (MAX_REFERENCE_IMAGES + 1),
            [_URL_IMAGE, 7],
            [_URL_IMAGE, "data:text/plain;base64,AAAA"],
            {"url": _URL_IMAGE},
            7,
        ],
    )
    def test_invalid_reference_images_are_a_400_before_any_provider_call(self, image: object) -> None:
        sent: Final[list[dict[str, object]]] = []
        with pytest.raises(litellm.BadRequestError) as raised:
            _generate(sent, image=image)
        assert raised.value.status_code == 400
        assert sent == []
        assert "`image" in str(raised.value)

    def test_rejection_names_the_bad_entry_without_echoing_any_payload(self) -> None:
        secret_payload: Final = "data:image/bmp;base64," + "c2VjcmV0" * 64
        with pytest.raises(litellm.BadRequestError) as raised:
            _generate([], image=[_BASE64_IMAGE, secret_payload])
        assert "`image[1]`" in str(raised.value)
        assert "c2VjcmV0" not in str(raised.value)
        assert _BASE64_IMAGE.split(",")[1] not in str(raised.value)

    @pytest.mark.usefixtures("local_model_cost_map")
    def test_reference_images_do_not_change_what_the_generated_images_cost(self) -> None:
        rate: Final = litellm.get_model_info(_SEEDREAM)["output_cost_per_image"]
        plain: Final = _generate([])
        referenced: Final = _generate([], image=[_URL_IMAGE, _BASE64_IMAGE])

        def cost(response: ImageResponse) -> float:
            return litellm.completion_cost(completion_response=response, model=_SEEDREAM, call_type="image_generation")

        assert rate is not None and rate > 0
        assert cost(referenced) == cost(plain) == 2 * rate

    def test_reference_images_are_top_level_fields_not_nested_in_extra_body(self) -> None:
        sent: Final[list[dict[str, object]]] = []
        _generate(sent, image=[_BASE64_IMAGE], size="2848x1600", additional_drop_params=["quality"])
        assert sent[0]["image"] == [_BASE64_IMAGE]
        assert "extra_body" not in sent[0]
        assert sent[0]["size"] == "2848x1600"


class TestBytePlusImageSize:
    @pytest.mark.parametrize("size", ["2848x1600", "1600x2848", "2304x1728", "2K", "3136x1344"])
    def test_seedream_sizes_reach_the_provider_unchanged(self, size: str) -> None:
        sent: Final[list[dict[str, object]]] = []
        _generate(sent, size=size)
        assert sent[0]["size"] == size

    @pytest.mark.parametrize(
        ("size", "sent_size"),
        [
            ("1024x1024", "2048x2048"),
            ("512x512", "2048x2048"),
            ("256x256", "2048x2048"),
            ("1536x1024", "2496x1664"),
            ("1024x1536", "1664x2496"),
            ("1792x1024", "2848x1600"),
            ("1024x1792", "1600x2848"),
        ],
    )
    def test_openai_sizes_become_the_seedream_size_of_the_same_ratio(self, size: str, sent_size: str) -> None:
        sent: Final[list[dict[str, object]]] = []
        _generate(sent, size=size)
        assert sent[0]["size"] == sent_size

    def test_no_size_is_the_square_default(self) -> None:
        sent: Final[list[dict[str, object]]] = []
        _generate(sent)
        assert sent[0]["size"] == "2048x2048"


# ----------------------------- video -----------------------------
class TestBytePlusVideoGeneration:
    def _cfg(self):
        return ProviderConfigManager.get_provider_video_config(
            model="seedance-1-5-pro-251215", provider=LlmProviders.BYTEPLUS
        )

    def test_config_resolves(self):
        from litellm.llms.byteplus.videos.transformation import BytePlusVideoConfig

        assert isinstance(self._cfg(), BytePlusVideoConfig)

    def test_size_to_aspect_ratio(self):
        from litellm.types.videos.main import VideoCreateOptionalRequestParams

        mapped = self._cfg().map_openai_params(
            VideoCreateOptionalRequestParams(size="1280x720", seconds=5),
            model="seedance-1-5-pro-251215",
            drop_params=False,
        )
        assert mapped["ratio"] == "16:9"
        assert mapped["duration"] == 5

    def test_create_request_builds_content_array(self):
        from litellm.types.router import GenericLiteLLMParams

        data, files, url = self._cfg().transform_video_create_request(
            model="seedance-1-5-pro-251215",
            prompt="ocean wave",
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            video_create_optional_request_params={
                "ratio": "16:9",
                "duration": 5,
                "generate_audio": True,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url.endswith("/contents/generations/tasks")
        assert data["model"] == "seedance-1-5-pro-251215"
        assert data["content"] == [{"type": "text", "text": "ocean wave"}]
        assert data["ratio"] == "16:9" and data["generate_audio"] is True

    def test_create_request_with_image_reference(self):
        from litellm.types.router import GenericLiteLLMParams

        data, _, _ = self._cfg().transform_video_create_request(
            model="seedance-1-5-pro-251215",
            prompt="make it move",
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            video_create_optional_request_params={
                "input_reference": "https://example.com/a.jpg"
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["content"] == [
            {"type": "text", "text": "make it move"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
        ]

    def test_create_request_with_multiple_reference_images(self):
        from litellm.types.router import GenericLiteLLMParams

        data, _, _ = self._cfg().transform_video_create_request(
            model="dreamina-seedance-2-5-260628",
            prompt="keep the same rider as [Image 1] and bike as [Image 2]",
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            video_create_optional_request_params={
                "input_reference": [
                    "https://example.com/a.jpg",
                    "https://example.com/b.jpg",
                ]
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["content"][1:] == [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/a.jpg"},
                "role": "reference_image",
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/b.jpg"},
                "role": "reference_image",
            },
        ]

    def test_create_request_with_explicit_frame_roles(self):
        from litellm.types.router import GenericLiteLLMParams

        data, _, _ = self._cfg().transform_video_create_request(
            model="dreamina-seedance-2-5-260628",
            prompt="from dawn to dusk",
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            video_create_optional_request_params={
                "input_reference": [
                    {"url": "https://example.com/first.jpg", "role": "first_frame"},
                    {"url": "https://example.com/last.jpg", "role": "last_frame"},
                ]
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["content"][1:] == [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/first.jpg"},
                "role": "first_frame",
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/last.jpg"},
                "role": "last_frame",
            },
        ]

    def test_create_request_with_video_and_audio_references(self):
        from litellm.types.router import GenericLiteLLMParams

        data, _, _ = self._cfg().transform_video_create_request(
            model="dreamina-seedance-2-5-260628",
            prompt="use the framing of [Video 1] and [Audio 1] as BGM",
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            video_create_optional_request_params={
                "input_reference": [
                    "https://example.com/subject.jpg",
                    {"url": "https://example.com/style.mp4", "role": "reference_video"},
                    {"url": "https://example.com/bgm.mp3", "role": "reference_audio"},
                ]
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["content"][1:] == [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/subject.jpg"},
                "role": "reference_image",
            },
            {
                "type": "video_url",
                "video_url": {"url": "https://example.com/style.mp4"},
                "role": "reference_video",
            },
            {
                "type": "audio_url",
                "audio_url": {"url": "https://example.com/bgm.mp3"},
                "role": "reference_audio",
            },
        ]

    def test_response_maps_last_frame_url_and_seed(self):
        cfg = self._cfg()
        succeeded = {
            "id": "cgt-456",
            "model": "dreamina-seedance-2-5-260628",
            "status": "succeeded",
            "content": {
                "video_url": "https://example.com/v.mp4",
                "last_frame_url": "https://example.com/last.png",
            },
            "seed": 42,
            "created_at": 1780577242,
            "duration": 8,
        }
        out = cfg.transform_video_status_retrieve_response(
            raw_response=httpx.Response(200, json=succeeded),
            logging_obj=None,
            custom_llm_provider="byteplus",
        )
        assert out.output_url == "https://example.com/v.mp4"
        assert out.last_frame_url == "https://example.com/last.png"
        assert out.seed == 42

    def test_status_mapping_and_url_extraction(self):
        cfg = self._cfg()
        assert cfg._map_status("queued") == "queued"
        assert cfg._map_status("running") == "in_progress"
        assert cfg._map_status("succeeded") == "completed"
        assert cfg._map_status("failed") == "failed"

        succeeded = {
            "id": "cgt-123",
            "model": "seedance-1-5-pro-251215",
            "status": "succeeded",
            "content": {"video_url": "https://example.com/v.mp4"},
            "created_at": 1780577242,
            "duration": 5,
        }
        out = cfg.transform_video_status_retrieve_response(
            raw_response=httpx.Response(200, json=succeeded),
            logging_obj=None,
            custom_llm_provider="byteplus",
        )
        assert out.status == "completed"
        # The video bytes are fetched via video_content, which extracts the URL.
        assert (
            cfg._extract_video_url_from_response(succeeded)
            == "https://example.com/v.mp4"
        )
        # Still-processing / failed responses surface a clear error.
        with pytest.raises(ValueError, match='Video is still processing'):
            cfg._extract_video_url_from_response({"status": "running"})

    @pytest.mark.skipif(
        not os.environ.get("BYTEPLUS_API_KEY"), reason="BYTEPLUS_API_KEY not set"
    )
    def test_video_create_live(self):
        v = litellm.video_generation(
            model="byteplus/seedance-1-5-pro-251215",
            prompt="a calm ocean wave",
            seconds=5,
            size="1280x720",
        )
        assert v.id and v.status in ("queued", "in_progress", "completed")
