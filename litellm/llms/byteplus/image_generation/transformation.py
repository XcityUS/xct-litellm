import re
from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.exceptions import BadRequestError
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


# BytePlus (ByteDance Ark) requires the generated image area to be at least
# 3,686,400 pixels; the OpenAI default of 1024x1024 is rejected. Default to a
# valid size when the caller does not specify one.
DEFAULT_IMAGE_SIZE = "2048x2048"

# OpenAI sizes are all under that minimum; each becomes the 2K size of the same ratio from the `size` table of
# https://docs.byteplus.com/en/docs/ModelArk/1541523 (read 2026-11-19)
_OPENAI_SIZE_TO_SEEDREAM: Final = {
    "256x256": DEFAULT_IMAGE_SIZE,
    "512x512": DEFAULT_IMAGE_SIZE,
    "1024x1024": DEFAULT_IMAGE_SIZE,
    "1536x1024": "2496x1664",
    "1024x1536": "1664x2496",
    "1792x1024": "2848x1600",
    "1024x1792": "1600x2848",
}

# https://docs.byteplus.com/en/docs/ModelArk/1541523 (Image generation API, `image`), read 2026-11-19: "Seedream 5.0
# lite, 4.5, and 4.0 support up to 14 reference images". The 5.0 pro / flash and 5.5 models allow 10; BytePlus itself
# rejects those above their own limit.
MAX_REFERENCE_IMAGES: Final = 14
_REFERENCE_IMAGE: Final = re.compile(
    r"https://[^\s/?#]+\S*|data:image/(?:png|jpeg|jpg|webp);base64,[A-Za-z0-9+/]+={0,2}"
)
_REFERENCE_IMAGE_FORMATS: Final = "an https:// URL or a data:image/<png|jpeg|jpg|webp>;base64,<data> URI"


def _is_reference_image(image: object) -> bool:
    return isinstance(image, str) and _REFERENCE_IMAGE.fullmatch(image) is not None


def reference_image_error(image: object) -> str | None:
    """Why ``image`` is not a valid Seedream reference image field, or None when it is. Never echoes the payload."""
    if isinstance(image, str):
        return None if _is_reference_image(image) else f"`image` must be {_REFERENCE_IMAGE_FORMATS}"
    if not isinstance(image, list) or not image:
        return "`image` must be a string or a non-empty list of strings"
    if len(image) > MAX_REFERENCE_IMAGES:
        return f"`image` accepts at most {MAX_REFERENCE_IMAGES} reference images, got {len(image)}"
    invalid: Final = next((index for index, item in enumerate(image) if not _is_reference_image(item)), None)
    return None if invalid is None else f"`image[{invalid}]` must be {_REFERENCE_IMAGE_FORMATS}"


class BytePlusImageGenerationConfig(BaseImageGenerationConfig):
    """
    BytePlus image generation (seedream models).

    Reference: https://docs.byteplus.com/en/docs/ModelArk
    The endpoint is OpenAI-compatible: request {model, prompt, size, response_format, n},
    response {model, created, data: [{url | b64_json}], usage}.
    """

    DEFAULT_BASE_URL: str = "https://ark.ap-southeast.bytepluses.com/api/v3"
    IMAGE_GENERATION_ENDPOINT: str = "images/generations"

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "response_format", "size", "image"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        image_error: Final = (
            reference_image_error(non_default_params["image"]) if "image" in non_default_params else None
        )
        if image_error is not None:
            raise BadRequestError(message=image_error, model=model, llm_provider="byteplus")

        for k, v in non_default_params.items():
            if k in optional_params:
                continue
            if k in supported_params:
                # BytePlus is OpenAI-compatible — pass params through unchanged.
                optional_params[k] = v
            elif drop_params:
                pass
            else:
                raise ValueError(
                    f"Parameter {k} is not supported for model {model}. Supported "
                    f"parameters are {supported_params}. Set drop_params=True to drop "
                    "unsupported parameters."
                )

        # BytePlus rejects sizes under 3,686,400 px (e.g. the OpenAI 1024x1024
        # default). Apply a valid default when size is not provided.
        optional_params.setdefault("size", DEFAULT_IMAGE_SIZE)
        size: Final = optional_params["size"]
        if isinstance(size, str) and size in _OPENAI_SIZE_TO_SEEDREAM:
            optional_params["size"] = _OPENAI_SIZE_TO_SEEDREAM[size]
        # Default to base64 output when the caller does not specify a format.
        # BytePlus otherwise returns a signed URL, but several OpenAI-compatible
        # clients (e.g. LibreChat's image_gen_oai tool) only read `b64_json`
        # from the response and would otherwise see an empty image.
        optional_params.setdefault("response_format", "b64_json")
        return optional_params

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        complete_url: str = api_base or get_secret_str("BYTEPLUS_API_BASE") or self.DEFAULT_BASE_URL
        complete_url = complete_url.rstrip("/")
        return f"{complete_url}/{self.IMAGE_GENERATION_ENDPOINT}"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        final_api_key: str | None = api_key or get_secret_str("BYTEPLUS_API_KEY")
        if not final_api_key:
            raise ValueError("BYTEPLUS_API_KEY is not set")

        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return {"model": model, "prompt": prompt, **optional_params}

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []

        for image in response_data.get("data", []):
            model_response.data.append(
                ImageObject(
                    url=image.get("url"),
                    b64_json=image.get("b64_json"),
                    revised_prompt=image.get("revised_prompt"),
                )
            )

        if "created" in response_data:
            model_response.created = response_data["created"]

        return model_response
