"""Chat and image-generation clients used by pipeline services."""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import asdict
from io import BytesIO
from typing import Any, Optional, Sequence

import requests
from openai import OpenAI
from PIL import Image

from ..config import ModelServiceConfig
from ..io import get_logger
from .runtime import create_client

logger = get_logger(__name__)


class VisionToTextModelClient:

    def __init__(self, model_config: ModelServiceConfig):
        self.model = create_client(model_config)

    def generate_text(self, prompt: str, images: Sequence[Image.Image]) -> str:
        return self.model.complete(prompt, images)

    def generate_json(self, prompt: str, images: Sequence[Image.Image]) -> Any:
        response_text = self.generate_text(prompt, images)
        
        results = []
        decoder = json.JSONDecoder()
        i = 0
        n = len(response_text)

        while i < n:
            if response_text[i] in ('{', '['):
                try:
                    obj, end = decoder.raw_decode(response_text[i:])
                    results.append(obj)
                    i += end
                    continue
                except json.JSONDecodeError:
                    pass
            i += 1
        
        if not results:
            # No JSON found: let json.loads raise so the caller sees the raw text.
            return json.loads(response_text)
        
        # Models often trail extra objects after the answer; the first one is the answer.
        return results[0]


class TextToTextModelClient:

    def __init__(self, model_config: ModelServiceConfig):
        self.model = create_client(model_config)

    def generate_text(self, prompt: str) -> str:
        return self.model.complete(prompt)

        

_POLL_INTERVAL_SECONDS = 2.0
_POLL_TIMEOUT_SECONDS = 300.0
_HTTP_TIMEOUT_SECONDS = 60.0


class TextToImageModelClient:
    """Generate a texture image.

    * ``openai_server`` — OpenAI Images API (``images.generate`` / ``images.edit``).
    * ``gemini`` — Google ``generate_content`` via ``google-genai``.
    * ``async_polling`` — POST a task, poll ``GET <api_base>/task/{id}``.
    """

    _DIRECT_PROVIDER = "openai_server"
    _GEMINI_PROVIDER = "gemini"
    _POLLING_PROVIDER = "async_polling"

    def __init__(self, model_config: ModelServiceConfig):
        assert isinstance(model_config, ModelServiceConfig)
        self._raw_config = asdict(model_config)
        self._provider = self._raw_config["provider"]
        supported = (self._DIRECT_PROVIDER, self._GEMINI_PROVIDER, self._POLLING_PROVIDER)
        if self._provider not in supported:
            raise ValueError(
                f"TextToImageModelClient: unsupported provider {self._provider!r}; "
                f"expected one of {supported}"
            )
        self._model_id = self._raw_config.get("model_id")
        if not self._model_id:
            raise ValueError("model_id is required in model configuration")
        self._client: Any = None

    def _openai_client(self) -> OpenAI:
        if self._client is None:
            api_key = self._raw_config.get("api_key")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required for texture generation")
            kwargs: dict[str, Any] = {"api_key": api_key}
            api_base = self._raw_config.get("api_base")
            if api_base:
                kwargs["base_url"] = api_base
            self._client = OpenAI(**kwargs)
        return self._client

    def _gemini_client(self) -> tuple[Any, Any]:
        from google import genai
        from google.genai import types as genai_types

        if self._client is None:
            api_key = self._raw_config.get("api_key")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is required for gemini texture generation")
            kwargs: dict[str, Any] = {"api_key": api_key}
            api_base = self._raw_config.get("api_base")
            if api_base:
                kwargs["http_options"] = genai_types.HttpOptions(base_url=api_base)
            self._client = genai.Client(**kwargs)
        return self._client, genai_types

    def _generate_image_gemini(
        self,
        prompt: str,
        output_path: str,
        reference_images: Optional[Sequence[Image.Image]] = None,
    ) -> str:
        client, genai_types = self._gemini_client()

        contents: list[Any] = [prompt]
        for img in reversed(list(reference_images or ())):
            buf = BytesIO()
            img.save(buf, format="PNG")
            contents.insert(
                0,
                genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
            )

        image_config = genai_types.ImageConfig(
            aspect_ratio=self._raw_config.get("aspect_ratio") or "1:1",
            image_size=self._raw_config.get("image_size") or "2K",
        )
        response = client.models.generate_content(
            model=self._model_id,
            contents=contents,
            config=genai_types.GenerateContentConfig(image_config=image_config),
        )

        for part in getattr(response, "parts", None) or ():
            if hasattr(part, "as_image"):
                img = part.as_image()
                if img is not None:
                    img.save(output_path)
                    return output_path
            inline = getattr(part, "inline_data", None)
            if inline is not None and isinstance(inline.data, (bytes, str)):
                payload = (
                    inline.data if isinstance(inline.data, bytes)
                    else base64.b64decode(inline.data)
                )
                with open(output_path, "wb") as handle:
                    handle.write(payload)
                return output_path

        raise RuntimeError("No image data found in gemini generation response")

    @staticmethod
    def _png_bytes(image: Image.Image) -> BytesIO:
        buf = BytesIO()
        image.convert("RGBA").save(buf, format="PNG")
        buf.name = "reference.png"
        buf.seek(0)
        return buf

    @staticmethod
    def _write_image_payload(data: Any, output_path: str) -> None:
        b64 = getattr(data, "b64_json", None)
        url = getattr(data, "url", None)
        if b64:
            with open(output_path, "wb") as handle:
                handle.write(base64.b64decode(b64))
            return
        if url:
            resp = requests.get(url, timeout=_HTTP_TIMEOUT_SECONDS)
            resp.raise_for_status()
            with Image.open(BytesIO(resp.content)) as im:
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                im.save(output_path)
            return
        raise RuntimeError(f"No image data in Images API response: {data!r}")

    def generate_image(
        self,
        prompt: str,
        output_path: str,
        reference_images: Optional[Sequence[Image.Image]] = None,
    ) -> str:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if self._provider == self._POLLING_PROVIDER:
            if reference_images:
                logger.warning(
                    "%s provider: reference_images are ignored.",
                    self._POLLING_PROVIDER,
                )
            return self._generate_image_polling(prompt, output_path)

        if self._provider == self._GEMINI_PROVIDER:
            return self._generate_image_gemini(prompt, output_path, reference_images)

        client = self._openai_client()
        size = self._raw_config.get("image_size") or "1024x1024"
        if reference_images:
            response = client.images.edit(
                model=self._model_id,
                image=self._png_bytes(reference_images[0]),
                prompt=prompt,
                n=1,
                size=size,
            )
        else:
            response = client.images.generate(
                model=self._model_id,
                prompt=prompt,
                n=1,
                size=size,
            )
        items = getattr(response, "data", None) or []
        if not items:
            raise RuntimeError("Images API returned no data")
        self._write_image_payload(items[0], output_path)
        return output_path


    def _require_api_key(self) -> str:
        key = self._raw_config.get("api_key")
        if not key:
            raise ValueError(
                f"{self._POLLING_PROVIDER}: api_key is required "
                "(set in config or via OPENAI_API_KEY)."
            )
        return key

    def _generate_image_polling(self, prompt: str, output_path: str) -> str:
        base_url = (self._raw_config.get("api_base") or "").rstrip("/")
        if not base_url:
            raise ValueError(f"{self._POLLING_PROVIDER}: api_base is required")
        headers = {
            "Authorization": f"Bearer {self._require_api_key()}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model_id,
            "input": [{
                "params": {
                    "prompt": prompt,
                    "aspect_ratio": self._raw_config.get("aspect_ratio", "1:1"),
                    "resolution": self._raw_config.get("resolution", "2K"),
                },
            }],
        }

        logger.info("submitting async generation task to %s", base_url)
        resp = requests.post(base_url, headers=headers, json=payload, timeout=_HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        task = resp.json()
        task_id = task.get("id")
        if not task_id:
            raise RuntimeError(f"async generation: response missing task id: {task!r}")

        task_url = f"{base_url}/task/{task_id}"
        deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
        last_progress: Optional[float] = None

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"async generation: task {task_id} did not complete within "
                    f"{_POLL_TIMEOUT_SECONDS:.0f}s"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)
            poll = requests.get(task_url, headers=headers, timeout=_HTTP_TIMEOUT_SECONDS)
            poll.raise_for_status()
            data = poll.json()
            status = data.get("status")
            if status == "completed":
                return self._download_first_image(data, output_path)
            if status in ("failed", "error", "canceled"):
                raise RuntimeError(
                    f"async generation: task {task_id} ended with status={status!r}, body={data!r}"
                )
            progress = data.get("progress")
            if progress != last_progress:
                logger.info("async generation: task %s status=%s progress=%s", task_id, status, progress)
                last_progress = progress

    @staticmethod
    def _download_first_image(task_payload: dict[str, Any], output_path: str) -> str:
        for out in task_payload.get("output") or []:
            for content in out.get("content") or []:
                if content.get("type") == "image" and content.get("url"):
                    url = content["url"]
                    logger.info("downloading generated image from %s", url)
                    img_resp = requests.get(url, timeout=_HTTP_TIMEOUT_SECONDS)
                    img_resp.raise_for_status()
                    # The remote may hand back any format (e.g. webp), so re-encode
                    # through PIL to match the extension output_path implies.
                    with Image.open(BytesIO(img_resp.content)) as im:
                        if im.mode not in ("RGB", "RGBA"):
                            im = im.convert("RGB")
                        im.save(output_path)
                    return output_path
        raise RuntimeError(f"async generation: no image url in completed task payload: {task_payload!r}")


class PromptEnricherService:
    def __init__(self, model_client: TextToTextModelClient):
        self.model_client = model_client

    @classmethod
    def from_config(cls, model_config: ModelServiceConfig) -> "PromptEnricherService":
        return cls(TextToTextModelClient(model_config))

    def enrich(self, description: str) -> str:
        from ..prompt import PROMPT_ENRICH_TEMPLATE
        request = PROMPT_ENRICH_TEMPLATE.format(description=description)
        try:
            result = self.model_client.generate_text(request)
        except Exception as e:
            logger.error("Prompt enrichment failed: %s", e)
            result = description
        else:
            logger.info("Prompt enrichment succeeded.")
        return result.strip()

    def __call__(self, *args, **kwargs):
        return self.enrich(*args, **kwargs)


class ImageGeneratorService:
    """Generates a single reference image from a prompt."""

    def __init__(self, model_client: TextToImageModelClient):
        self.model_client = model_client

    @classmethod
    def from_config(cls, model_config: ModelServiceConfig) -> "ImageGeneratorService":
        return cls(TextToImageModelClient(model_config))

    def generate(self, prompt: str, output_dir: str, reference_image_path: Optional[str] = None) -> Optional[str]:
        from ..io import load_images
        from ..prompt import IMAGE_GENERATE_TEMPLATE
        logger.info("Generating reference image...")
        reference_images = None
        if reference_image_path:
            reference_images = load_images([reference_image_path])
        output_path = os.path.join(output_dir, "generated_ref_image.png")
        full_prompt = IMAGE_GENERATE_TEMPLATE.format(prompt=prompt)
        save_path = self.model_client.generate_image(full_prompt, output_path, reference_images=reference_images)
        logger.info("Generated reference image saved to: %s", save_path)
        return save_path


class TextureGeneratorService:
    """Generates a single base color texture map."""

    def __init__(self, model_client: TextToImageModelClient):
        self.model_client = model_client

    @classmethod
    def from_config(cls, model_config: ModelServiceConfig) -> "TextureGeneratorService":
        return cls(TextToImageModelClient(model_config))

    def generate(self, description: str, output_dir: str, reference_images: Optional[Sequence[Image.Image]] = None) -> Optional[str]:
        from ..prompt import TEXTURE_GENERATION_TEMPLATE
        prompt = TEXTURE_GENERATION_TEMPLATE.format(description=description)
        logger.info("Generating base color texture for: %s...", description)
        output_path = os.path.join(output_dir, "base_color_texture.png")
        return self.model_client.generate_image(prompt, output_path, reference_images=reference_images)
