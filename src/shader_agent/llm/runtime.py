"""Bounded multimodal chat clients.

Models return text/JSON/DSL; ShaderAgent validates and applies it locally.
Two wire formats:

* ``openai_server`` — OpenAI Chat Completions (GPT, Gemini's OpenAI layer, Kimi, proxies)
* ``anthropic`` — Anthropic Messages (official Claude)
"""
from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any, Sequence

from openai import OpenAI
from PIL import Image

from ..config import AgentConfig, ModelServiceConfig, resolve_api_key
from ..usage import get_active

ClientConfig = ModelServiceConfig | AgentConfig


def stream_enabled() -> bool:
    """Whether model responses should be streamed.

    Off by default; set ``SHADER_AGENT_STREAM=1`` when an intermediate proxy drops
    long idle connections. A thinking-heavy Designer call can sit quiet for well
    over a minute before the first token.
    """
    return os.environ.get("SHADER_AGENT_STREAM", "").strip().lower() in ("1", "true", "yes")


def _jpeg_base64(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((768, 768))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_url(image: Image.Image) -> str:
    return f"data:image/jpeg;base64,{_jpeg_base64(image)}"


def _record_usage(label: str, prompt_tokens: int, completion_tokens: int) -> None:
    collector = get_active()
    if collector is not None:
        collector.record(label, prompt_tokens, completion_tokens)


class LightweightClient:
    """One bounded OpenAI-compatible multimodal completion."""

    def __init__(
        self,
        config: ClientConfig,
        client: Any | None = None,
        stream: bool | None = None,
    ):
        self.runtime = "lightweight"
        if config.provider != "openai_server":
            raise ValueError("LightweightClient requires provider=openai_server")
        if not config.api_base:
            raise ValueError("api_base is required for provider=openai_server")
        self.model_id = config.model_id
        self.label = f"{config.provider}:{config.model_id}"
        api_key = config.api_key or resolve_api_key(config.provider)
        self.client = client or OpenAI(api_key=api_key, base_url=config.api_base)
        if stream is None:
            stream = stream_enabled()
        self.stream = stream

    def complete(self, prompt: str, images: Sequence[Image.Image] = ()) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": _image_url(image)}} for image in images)
        if self.stream:
            return self._complete_streaming(content)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
        )
        usage = response.usage
        _record_usage(
            self.label,
            getattr(usage, "prompt_tokens", 0) if usage else 0,
            getattr(usage, "completion_tokens", 0) if usage else 0,
        )
        return response.choices[0].message.content or ""

    def _complete_streaming(self, content: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        usage = None
        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            stream=True,
            stream_options={"include_usage": True},
        )
        for event in stream:
            if getattr(event, "usage", None):
                usage = event.usage
            for choice in getattr(event, "choices", None) or ():
                piece = getattr(choice.delta, "content", None)
                if piece:
                    parts.append(piece)
        _record_usage(
            self.label,
            getattr(usage, "prompt_tokens", 0) if usage else 0,
            getattr(usage, "completion_tokens", 0) if usage else 0,
        )
        return "".join(parts)


class AnthropicClient:
    """One bounded Anthropic Messages completion."""

    def __init__(
        self,
        config: ClientConfig,
        client: Any | None = None,
        stream: bool | None = None,
        max_tokens: int = 16384,
    ):
        self.runtime = "lightweight"
        if config.provider != "anthropic":
            raise ValueError("AnthropicClient requires provider=anthropic")
        self.model_id = config.model_id
        self.label = f"{config.provider}:{config.model_id}"
        self.max_tokens = max_tokens
        api_key = config.api_key or resolve_api_key(config.provider)
        if client is not None:
            self.client = client
        else:
            import anthropic

            kwargs: dict[str, Any] = {"api_key": api_key}
            if config.api_base:
                kwargs["base_url"] = config.api_base
            self.client = anthropic.Anthropic(**kwargs)
        if stream is None:
            stream = stream_enabled()
        self.stream = stream

    def complete(self, prompt: str, images: Sequence[Image.Image] = ()) -> str:
        content: list[dict[str, Any]] = []
        for image in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _jpeg_base64(image),
                    },
                }
            )
        content.append({"type": "text", "text": prompt})
        kwargs = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if self.stream:
            return self._complete_streaming(kwargs)
        message = self.client.messages.create(**kwargs)
        usage = getattr(message, "usage", None)
        _record_usage(
            self.label,
            getattr(usage, "input_tokens", 0) if usage else 0,
            getattr(usage, "output_tokens", 0) if usage else 0,
        )
        return _anthropic_text(message)

    def _complete_streaming(self, kwargs: dict[str, Any]) -> str:
        parts: list[str] = []
        with self.client.messages.stream(**kwargs) as stream:
            for piece in stream.text_stream:
                if piece:
                    parts.append(piece)
            message = stream.get_final_message()
        usage = getattr(message, "usage", None)
        _record_usage(
            self.label,
            getattr(usage, "input_tokens", 0) if usage else 0,
            getattr(usage, "output_tokens", 0) if usage else 0,
        )
        return "".join(parts) or _anthropic_text(message)


def _anthropic_text(message: Any) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", None) or ():
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


def create_client(
    config: ClientConfig,
    client: Any | None = None,
    stream: bool | None = None,
) -> LightweightClient | AnthropicClient:
    provider = getattr(config, "provider", "openai_server")
    if provider == "anthropic":
        return AnthropicClient(config, client=client, stream=stream)
    if provider == "openai_server":
        return LightweightClient(config, client=client, stream=stream)
    raise ValueError(
        f"Unknown provider {provider!r}; use openai_server or anthropic."
    )
