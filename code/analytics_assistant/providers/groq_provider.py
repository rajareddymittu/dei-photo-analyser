"""Groq Chat Completions adapter."""

from __future__ import annotations

from collections.abc import Iterator


class GroqAnalyticsProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[str]:
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Groq SDK is missing. Install the project dependencies from requirements.txt.") from exc

        client = Groq(api_key=self.api_key)
        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=600,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content
