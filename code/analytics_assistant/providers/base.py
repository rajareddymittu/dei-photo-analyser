"""Provider contract for analytics language models."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class AnalyticsProvider(Protocol):
    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield response text chunks for a chat conversation."""

