"""Provider configuration and selection for analytics chat."""

from __future__ import annotations

import os

import streamlit as st

from .base import AnalyticsProvider


def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or default)


def configured_provider() -> tuple[AnalyticsProvider | None, str, str]:
    provider_name = _setting("ANALYTICS_LLM_PROVIDER", "groq").lower()
    model = _setting("GROQ_MODEL", "openai/gpt-oss-120b")
    if provider_name != "groq":
        return None, provider_name, f"Provider '{provider_name}' is not registered yet. Add an adapter under analytics_assistant/providers/ and register it here."
    api_key = _setting("GROQ_API_KEY")
    if not api_key:
        return None, provider_name, "Add GROQ_API_KEY to .streamlit/secrets.toml or your environment to enable analytics chat."
    from .groq_provider import GroqAnalyticsProvider

    return GroqAnalyticsProvider(api_key=api_key, model=model), provider_name, ""


__all__ = ["configured_provider", "AnalyticsProvider"]
