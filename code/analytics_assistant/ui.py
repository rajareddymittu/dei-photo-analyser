"""Streamlit chat UI for deterministic photo analytics context."""

from __future__ import annotations

import json
import streamlit as st

from .providers import configured_provider


SYSTEM_PROMPT = """You are the expert AI analytics assistant for a high-end clothing photo intelligence dashboard.
Answer questions accurately, insightfully, and concisely using the supplied analytics JSON.

Key data sections available to you:
1. `brand_analytics`: Covers target sportswear brands (Nike, Puma, Adidas, Under Armour, New Balance) and unbranded/other items. Contains:
   - `target_brand_counts`: Detection counts per brand.
   - `top_brand` and `top_brand_count`: Leading brand and its sightings.
   - `branded_penetration_percent`: Percentage of persons detected wearing target brands.
   - `rankings`: Ordered list of brands with counts and market share percentages.
   - `brand_co_occurrences`: Associated garments, colors, styles, and unique photos for each brand.
   - `insights`: Generated strategic insights.
2. `all_results`: Aggregate wardrobe metrics including `tag_appearances` (garments, colors, patterns, styles, brands), `attribute_coverage`, `people_per_photo`, and monthly trends.
3. `face_tracking`: Face identity counts, repeat person sightings across photos, and match review statuses.
4. `current_analysis_filter`: Active filter slice if the user has focused on a specific attribute or label.

Guidelines for answering:
- When answering brand questions (rankings, counts, or comparisons between Nike, Puma, Adidas, Under Armour, New Balance):
  * State exact numbers and percentages from `brand_analytics`.
  * Use concise markdown comparison tables or structured bullet points.
  * Highlight the leader (e.g. "**Nike** leads with 14 appearances (45.2% share)...").
- When asked about outfits, colors, or styles paired with a brand, use the data in `brand_co_occurrences`.
- Tag appearances reflect computer-vision suggestions rather than verified receipts.
- If no classification data exists yet, inform the user to process photos in the Upload or Train photos tab first.
- Always provide structured, elegant, professional answers with bold headers and clear formatting."""

def get_chat_css(is_dark: bool = False) -> str:
    """Return theme-adaptive CSS for the AI Analytics Assistant chat."""
    if is_dark:
        user_bg = "linear-gradient(135deg, #4F46E5 0%, #6366F1 100%)"
        user_shadow = "0 4px 16px rgba(99, 102, 241, 0.4)"
        assistant_bg = "rgba(30, 41, 59, 0.85)"
        assistant_border = "1px solid rgba(255, 255, 255, 0.1)"
        assistant_shadow = "0 4px 16px rgba(0, 0, 0, 0.3)"
        assistant_text = "#F8FAFC"
        banner_bg = "linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%)"
        banner_border = "rgba(99, 102, 241, 0.3)"
        title_color = "#F8FAFC"
        subtitle_color = "#94A3B8"
    else:
        user_bg = "linear-gradient(135deg, #4338CA 0%, #4F46E5 100%)"
        user_shadow = "0 4px 14px rgba(67, 56, 202, 0.25)"
        assistant_bg = "#F8FAFC"
        assistant_border = "1px solid #E2E8F0"
        assistant_shadow = "0 2px 8px rgba(15, 23, 42, 0.04)"
        assistant_text = "#0F172A"
        banner_bg = "linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(14, 165, 233, 0.05) 100%)"
        banner_border = "rgba(79, 70, 229, 0.2)"
        title_color = "#1E293B"
        subtitle_color = "#64748B"

    return f"""
    <style>
    /* Chat Container & Header */
    .chat-hero-banner {{
        background: {banner_bg};
        border: 1px solid {banner_border};
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .chat-hero-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {title_color};
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .chat-hero-subtitle {{
        font-size: 0.85rem;
        color: {subtitle_color};
        margin-top: 4px;
    }}

    /* ========================================================
       USER CHAT BUBBLE - RIGHT ALIGNED
       ======================================================== */
    div[data-testid="stChatMessage"]:has([aria-label*="user" i]),
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        display: flex !important;
        flex-direction: row-reverse !important;
        align-items: flex-start !important;
        justify-content: flex-start !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        max-width: 78% !important;
        padding: 6px 0 !important;
        gap: 10px !important;
    }}

    div[data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
        background: {user_bg} !important;
        color: #FFFFFF !important;
        border-radius: 20px 20px 4px 20px !important;
        padding: 12px 18px !important;
        box-shadow: {user_shadow} !important;
        text-align: left !important;
        border: none !important;
    }}

    div[data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] span {{
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        margin: 0 !important;
    }}

    div[data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageAvatarUser"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {{
        background: #4F46E5 !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 6px rgba(79, 70, 229, 0.3) !important;
    }}

    /* ========================================================
       ASSISTANT CHAT BUBBLE - LEFT ALIGNED
       ======================================================== */
    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]),
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        display: flex !important;
        flex-direction: row !important;
        align-items: flex-start !important;
        justify-content: flex-start !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        max-width: 88% !important;
        padding: 6px 0 !important;
        gap: 12px !important;
    }}

    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {{
        background: {assistant_bg} !important;
        border: {assistant_border} !important;
        border-radius: 20px 20px 20px 4px !important;
        padding: 16px 22px !important;
        box-shadow: {assistant_shadow} !important;
        color: {assistant_text} !important;
        text-align: left !important;
    }}

    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] li {{
        color: {assistant_text} !important;
    }}

    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageAvatarAssistant"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageAvatarAssistant"] {{
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.25) !important;
    }}
    </style>
    """


SUGGESTIONS: dict[str, str] = {
    "🏆 Top brands": "Which sportswear brands are most commonly detected and what is their market share?",
    "⚖️ Compare Nike vs Adidas": "Compare Nike and Adidas in terms of sightings, styles, and outfit colors.",
    "👕 Outfit pairings": "What garments and styles are most often paired with each sportswear brand?",
    "📊 Brand penetration": "What percentage of detected people are wearing target brands vs unbranded clothing?",
    "👥 Face tracking": "Summarize the face tracking metrics and repeat identities across photos.",
}


def render_analytics_chat(context: dict, is_dark: bool = False) -> None:
    """Render the AI Analytics Assistant with right-aligned user questions and left-aligned replies."""
    st.html(get_chat_css(is_dark))

    provider, provider_name, setup_message = configured_provider()
    if provider is None:
        st.info(f"{setup_message} Configure `ANALYTICS_LLM_PROVIDER` and `GROQ_API_KEY` in `.streamlit/secrets.toml`.")
        return

    if "analytics_chat_messages" not in st.session_state:
        st.session_state["analytics_chat_messages"] = []
    messages: list[dict[str, str]] = st.session_state["analytics_chat_messages"]

    # Header controls with status indicator
    with st.container(border=True):
        head_col1, head_col2 = st.columns([4, 1], vertical_alignment="center")
        with head_col1:
            st.markdown(
                f"**:material/auto_awesome: AI Intelligence Chat** · "
                f":green-badge[Live Context] · Provider: **{provider_name}** (`{provider.model}`)"
            )
        with head_col2:
            if st.button("Clear chat", icon=":material/delete_sweep:", key="analytics-chat-clear", width="stretch"):
                st.session_state["analytics_chat_messages"] = []
                st.toast("Conversation cleared", icon=":material/check_circle:")
                st.rerun()

    # Suggestion pills if conversation is empty
    if not messages:
        st.caption("✨ **Quick questions you can ask right now:**")
        selected_pill = st.pills(
            "Quick questions",
            options=list(SUGGESTIONS.keys()),
            label_visibility="collapsed",
            key="analytics_suggestion_pills",
        )
        if selected_pill:
            selected_prompt = SUGGESTIONS[selected_pill]
            st.session_state["analytics_chat_messages"].append({"role": "user", "content": selected_prompt})
            st.rerun()

    # Render conversation history (relying on default avatars for accurate data-testid hooks)
    for message in messages:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])

    # If the last message was a user message waiting for an assistant response (e.g. from pill click)
    if messages and messages[-1]["role"] == "user":
        current_context = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        provider_messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCurrent analytics JSON:\n{current_context}"}]
        provider_messages.extend(messages[-8:])
        with st.chat_message("assistant"):
            try:
                response = st.write_stream(provider.stream_chat(provider_messages))
            except Exception as exc:
                response = "I couldn't reach the analytics provider. Check your Groq API key, model setting, and network connection, then try again."
                st.error(response)
                st.caption(f"Provider error: {type(exc).__name__}")
        if response:
            messages.append({"role": "assistant", "content": str(response)})
            st.rerun()

    # Input bar
    prompt = st.chat_input("Ask about brand market share, garment combinations, face metrics…", key="analytics-chat-input")
    if not prompt:
        return

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    current_context = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    provider_messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCurrent analytics JSON:\n{current_context}"}]
    provider_messages.extend(messages[-8:])
    with st.chat_message("assistant"):
        try:
            response = st.write_stream(provider.stream_chat(provider_messages))
        except Exception as exc:
            response = "I couldn't reach the analytics provider. Check your Groq API key, model setting, and network connection, then try again."
            st.error(response)
            st.caption(f"Provider error: {type(exc).__name__}")
    if response:
        messages.append({"role": "assistant", "content": str(response)})
