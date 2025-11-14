"""UI utility functions."""

import base64
from typing import Optional
import streamlit as st

QUESTION_AUDIO_MIME = "audio/mp3"


def render_audio_clip(
    audio_bytes: Optional[bytes],
    mime: str = QUESTION_AUDIO_MIME,
    autoplay: bool = False
):
    """
    Render an audio clip in Streamlit.

    Args:
        audio_bytes: Audio data as bytes
        mime: MIME type
        autoplay: Whether to autoplay
    """
    if not audio_bytes:
        return
    if autoplay:
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        autoplay_tag = f"""
            <audio src="data:{mime};base64,{b64_audio}" autoplay controls style="width: 100%; margin-top: 0.5rem;">
                Your browser does not support the audio element.
            </audio>
        """
        st.markdown(autoplay_tag, unsafe_allow_html=True)
    else:
        st.audio(audio_bytes, format=mime)

