from __future__ import annotations

import io
import logging
from typing import Optional

import speech_recognition as sr
from gtts import gTTS


LOGGER = logging.getLogger(__name__)


def transcribe_audio(audio_bytes: bytes, language: str = "en-US") -> str:
    """Convert spoken audio bytes into text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:
        LOGGER.warning("Speech recognition request failed: %s", exc)
        return ""


def synthesize_speech(text: str, language: str = "en") -> Optional[bytes]:
    """Turn text into spoken audio (MP3) using gTTS."""
    if not text.strip():
        return None

    try:
        tts = gTTS(text=text, lang=language)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as exc:  # pragma: no cover - depends on external service
        LOGGER.warning("Text-to-speech synthesis failed: %s", exc)
        return None

