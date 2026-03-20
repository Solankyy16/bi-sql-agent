"""
voice/stt.py
────────────
Speech-to-Text using Groq's Whisper API.

HOW IT WORKS:
    1. User holds the mic button in Streamlit
    2. audio-recorder-streamlit captures audio and returns raw bytes
    3. We write those bytes to a temp .wav file
    4. We send the file to Groq's Whisper endpoint (same API key as the LLM)
    5. Whisper returns the transcribed text
    6. We return that text — from here, voice and text paths are IDENTICAL

WHY GROQ FOR WHISPER (not local Whisper)?
    Running Whisper locally takes 3-10 seconds per transcription.
    Groq's Whisper endpoint takes ~0.3 seconds.
    For a voice assistant, latency is everything.
    Both use the same free API key — zero extra cost.

WINDOWS NOTE:
    Windows doesn't allow reading a NamedTemporaryFile while it's open.
    We use delete=False + manual close + manual unlink to handle this.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional


def transcribe_audio(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribes audio bytes to text using Groq's Whisper API.

    Args:
        audio_bytes: Raw audio data from audio-recorder-streamlit.
                     Typically WAV format, but Whisper handles MP3/M4A too.

    Returns:
        Transcribed text string, or None if transcription failed.

    Example:
        audio_bytes = audio_recorder()  # from audio-recorder-streamlit
        if audio_bytes:
            question = transcribe_audio(audio_bytes)
    """
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Check your .env file.")

    client = Groq(api_key=api_key)

    # Write audio bytes to a temp file
    # We use delete=False for Windows compatibility (can't read open temp files)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
            prefix="bi_agent_audio_",
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        # File is closed here — safe to read on all platforms

        # Send to Groq Whisper
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",    # Groq's fastest Whisper model
                file=audio_file,
                response_format="text",      # Returns plain string, not JSON
                language="en",               # Hint speeds up transcription
            )

        # With response_format="text", the result IS the string
        return transcription.strip() if transcription else None

    except Exception as e:
        # Don't crash the app on transcription failure — return None
        # The caller should check for None and show an error in the UI
        print(f"[STT] Transcription failed: {e}")
        return None

    finally:
        # Always clean up the temp audio file
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except Exception:
                pass  # Best-effort cleanup — don't fail the whole function
