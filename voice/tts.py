"""
voice/tts.py
────────────
Text-to-Speech using gTTS (Google Text-to-Speech).

HOW IT WORKS:
    1. We take the synthesis summary from Node 3 of the LangGraph
    2. gTTS sends the text to Google's TTS service and returns MP3 audio
    3. We save it to a temp file
    4. app.py passes the file path to st.audio(autoplay=True)
    5. The browser plays the audio automatically when the result appears

WHY gTTS AND NOT pyttsx3?
    pyttsx3 uses the OS text-to-speech engine (robotic, flat voice).
    gTTS uses Google's TTS (natural, clear voice — same as Google Translate).
    The tradeoff: gTTS requires internet. pyttsx3 works fully offline.
    For a demo, gTTS sounds dramatically better.

ALTERNATIVE (premium):
    ElevenLabs has a free tier with very natural voices.
    You could swap this module out without changing app.py at all —
    that's why we isolate voice logic in its own module.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional


def synthesize_speech(text: str) -> Optional[str]:
    """
    Converts text to an MP3 file using gTTS.

    Args:
        text: The summary sentence from the agent (1-2 sentences max).
              Longer text = longer audio = worse UX for a voice assistant.

    Returns:
        Path to the temporary MP3 file, or None if synthesis failed.
        The caller (app.py) passes this path to st.audio().

    NOTE: The caller is responsible for cleaning up the temp file after
          st.audio() has loaded it. Or just let atexit handle it on shutdown.
    """
    from gtts import gTTS

    if not text or not text.strip():
        return None

    tmp_path = None
    try:
        # Create the temp file path first (don't open it yet — gTTS opens it)
        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3",
            prefix="bi_agent_tts_",
        )
        tmp.close()  # gTTS will write to it by path
        tmp_path = tmp.name

        # gTTS makes a network call to Google's TTS API
        # lang="en" and slow=False gives natural speech speed
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(tmp_path)

        return tmp_path

    except Exception as e:
        print(f"[TTS] Speech synthesis failed: {e}")
        # Clean up if we created a file but failed to write to it
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return None
