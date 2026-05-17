import asyncio
import io
import os
import tempfile

import edge_tts
import pygame


class EdgeTTS:
    """
    Local TTS using Microsoft Edge neural voices — no model download, no GPU.
    Swap this class with the megakernel TTS client for prod.
    """

    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice
        pygame.mixer.pre_init(frequency=24000, size=-16, channels=1, buffer=512)
        pygame.mixer.init()

    async def speak(self, text: str):
        if not text.strip():
            return

        communicate = edge_tts.Communicate(text, self.voice)

        # Collect audio bytes from the stream
        mp3_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_bytes += chunk["data"]

        if not mp3_bytes:
            return

        # Write to temp file and play via pygame (SDL handles mp3 natively)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(mp3_bytes)
            tmp_path = f.name

        try:
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.05)
        finally:
            pygame.mixer.music.unload()
            os.unlink(tmp_path)
