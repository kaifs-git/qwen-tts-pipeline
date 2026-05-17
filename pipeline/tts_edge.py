"""
Edge-TTS as a Pipecat TTSService.

Yields TTSStartedFrame → TTSAudioRawFrame chunks (24kHz mono s16le) → TTSStoppedFrame.
Streams mp3 chunks from edge_tts, decodes incrementally with miniaudio, emits raw PCM
frames so the pipeline can push audio to the speaker without buffering the full utterance.

This is the local stand-in for the eventual megakernel TTS service. The class signature
(run_tts AsyncGenerator[Frame, None]) is what we will reuse for MegakernelTTSService.
"""

from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import edge_tts
import miniaudio
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1


@dataclass
class EdgeTTSSettings(TTSSettings):
    """edge-tts doesn't expose model or language as runtime knobs — only voice."""

    pass


class EdgeTTSService(TTSService):
    """Microsoft Edge neural TTS over the edge-tts streaming endpoint."""

    Settings = EdgeTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        voice: str = "en-US-AriaNeural",
        sample_rate: Optional[int] = TARGET_SAMPLE_RATE,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        default_settings = self.Settings(
            model=None,
            voice=voice,
            language=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_stop_frames=True,
            push_start_frame=True,
            settings=default_settings,
            **kwargs,
        )
        self._voice = default_settings.voice

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            return

        logger.debug(f"EdgeTTS run_tts ctx={context_id} text={text!r}")
        await self.start_ttfb_metrics()

        try:
            communicate = edge_tts.Communicate(text, self._voice)

            mp3_buf = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buf.extend(chunk["data"])

            if not mp3_buf:
                logger.warning("EdgeTTS got no audio")
                yield ErrorFrame("edge-tts returned no audio")
                return

            await self.stop_ttfb_metrics()

            decoded = miniaudio.decode(
                bytes(mp3_buf),
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=TARGET_CHANNELS,
                sample_rate=TARGET_SAMPLE_RATE,
            )
            pcm: bytes = decoded.samples.tobytes()

            # 60ms chunks — small enough for low latency, large enough that the
            # output transport doesn't underrun between writes. 20ms was choppy
            # because edge-tts collects the full mp3 before we emit anything,
            # so all 126 frames hit pipecat back-to-back; tiny frames stress
            # the output device's ring buffer. Megakernel TTS (Phase B) will
            # stream PCM natively and can go back to 20ms.
            chunk_ms = 60
            bytes_per_sample = 2
            chunk_size = (
                TARGET_SAMPLE_RATE * chunk_ms // 1000
            ) * bytes_per_sample * TARGET_CHANNELS

            for i in range(0, len(pcm), chunk_size):
                yield TTSAudioRawFrame(
                    audio=pcm[i : i + chunk_size],
                    sample_rate=TARGET_SAMPLE_RATE,
                    num_channels=TARGET_CHANNELS,
                )

        except Exception as e:
            logger.exception(f"EdgeTTS error: {e}")
            yield ErrorFrame(f"edge-tts: {e}")
