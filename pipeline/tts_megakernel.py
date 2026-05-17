"""
MegakernelTTSService — Pipecat TTSService whose audio comes from a TalkerBackend.

Same shape as EdgeTTSService (run_tts AsyncGenerator yielding TTSAudioRawFrames).
The whole point: this class will not change when we swap MockTalkerBackend for
the real GPU-backed MegakernelTalkerBackend on vast.ai. Pipecat wiring, frame
format, settings shape, sample rate — all pinned here, backend is the only
moving part.
"""

from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from pipeline.talker_backend import SAMPLE_RATE, CHANNELS, TalkerBackend


@dataclass
class MegakernelTTSSettings(TTSSettings):
    """Megakernel TTS has no per-call model/voice/lang knobs — voice is baked into the model."""

    pass


class MegakernelTTSService(TTSService):
    """Wraps any TalkerBackend (mock or real) as a Pipecat TTSService."""

    Settings = MegakernelTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        backend: TalkerBackend,
        sample_rate: Optional[int] = SAMPLE_RATE,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        default_settings = self.Settings(
            model=None,
            voice=None,
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
        self._backend = backend

        # Sanity: backend must match the sample rate we report to pipecat.
        if backend.sample_rate != sample_rate:
            raise ValueError(
                f"backend sample_rate {backend.sample_rate} != service sample_rate {sample_rate}"
            )
        if backend.channels != CHANNELS:
            raise ValueError(
                f"backend channels {backend.channels} != service channels {CHANNELS}"
            )

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            return

        logger.debug(
            f"MegakernelTTS run_tts ctx={context_id} backend={type(self._backend).__name__} text={text!r}"
        )
        await self.start_ttfb_metrics()

        first = True
        try:
            async for pcm in self._backend.stream_pcm(text):
                if not pcm:
                    continue
                if first:
                    await self.stop_ttfb_metrics()
                    first = False
                yield TTSAudioRawFrame(
                    audio=pcm,
                    sample_rate=self._backend.sample_rate,
                    num_channels=self._backend.channels,
                )
        except Exception as e:
            logger.exception(f"MegakernelTTS backend error: {e}")
            yield ErrorFrame(f"megakernel backend: {e}")
