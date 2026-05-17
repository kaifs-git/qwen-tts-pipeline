"""
TalkerBackend protocol + mock implementation.

The real megakernel backend (Phase C) will satisfy the same protocol:
    text → AsyncIterator[bytes] of 24kHz mono s16le PCM frames.

`MegakernelTTSService` (pipeline/tts_megakernel.py) delegates to whatever
backend you hand it. This lets us prove the Pipecat wiring + frame format +
backpressure end-to-end on CPU before burning GPU dollars on vast.ai.

Phase C will add `MegakernelTalkerBackend` to this file (or a sibling) that
wraps the real Qwen3-TTS text-encoder → adapted-megakernel-talker
→ CodePredictor → vocoder graph. The TTSService and server.py stay unchanged.
"""

import asyncio
import math
import time
from typing import AsyncIterator, Protocol

SAMPLE_RATE = 24000
CHANNELS = 1
BYTES_PER_SAMPLE = 2

# Pacing knobs (mock backend)
DEFAULT_RTF = 0.15  # real-time factor: 1s of audio takes 150ms to generate
DEFAULT_WORDS_PER_SEC = 2.5  # ~150 wpm typical TTS speech rate
DEFAULT_CHUNK_MS = 20  # frame granularity into the pipeline


class TalkerBackend(Protocol):
    """Anything that turns text into a stream of PCM bytes."""

    sample_rate: int
    channels: int

    async def stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        """Yield 16-bit signed-LE PCM chunks. Caller decides chunk sizing semantics."""
        ...


class MockTalkerBackend:
    """
    Synthesizes a 440Hz sine wave sized to match the expected speech duration.

    Two reasons this exists:
      1) Proves the MegakernelTTSService → pipecat → speaker path works without GPU.
      2) Lets the A3 bench harness measure pipeline-side TTFC + RTF deterministically.

    Sleeps between chunks to simulate the kernel generating PCM at `pace_rtf`
    real-time-factor (default 0.15 = target spec). Set `pace_rtf=0` to disable
    pacing (max-throughput stress test).
    """

    sample_rate = SAMPLE_RATE
    channels = CHANNELS

    def __init__(
        self,
        *,
        words_per_sec: float = DEFAULT_WORDS_PER_SEC,
        pace_rtf: float = DEFAULT_RTF,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        tone_hz: float = 440.0,
        amplitude: float = 0.2,
    ):
        self._words_per_sec = words_per_sec
        self._pace_rtf = pace_rtf
        self._chunk_ms = chunk_ms
        self._tone_hz = tone_hz
        self._amplitude = amplitude

    def _estimate_duration_s(self, text: str) -> float:
        words = max(1, len(text.split()))
        return max(0.5, words / self._words_per_sec)

    async def stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        if not text.strip():
            return

        total_s = self._estimate_duration_s(text)
        chunk_samples = self.sample_rate * self._chunk_ms // 1000
        total_chunks = max(1, int(total_s * 1000 // self._chunk_ms))

        phase = 0.0
        phase_inc = 2.0 * math.pi * self._tone_hz / self.sample_rate
        chunk_wallclock_budget = (self._chunk_ms / 1000.0) * self._pace_rtf
        last_emit = time.perf_counter()

        for _ in range(total_chunks):
            buf = bytearray(chunk_samples * BYTES_PER_SAMPLE * self.channels)
            amp = int(self._amplitude * 32767)
            for i in range(chunk_samples):
                sample = int(math.sin(phase) * amp)
                phase += phase_inc
                buf[i * 2] = sample & 0xFF
                buf[i * 2 + 1] = (sample >> 8) & 0xFF

            yield bytes(buf)

            if self._pace_rtf > 0:
                elapsed = time.perf_counter() - last_emit
                sleep_for = chunk_wallclock_budget - elapsed
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                last_emit = time.perf_counter()
            else:
                await asyncio.sleep(0)  # cooperative yield
