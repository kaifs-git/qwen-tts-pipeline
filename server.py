"""
Pipecat voice agent — local mode.

Pipeline:
    LocalAudioInput (mic + Silero VAD)
      → Groq STT (whisper-large-v3-turbo)
      → context aggregator (user)
      → Groq LLM (llama-3.3-70b)
      → TTS service (chosen via --tts)
      → LocalAudioOutput (speaker)
      → context aggregator (assistant)

TTS backends:
    --tts edge              edge-tts (default; placeholder, internet round-trip)
    --tts mock-megakernel   MegakernelTTSService + MockTalkerBackend
                            (sine wave, paced at RTF=0.15 — same interface the
                             real megakernel backend will satisfy on vast.ai)
    --tts megakernel        MegakernelTTSService + MegakernelTalkerBackend
                            (real Qwen3-TTS talker via adapted CUDA megakernel,
                             RTX 5090 / sm_120 required — Phase C)

Run:
    venv/bin/python server.py
    venv/bin/python server.py --tts mock-megakernel

Speak after the startup line. Pipecat handles VAD turn-taking — no Enter-to-talk.
Ctrl+C to quit.
"""

import argparse
import asyncio
import os
import sys

import pyaudio
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransportParams,
)

from pipeline.talker_backend import MockTalkerBackend
from pipeline.tts_edge import EdgeTTSService
from pipeline.tts_megakernel import MegakernelTTSService

load_dotenv()

TTS_BACKENDS = ("edge", "mock-megakernel", "megakernel")


def build_tts(name: str):
    if name == "edge":
        return EdgeTTSService(voice="en-US-AriaNeural")
    if name == "mock-megakernel":
        return MegakernelTTSService(backend=MockTalkerBackend())
    if name == "megakernel":
        # Lazy import — pulls in torch, transformers, qwen_tts, and the talker
        # megakernel CUDA build. Local box without GPU will error here, which
        # is intentional — keeps `--tts edge` / `--tts mock-megakernel` paths
        # GPU-free.
        from pipeline.talker_backend_megakernel import MegakernelTalkerBackend
        return MegakernelTTSService(backend=MegakernelTalkerBackend())
    raise ValueError(f"unknown --tts backend: {name}")

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep every response concise and conversational — 2-3 sentences max. "
    "No bullet points or markdown, plain natural speech only."
)


async def main(tts_backend: str):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not set — add it to .env")
        sys.exit(1)

    py_audio = pyaudio.PyAudio()
    params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=24000,
        vad_analyzer=SileroVADAnalyzer(),
    )
    transport_in = LocalAudioInputTransport(py_audio, params)
    transport_out = LocalAudioOutputTransport(py_audio, params)

    stt = GroqSTTService(api_key=api_key)
    llm = GroqLLMService(api_key=api_key, model="llama-3.3-70b-versatile")
    tts = build_tts(tts_backend)

    # Pipecat 1.2.x: universal LLMContext + LLMContextAggregatorPair replace the
    # removed OpenAILLMContext / llm.create_context_aggregator().
    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport_in,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport_out,
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    print("\n========================================")
    print("  Voice Pipeline | Pipecat + Groq")
    print("  STT: whisper-large-v3-turbo | LLM: llama-3.3-70b")
    print(f"  TTS: {tts_backend}  ({type(tts).__name__})")
    print("========================================")
    print("  Speak whenever you want — Silero VAD handles turn-taking.")
    print("  Ctrl+C to quit.\n")

    runner = PipelineRunner(handle_sigint=True)
    try:
        await runner.run(task)
    finally:
        py_audio.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--tts",
        choices=TTS_BACKENDS,
        default="edge",
        help="TTS backend (default: edge)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.tts))
