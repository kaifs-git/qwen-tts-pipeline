"""
Pipecat voice agent — local mode.

Pipeline:
    LocalAudioInput (mic + Silero VAD)
      → Groq STT (whisper-large-v3-turbo)
      → context aggregator (user)
      → Groq LLM (llama-3.3-70b)
      → EdgeTTSService (placeholder; will swap to MegakernelTTSService on vast.ai)
      → LocalAudioOutput (speaker)
      → context aggregator (assistant)

Run:
    venv/bin/python server.py

Speak after the startup line. Pipecat handles VAD turn-taking — no Enter-to-talk.
Ctrl+C to quit.
"""

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
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransportParams,
)

from pipeline.tts_edge import EdgeTTSService

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep every response concise and conversational — 2-3 sentences max. "
    "No bullet points or markdown, plain natural speech only."
)


async def main():
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
    tts = EdgeTTSService(voice="en-US-AriaNeural")

    context = OpenAILLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = llm.create_context_aggregator(context)

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
    print("  Voice Pipeline | Pipecat + Groq + edge-tts")
    print("  STT: whisper-large-v3-turbo | LLM: llama-3.3-70b")
    print("  TTS: edge-tts (placeholder for megakernel)")
    print("========================================")
    print("  Speak whenever you want — Silero VAD handles turn-taking.")
    print("  Ctrl+C to quit.\n")

    runner = PipelineRunner(handle_sigint=True)
    try:
        await runner.run(task)
    finally:
        py_audio.terminate()


if __name__ == "__main__":
    asyncio.run(main())
