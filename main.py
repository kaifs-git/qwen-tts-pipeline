import asyncio
import os
import sys
import time

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from pipeline.stt import GroqSTT
from pipeline.llm import GroqLLM
from pipeline.tts_local import EdgeTTS

load_dotenv()

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 6


def record_audio() -> np.ndarray:
    print(f"  [Recording {RECORD_SECONDS}s — speak now]", flush=True)
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    return audio.flatten()


async def run_pipeline():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    stt = GroqSTT(api_key)
    llm = GroqLLM(api_key)
    tts = EdgeTTS()

    print("\n========================================")
    print("  Voice Pipeline  |  Local Mode (edge-tts)")
    print("  STT: Groq Whisper | LLM: llama-3.3-70b")
    print("========================================")
    print("  Press Enter to speak, Ctrl+C to quit\n")

    while True:
        try:
            input("[ Press Enter to speak ]")

            # --- STT ---
            audio = record_audio()
            t0 = time.perf_counter()
            text = stt.transcribe(audio)
            stt_ms = (time.perf_counter() - t0) * 1000

            if not text:
                print("  (nothing transcribed, try again)\n")
                continue

            print(f"\n  You [{stt_ms:.0f}ms STT]: {text}")

            # --- LLM ---
            print("  Assistant: ", end="", flush=True)
            t1 = time.perf_counter()
            full_response = ""
            for token in llm.chat(text):
                print(token, end="", flush=True)
                full_response += token
            llm_ms = (time.perf_counter() - t1) * 1000
            print(f"  [{llm_ms:.0f}ms LLM]\n")

            # --- TTS ---
            t2 = time.perf_counter()
            await tts.speak(full_response)
            tts_ms = (time.perf_counter() - t2) * 1000
            print(f"  [TTS done in {tts_ms:.0f}ms]\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    asyncio.run(run_pipeline())
