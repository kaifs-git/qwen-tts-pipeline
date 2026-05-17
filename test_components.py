"""
Quick sanity test for each pipeline component.
Run this before main.py to confirm keys and audio work.
Usage: venv/bin/python test_components.py
"""
import asyncio
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()


def check_env():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("FAIL  GROQ_API_KEY not set — create a .env file with it")
        sys.exit(1)
    print(f"OK    GROQ_API_KEY loaded ({key[:8]}...)")
    return key


def test_llm(api_key: str):
    print("\n--- Testing Groq LLM ---")
    from pipeline.llm import GroqLLM
    llm = GroqLLM(api_key)
    t = time.perf_counter()
    response = ""
    for token in llm.chat("Say exactly: hello from Groq LLM"):
        response += token
    ms = (time.perf_counter() - t) * 1000
    print(f"OK    Response ({ms:.0f}ms): {response.strip()}")


async def test_tts():
    print("\n--- Testing edge-tts ---")
    from pipeline.tts_local import EdgeTTS
    tts = EdgeTTS()
    t = time.perf_counter()
    print("      Playing test audio (you should hear: 'Voice pipeline is ready')")
    await tts.speak("Voice pipeline is ready.")
    ms = (time.perf_counter() - t) * 1000
    print(f"OK    TTS done ({ms:.0f}ms)")


def test_sounddevice():
    print("\n--- Testing sounddevice (microphone) ---")
    import sounddevice as sd
    import numpy as np
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        print(f"OK    Default input device: {devices[default_in]['name']}")
        print("      Recording 1s silence test...", end="", flush=True)
        audio = sd.rec(16000, samplerate=16000, channels=1, dtype="int16")
        sd.wait()
        print(f" peak={np.abs(audio).max()} — mic is {'working' if np.abs(audio).max() > 10 else 'silent (check mic)'}")
    except Exception as e:
        print(f"WARN  Sounddevice issue: {e}")


async def main():
    print("=== Pipeline Component Tests ===\n")
    api_key = check_env()
    test_llm(api_key)
    await test_tts()
    test_sounddevice()
    print("\n=== All tests done. Run main.py to start the voice pipeline. ===")


if __name__ == "__main__":
    asyncio.run(main())
