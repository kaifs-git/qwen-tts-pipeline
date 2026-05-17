"""
Smoke tests — run before server.py to catch config / dep / audio issues.

Tests, in order:
    1. Env: GROQ_API_KEY present
    2. Imports: pipecat, groq, edge-tts, miniaudio, pyaudio
    3. Audio devices: PyAudio finds default input + output
    4. Groq STT: round-trip a synthesized "hello world" clip
    5. Groq LLM: one-token completion via pipecat service
    6. EdgeTTSService: synth + decode + count PCM frames
    7. Pipeline wiring: build the full Pipeline graph (no run, no mic)

All steps log PASS / FAIL with timing. Exit code = number of failures.

Run:
    venv/bin/python -m tests.smoke
"""

import asyncio
import os
import sys
import time
import wave
from io import BytesIO
from pathlib import Path

# Allow `python tests/smoke.py` as well as `python -m tests.smoke`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

PASS, FAIL = "\033[32m PASS \033[0m", "\033[31m FAIL \033[0m"
failures = 0


def step(name):
    print(f"\n[{name}]")


def ok(msg, ms=None):
    suffix = f" ({ms:.0f}ms)" if ms is not None else ""
    print(f"  {PASS} {msg}{suffix}")


def fail(msg, err=None):
    global failures
    failures += 1
    print(f"  {FAIL} {msg}")
    if err:
        print(f"         {err}")


# ---------- 1. env ----------
def test_env():
    step("1/7  env vars")
    key = os.getenv("GROQ_API_KEY")
    if not key:
        fail("GROQ_API_KEY not set — add to .env")
        return None
    ok(f"GROQ_API_KEY loaded ({key[:8]}…)")
    return key


# ---------- 2. imports ----------
def test_imports():
    step("2/7  imports")
    mods = [
        "pipecat",
        "pipecat.services.groq.stt",
        "pipecat.services.groq.llm",
        "pipecat.transports.local.audio",
        "pipecat.audio.vad.silero",
        "edge_tts",
        "miniaudio",
        "pyaudio",
        "pipeline.tts_edge",
    ]
    for m in mods:
        try:
            __import__(m)
            ok(m)
        except Exception as e:
            fail(m, e)


# ---------- 3. audio devices ----------
def _silence_alsa_stderr():
    """ALSA probes every backend in /etc/asound.conf (rear/center/USB/OSS) and
    spams stderr with 'unable to open slave' warnings even when default device
    is fine. Redirect fd 2 to /dev/null around PyAudio init to suppress."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull, 2)
    return saved_stderr, devnull


def _restore_stderr(saved_stderr, devnull):
    os.dup2(saved_stderr, 2)
    os.close(devnull)
    os.close(saved_stderr)


def test_audio_devices():
    step("3/7  audio devices (PyAudio)")
    try:
        import pyaudio

        saved, devnull = _silence_alsa_stderr()
        try:
            pa = pyaudio.PyAudio()
        finally:
            _restore_stderr(saved, devnull)
        try:
            in_info = pa.get_default_input_device_info()
            out_info = pa.get_default_output_device_info()
            ok(f"input:  {in_info['name']}  ({int(in_info['defaultSampleRate'])}Hz)")
            ok(f"output: {out_info['name']} ({int(out_info['defaultSampleRate'])}Hz)")
        finally:
            pa.terminate()
    except Exception as e:
        fail("PyAudio device query", e)


# ---------- 4. Groq STT ----------
def _gen_silence_wav(seconds=1.0, rate=16000) -> bytes:
    import numpy as np

    pcm = (np.random.randint(-200, 200, int(rate * seconds), dtype="int16")).tobytes()
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def test_groq_stt(api_key):
    step("4/7  Groq STT round-trip")
    if not api_key:
        fail("skipped — no api key")
        return
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        wav = _gen_silence_wav()
        t = time.perf_counter()
        client.audio.transcriptions.create(
            file=("test.wav", wav, "audio/wav"),
            model="whisper-large-v3-turbo",
            response_format="text",
        )
        ok("whisper-large-v3-turbo responded", (time.perf_counter() - t) * 1000)
    except Exception as e:
        fail("Groq STT call", e)


# ---------- 5. Groq LLM ----------
def test_groq_llm(api_key):
    step("5/7  Groq LLM completion")
    if not api_key:
        fail("skipped — no api key")
        return
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        t = time.perf_counter()
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say only: ok"}],
            max_tokens=4,
        )
        text = r.choices[0].message.content.strip()
        ok(f"llama-3.3-70b → {text!r}", (time.perf_counter() - t) * 1000)
    except Exception as e:
        fail("Groq LLM call", e)


# ---------- 6. EdgeTTSService ----------
async def test_edge_tts():
    step("6/7  EdgeTTSService (synth + decode)")
    try:
        from pipeline.tts_edge import EdgeTTSService

        tts = EdgeTTSService(voice="en-US-AriaNeural")
        t = time.perf_counter()
        n, total = 0, 0
        ttfc_ms = None
        async for f in tts.run_tts("Smoke test successful.", "smoke-1"):
            if type(f).__name__ == "TTSAudioRawFrame":
                if ttfc_ms is None:
                    ttfc_ms = (time.perf_counter() - t) * 1000
                n += 1
                total += len(f.audio)
        secs = total / (24000 * 2)
        ok(f"chunks={n}  audio={secs:.2f}s  TTFC≈{ttfc_ms:.0f}ms")
        if n == 0:
            fail("no PCM frames produced")
    except Exception as e:
        fail("EdgeTTSService.run_tts", e)


# ---------- 7. Pipeline wiring ----------
def test_pipeline_build(api_key):
    step("7/7  Pipecat pipeline build (no run)")
    if not api_key:
        fail("skipped — no api key")
        return
    try:
        import pyaudio

        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
        from pipecat.services.groq.llm import GroqLLMService
        from pipecat.services.groq.stt import GroqSTTService
        from pipecat.transports.local.audio import (
            LocalAudioInputTransport,
            LocalAudioOutputTransport,
            LocalAudioTransportParams,
        )

        from pipeline.tts_edge import EdgeTTSService

        saved, devnull = _silence_alsa_stderr()
        try:
            pa = pyaudio.PyAudio()
        finally:
            _restore_stderr(saved, devnull)
        try:
            params = LocalAudioTransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=24000,
                vad_analyzer=SileroVADAnalyzer(),
            )
            tin = LocalAudioInputTransport(pa, params)
            tout = LocalAudioOutputTransport(pa, params)
            stt = GroqSTTService(api_key=api_key)
            llm = GroqLLMService(api_key=api_key, model="llama-3.3-70b-versatile")
            tts = EdgeTTSService()
            ctx = OpenAILLMContext(messages=[{"role": "system", "content": "x"}])
            agg = llm.create_context_aggregator(ctx)
            Pipeline([tin, stt, agg.user(), llm, tts, tout, agg.assistant()])
            ok("pipeline graph builds (mic + speaker + Silero + STT + LLM + TTS)")
        finally:
            pa.terminate()
    except Exception as e:
        fail("Pipeline build", e)


async def main():
    print("=" * 60)
    print(" Smoke tests — Qwen-TTS Pipecat pipeline")
    print("=" * 60)

    key = test_env()
    test_imports()
    test_audio_devices()
    test_groq_stt(key)
    test_groq_llm(key)
    await test_edge_tts()
    test_pipeline_build(key)

    print("\n" + "=" * 60)
    if failures == 0:
        print(f" {PASS} all checks green — run `venv/bin/python server.py` next")
        sys.exit(0)
    else:
        print(f" {FAIL} {failures} failure(s) — fix above before running server.py")
        sys.exit(failures)


if __name__ == "__main__":
    asyncio.run(main())
