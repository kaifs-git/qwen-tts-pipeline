"""
Baseline reference: stock Qwen3-TTS-0.6B-CustomVoice end-to-end (no megakernel).

Purpose (Phase C, step C9.1):
  1. Confirm the model + vocoder actually produce audio on the GPU box.
  2. Surface the exact runtime API (supported speakers, generate, vocoder decode).
  3. Establish a baseline RTF/time to compare the megakernel-backed path against.

This uses the STOCK talker (torch), NOT our kernel. It's the correctness +
perf reference. The megakernel path (pipeline/talker_backend_megakernel.py)
must match this audio and beat (or match) this speed.

Run on the vast.ai box (model already downloaded):
    cd /workspace/qwen-tts-pipeline
    python scripts/baseline_tts.py
Writes baseline.wav, prints supported speakers + timing + RTF.
"""

import sys
import time

import torch

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TEXT = "Real time voice agents live and die by latency. This is a baseline test."
LANGUAGE = "English"


def main():
    from qwen_tts import Qwen3TTSModel

    print(f"loading {MODEL} ...")
    # No flash-attn on the box → fall back to eager (sdpa). attn_implementation
    # left default so it doesn't hard-require flash_attention_2.
    tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda", dtype=torch.bfloat16)

    speakers = tts.get_supported_speakers()
    print(f"supported speakers: {speakers}")
    if not speakers:
        print("ERROR: no speakers reported — cannot run CustomVoice baseline")
        sys.exit(1)
    speaker = list(speakers)[0]
    print(f"using speaker: {speaker}")

    # Warmup (first run JITs + caches).
    print("warmup ...")
    tts.generate_custom_voice(text="warm up.", speaker=speaker, language=LANGUAGE)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wavs, sr = tts.generate_custom_voice(text=TEXT, speaker=speaker, language=LANGUAGE)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    wav = wavs[0]
    audio_s = len(wav) / sr
    rtf = dt / audio_s if audio_s > 0 else float("nan")

    print(f"\n--- baseline ---")
    print(f"  sample_rate = {sr}")
    print(f"  audio_len   = {audio_s:.2f} s ({len(wav)} samples)")
    print(f"  synth_time  = {dt*1000:.0f} ms")
    print(f"  RTF         = {rtf:.3f}   (target < 0.15)")

    try:
        import soundfile as sf
        sf.write("baseline.wav", wav, sr)
        print("  wrote baseline.wav")
    except ImportError:
        # Fallback: raw PCM via numpy if soundfile missing.
        import numpy as np
        pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2")
        pcm.tofile("baseline.pcm")
        print(f"  soundfile missing — wrote baseline.pcm (raw s16le @ {sr}Hz)")


if __name__ == "__main__":
    main()
