"""
Per-stage profiler for stock Qwen3-TTS-0.6B generate (Phase C, step C9.2).

Wraps the three hot callables with cumulative CUDA-synced timers:
  - talker decode   : Qwen3TTSTalkerModel.forward   (28-layer backbone — the
                      part our megakernel REPLACES)
  - code_predictor  : Qwen3TTSTalkerCodePredictorModel.forward (5-layer sub-
                      codebook decoder — stays in torch per spec)
  - vocoder         : speech_tokenizer.decode

Tells us whether accelerating the talker (our kernel) actually moves RTF, or
whether the CodePredictor / vocoder dominate. This is the bottleneck analysis
the take-home asks for ("show us where the bottlenecks are").

Run on the box:
    cd /workspace/qwen-tts-pipeline
    python scripts/profile_stages.py
"""

import time

import torch

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TEXT = "Real time voice agents live and die by latency. This is a baseline test."
LANGUAGE = "English"


class Timer:
    def __init__(self):
        self.total = 0.0
        self.calls = 0

    def wrap(self, fn):
        def inner(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            self.total += time.perf_counter() - t0
            self.calls += 1
            return out
        return inner


def main():
    from qwen_tts import Qwen3TTSModel

    print(f"loading {MODEL} ...")
    tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda", dtype=torch.bfloat16)
    m = tts.model
    speaker = list(tts.get_supported_speakers())[0]

    talker_t = Timer()
    codepred_t = Timer()
    vocoder_t = Timer()

    # Patch the three stages with timers.
    m.talker.model.forward = talker_t.wrap(m.talker.model.forward)
    m.talker.code_predictor.model.forward = codepred_t.wrap(m.talker.code_predictor.model.forward)
    m.speech_tokenizer.decode = vocoder_t.wrap(m.speech_tokenizer.decode)

    # Warmup.
    tts.generate_custom_voice(text="warm up.", speaker=speaker, language=LANGUAGE)
    talker_t.total = codepred_t.total = vocoder_t.total = 0.0
    talker_t.calls = codepred_t.calls = vocoder_t.calls = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wavs, sr = tts.generate_custom_voice(text=TEXT, speaker=speaker, language=LANGUAGE)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    audio_s = len(wavs[0]) / sr
    other = total - talker_t.total - codepred_t.total - vocoder_t.total

    print(f"\n--- per-stage breakdown (audio {audio_s:.2f}s, total {total*1000:.0f}ms, RTF {total/audio_s:.3f}) ---")
    def row(name, t, calls):
        print(f"  {name:<16} {t.total*1000:8.0f} ms  {100*t.total/total:5.1f}%   ({t.calls} calls)" if hasattr(t,'total') else "")
    print(f"  {'talker decode':<16} {talker_t.total*1000:8.0f} ms  {100*talker_t.total/total:5.1f}%   ({talker_t.calls} calls)  ← megakernel replaces this")
    print(f"  {'code_predictor':<16} {codepred_t.total*1000:8.0f} ms  {100*codepred_t.total/total:5.1f}%   ({codepred_t.calls} calls)  (stays torch)")
    print(f"  {'vocoder':<16} {vocoder_t.total*1000:8.0f} ms  {100*vocoder_t.total/total:5.1f}%   ({vocoder_t.calls} calls)")
    print(f"  {'other (prefix/etc)':<16} {other*1000:8.0f} ms  {100*other/total:5.1f}%")
    print(f"\nIf megakernel makes talker-decode ~free (~{talker_t.calls}x1ms), projected RTF ≈ "
          f"{(total - talker_t.total + talker_t.calls*0.001)/audio_s:.3f}")


if __name__ == "__main__":
    main()
