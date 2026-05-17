# Qwen-TTS Pipecat Voice Pipeline

Take-home project: wire AlpinDale's `qwen_megakernel` (Qwen3-0.6B CUDA decode kernel, ~1000 tok/s on RTX 5090) as the LLM decode backend for **Qwen3-TTS's talker decoder**, streaming audio into a [Pipecat](https://docs.pipecat.ai) voice pipeline.

Targets: **TTFC < 60ms**, **RTF < 0.15**, streaming frame-by-frame to Pipecat (no full-utterance buffering).

---

## Status

| Phase | What | Where |
|-------|------|-------|
| A1 | Pipecat skeleton: Groq STT + Groq LLM + edge-tts TTSService + LocalAudioTransport + Silero VAD + 7-stage smoke test | ✅ `server.py`, `pipeline/tts_edge.py`, `tests/smoke.py` |
| A2 | `MegakernelTTSService` + `TalkerBackend` protocol + `MockTalkerBackend` + `--tts` CLI flag | ✅ `pipeline/tts_megakernel.py`, `pipeline/talker_backend.py` |
| A3 | Benchmark harness (TTFC / RTF / tok/s) | ⏳ `bench/` |
| B  | Talker kernel fork: NUM_LAYERS 28→20, NUM_KV_HEADS 8→2, VOCAB 151936→3072, prefill API | ⏳ `kernels/talker_kernel/` |
| C  | vast.ai RTX 5090 bring-up + real megakernel TTS + bench + demo | ⏳ |

Local box has no GPU — Phase A and Python-side Phase B work happen here, kernel build + real Qwen3-TTS run on vast.ai.

---

## Architecture

```
mic → LocalAudioInput → Silero VAD → GroqSTT (whisper-large-v3-turbo)
    → context aggregator → GroqLLM (llama-3.3-70b)
    → TTSService → LocalAudioOutput → speaker
                ↑
                ├── EdgeTTSService               (--tts edge)
                └── MegakernelTTSService         (--tts mock-megakernel today,
                        ↑                         real on vast.ai)
                        TalkerBackend protocol
                        ├── MockTalkerBackend       (sine wave, paced at RTF=0.15)
                        └── MegakernelTalkerBackend (Phase C — text encoder
                            → adapted megakernel talker → CodePredictor → vocoder)
```

`TTSService` interface is identical for both backends — `run_tts(text, context_id) -> AsyncGenerator[Frame, None]` yielding `TTSStartedFrame → TTSAudioRawFrame(24kHz mono s16le) → TTSStoppedFrame`.

`MegakernelTTSService` delegates to whatever `TalkerBackend` you hand it. The real GPU backend (Phase C) drops in without touching pipeline wiring, smoke tests, or `server.py`.

---

## Setup

### System deps (Ubuntu 22.04 / Debian)

```bash
sudo apt update
sudo apt install -y \
    python3.10 python3.10-venv python3-dev \
    portaudio19-dev libportaudio2 \
    build-essential git
```

- `portaudio19-dev` + `libportaudio2` — required by PyAudio (build + runtime)
- `python3-dev` + `build-essential` — required by PyAudio + miniaudio C extensions
- `git` — for submodule clones (`kernels/qwen_megakernel`, `kernels/Qwen3-TTS`)

**No ffmpeg needed** — mp3 decode handled in-process by `miniaudio` (libmpg123 under the hood, statically bundled).

### vast.ai-only system deps (Phase C)

```bash
# RTX 5090 / sm_120 / Blackwell
sudo apt install -y nvidia-driver-555 cuda-toolkit-12-8
# CUDA 12.8 minimum — kernel will not compile on older
```

### Python deps

```bash
python3.10 -m venv venv
venv/bin/pip install -r requirements.txt
```

### Env vars

```bash
cp .env.example .env
# add: GROQ_API_KEY=gsk_...
# Phase C also: HUGGINGFACE_TOKEN=hf_...   (Qwen/Qwen3-TTS is gated)
```

---

## Smoke test (run this FIRST)

```bash
venv/bin/python -m tests.smoke
```

Runs 8 checks before you ever touch the live pipeline:

| # | Check | What it proves |
|---|-------|----------------|
| 1 | `GROQ_API_KEY` loaded from `.env` | env wired |
| 2 | All Python imports resolve (pipecat, groq, edge-tts, miniaudio, pyaudio) | deps installed |
| 3 | PyAudio finds default mic + speaker | system audio works |
| 4 | Groq STT round-trips a synthesized clip | `whisper-large-v3-turbo` reachable |
| 5 | Groq LLM returns one token | `llama-3.3-70b-versatile` reachable |
| 6 | `EdgeTTSService.run_tts` yields PCM frames + reports TTFC | edge TTS path works end-to-end |
| 7 | Full Pipecat `Pipeline([...])` graph builds (no mic, no run) | wiring is valid |
| 8 | `MegakernelTTSService` + `MockTalkerBackend` yields PCM frames | mock backend works; same path real GPU backend will take |

All green → run the live agent. Any red → fix before `server.py`.

Exit code = number of failures (good for CI).

---

## Run the voice agent

```bash
venv/bin/python server.py                          # default: --tts edge
venv/bin/python server.py --tts mock-megakernel    # 440Hz sine, paced at RTF=0.15
```

| `--tts` | What | When to use |
|---------|------|-------------|
| `edge`            | Microsoft Edge cloud TTS (placeholder, real-sounding voice) | Default; demo the pipeline end-to-end |
| `mock-megakernel` | `MegakernelTTSService` + `MockTalkerBackend` (sine wave) | Verify pipeline shape; proves swap is one flag away on vast.ai |

- Silero VAD handles turn-taking — just speak, no Enter-to-talk
- Talking over the assistant interrupts it (Pipecat `allow_interruptions=True`)
- Ctrl+C to quit

---

## Repo layout

```
.
├── server.py              # Pipecat voice agent entry point
├── pipeline/              # custom Pipecat services
│   ├── __init__.py
│   ├── tts_edge.py        # EdgeTTSService (edge-tts placeholder)
│   ├── tts_megakernel.py  # MegakernelTTSService (backend-agnostic wrapper)
│   └── talker_backend.py  # TalkerBackend protocol + MockTalkerBackend
├── tests/
│   └── smoke.py           # pre-flight checks (7 stages, see above)
├── bench/                 # perf harness (Phase A3) — TTFC / RTF / tok/s
├── docs/
│   └── takehome_project.docx   # assignment spec
├── kernels/               # upstream clones (untouched)
│   ├── qwen_megakernel/   # AlpinDale's Qwen3-0.6B CUDA megakernel
│   └── Qwen3-TTS/         # Qwen3-TTS reference HF model
├── requirements.txt
├── .env.example
└── README.md
```

Phase B kernel fork will live at `kernels/talker_kernel/` to keep the upstream clone diff-clean.

---

## Performance numbers

Filled in after Phase C. Will report:
- Megakernel decode tok/s (upstream + adapted talker variant)
- TTFC end-to-end
- RTF
- Per-stage latency breakdown (STT / LLM / talker / codepredictor / vocoder)
- Where bottlenecks sit

---

## Known issues (Phase A1)

- **edge-tts audio occasionally choppy.** Root cause: `edge_tts.Communicate.stream()` only yields complete mp3 segments, and we wait for the whole mp3 before decoding (miniaudio needs aligned frames). So all PCM chunks land back-to-back in pipecat's output ring buffer — too-small chunks underrun the audio device. Mitigation: bumped frame size to 60ms. **Real fix lands with Phase B/C** — the megakernel TTS streams PCM frame-by-frame from the talker decoder, no mp3 round-trip.
- **TTFC dominated by edge-tts buffering** (~1.8s in smoke test). This is the placeholder backend, not the target. Megakernel target is <60ms.
- **No interruption barge-in fully tested** under noisy mic. Pipecat has `allow_interruptions=True` enabled; Silero VAD threshold may need tuning.

## Notes

- `kernels/qwen_megakernel` and `kernels/Qwen3-TTS` are upstream clones, not git submodules — kept loose so kernel fork (Phase B) can live alongside as `kernels/talker_kernel/` without subtree drama.
- Pipecat 0.0.108. API may drift — pin if upgrading.
- bfloat16 only. No quantization (per assignment spec).
- mp3 decode via `miniaudio` (bundles libmpg123) — no ffmpeg system dep.
