# Qwen-TTS Pipecat Voice Pipeline

Take-home project: wire AlpinDale's `qwen_megakernel` (Qwen3-0.6B CUDA decode kernel, ~1000 tok/s on RTX 5090) as the LLM decode backend for **Qwen3-TTS's talker decoder**, streaming audio into a [Pipecat](https://docs.pipecat.ai) voice pipeline.

Targets: **TTFC < 60ms**, **RTF < 0.15**, streaming frame-by-frame to Pipecat (no full-utterance buffering).

---

## Status

| Phase | What | Where |
|-------|------|-------|
| A1 | Pipecat skeleton: Groq STT + Groq LLM + edge-tts TTSService + LocalAudioTransport + Silero VAD + 7-stage smoke test | ✅ `server.py`, `pipeline/tts_edge.py`, `tests/smoke.py` |
| A2 | `MegakernelTTSService` + `TalkerBackend` protocol + `MockTalkerBackend` + `--tts` CLI flag | ✅ `pipeline/tts_megakernel.py`, `pipeline/talker_backend.py` |
| A3 | Benchmark harness — TTFC / synth_ms / RTF, per backend, p50+p95, JSON output, target-pass marks | ✅ `bench/perf.py` |
| B1.1 | Talker kernel fork: VOCAB 151936→3072 (only change — talker = Qwen3-0.6B with codec LM head; verified vs model config.json) | ✅ `kernels/talker_kernel/csrc/kernel.cu`, `talker_megakernel/model.py` |
| B1.2 | Weight loader for Qwen3-TTS talker `state_dict` — auto-detects `talker.model.*` vs `model.*` prefix; pulls embed from `codec_embedding`, lm_head from `codec_head` (NOT tied) | ✅ `talker_megakernel/model.py::load_weights` |
| B1.3 | Embed-bypass: `Decoder.step_embed(inputs_embeds[HIDDEN_SIZE]) -> codec_id` — pass single-row scratch as `embed_weight`, token_id=0. Zero CUDA edit. | ✅ `talker_megakernel/model.py::Decoder.step_embed` |
| B1.4 | Prefill API: `Decoder.prefill(seq[N, HIDDEN_SIZE])` loops `step_embed` to seed KV from text-encoder hiddens. | ✅ `talker_megakernel/model.py::Decoder.prefill` |
| B1.5 | Talker bench — three modes (`throughput` / `prefill` / `correctness`). Throughput uses synthetic random weights (kernel-only perf, no HF needed). Correctness compares kernel `last_hidden_state` vs stock HF `talker.forward(inputs_embeds=...)` cosine sim — quantifies 1D-RoPE vs 3D-mrope drift on real talker weights. | ✅ `kernels/talker_kernel/talker_megakernel/bench.py` |
| B1.6 | `MegakernelTalkerBackend` scaffold — TalkerBackend-protocol class wired into `server.py --tts megakernel` (lazy GPU import). Generate loop + CodePredictor inner step + embed compose implemented; `_build_prefix` + `_vocoder_decode` marked TODO for vast.ai validation. | ✅ scaffold `pipeline/talker_backend_megakernel.py` |
| C  | vast.ai RTX 5090 bring-up: kernel builds on CUDA 13.1 (1141 tok/s), correctness cosine 0.83–0.89 vs HF, kernel runs in-loop end-to-end. Known limitation: 1D-RoPE drift → talker never emits EOS (see Performance numbers). | ✅ measured; `docs/vastai_runbook.md`, `scripts/kernel_generate.py` |

Local box has no GPU — Phase A and Python-side Phase B work happen here, kernel build + Qwen3-TTS run on vast.ai.

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

## Kernel modifications (Phase B)

> Per take-home spec: *"If the talker decoder's backbone is a different size than 0.6B, document what you changed in the kernel and why."*

Forked AlpinDale's `qwen_megakernel` → `kernels/talker_kernel/`. Upstream targets **Qwen3-0.6B**. The Qwen3-TTS-0.6B talker decoder (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`) is **architecturally identical** to Qwen3-0.6B — verified against its `config.json`. The only difference: its LM head emits codec tokens, not text. So exactly **one constant changes**; CUDA kernel logic untouched.

### Constants (kernel.cu + talker_megakernel/model.py)

| Constant | Qwen3-0.6B (upstream) | Qwen3-TTS-0.6B talker | Changed? |
|---|---|---|---|
| `VOCAB_SIZE` (`LDG_VOCAB_SIZE`) | 151936 | **3072** | ✅ LM head emits codec tokens, not text |
| `NUM_LAYERS` | 28 | 28 | — |
| `NUM_KV_HEADS` | 8 | 8 | — (KV_SIZE 1024) |
| `INTERMEDIATE_SIZE` | 3072 | 3072 | — |
| `HIDDEN_SIZE` | 1024 | 1024 | — |
| `NUM_Q_HEADS` | 16 | 16 | — |
| `HEAD_DIM` | 128 | 128 | — |
| `MAX_SEQ_LEN` | 2048 | 2048 | — |

The `VOCAB_SIZE` drop (151936 → 3072) is also a perf win — the LM head matmul dominated per-step latency in the original kernel. Because `HIDDEN_SIZE` and all attention/MLP dims are unchanged, the kernel keeps upstream's exact shared-memory layout and perf tuning — no retuning needed. (We initially mis-sized the talker from config-class defaults; corrected after reading the actual model `config.json` on the GPU box.)

### Python-side adaptations (`talker_megakernel/model.py`)

1. **Weight loader rewrite.** Pulls from `talker.model.layers.{i}.*` keys (full Qwen3-TTS checkpoint) or `model.layers.{i}.*` (standalone talker), auto-detected via `_detect_prefix`. Embed table comes from `codec_embedding.weight`. LM head from `codec_head.weight` (NOT tied to embedding — distinct from upstream 0.6B which ties them).
2. **Embed-bypass — `Decoder.step_embed(inputs_embeds)`.** Talker never sees a token at its input; its input is a composed embedding (`sum(codec_embeds) + trailing_text_hidden[step] + tts_pad_embed`). To inject arbitrary `inputs_embeds` *without* modifying CUDA, the loader allocates a single-row scratch buffer; `step_embed` copies the incoming hidden into row 0 of scratch and calls the kernel with `embed_weight=scratch, token_id=0`. Kernel reads `scratch + 0 * HIDDEN_SIZE` → arbitrary embedding injected, no recompile.
3. **Prefill — `Decoder.prefill(seq)`.** Loops `step_embed` over a seq of text-encoder hiddens to seed the KV cache. Upstream kernel has no batched-prefill op; adding one is out of scope per the spec's *"integration, not research"* directive. N×kernel-launch overhead is logged for the per-stage breakdown.
4. **Hidden-state read.** `Decoder.last_hidden_state` returns the bf16 post-final-norm hidden buffer for the CodePredictor 15-step sub-codebook loop (CodePredictor stays in PyTorch — separate decoder, separate concern, not the megakernel target per spec).

### What was deliberately NOT touched

- **3D multimodal RoPE (mrope).** Talker uses `apply_multimodal_rotary_pos_emb` with `mrope_section` interleaved across head_dim. Kernel keeps 1D RoPE. Correctness impact will be measured against the HF reference on vast.ai (Phase C) and reported in the perf table — rewriting RoPE is *research*, not *integration*.
- **CodePredictor.** 5-layer sub-codebook decoder; runs in stock PyTorch.
- **Vocoder.** Stock Qwen3-TTS vocoder.

### Build

```bash
cd kernels/talker_kernel
pip install -r requirements.txt
python -c "import talker_megakernel"   # JIT-compiles the CUDA extension on first import
```

CUDA 12.8+ and sm_120 (Blackwell / RTX 5090) required. Will not build on older arch. Verified building + running on CUDA 13.1 / driver 590 / Python 3.12.

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
- `git` — for cloning upstream repos (`kernels/qwen_megakernel`, `kernels/Qwen3-TTS`)

**No ffmpeg needed** — mp3 decode handled in-process by `miniaudio` (libmpg123 under the hood, statically bundled).

### vast.ai-only system deps (Phase C)

```bash
# RTX 5090 / sm_120 / Blackwell
sudo apt install -y nvidia-driver-555 cuda-toolkit-12-8
# CUDA 12.8+ — kernel will not compile on older (tested on 13.1)
```

> **Full vast.ai bring-up is in `docs/vastai_runbook.md`** — instance picker, HF_HOME setup, kernel build, bench commands, correctness check, perf table population, demo recording. Follow that runbook on the rented GPU box.

### Python deps

```bash
python3.10 -m venv venv
venv/bin/pip install -r requirements.txt
```

### Env vars

```bash
cp .env.example .env
# add: GROQ_API_KEY=gsk_...
# Phase C also: HUGGINGFACE_HUB_TOKEN=hf_...   (gated: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
# Phase C also: HF_HOME=/workspace/hf_cache    (vast.ai disk layout — see runbook)
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
| `megakernel`      | `MegakernelTTSService` + `MegakernelTalkerBackend` (real Qwen3-TTS via talker megakernel) | **GPU only — requires CUDA 12.8+ / sm_120** |

- Silero VAD handles turn-taking — just speak, no Enter-to-talk
- Talking over the assistant interrupts it (Pipecat `allow_interruptions=True`)
- Ctrl+C to quit

---

## Benchmark TTS backends

```bash
venv/bin/python -m bench.perf                              # all backends (edge + mock)
venv/bin/python -m bench.perf --backends mock-megakernel   # one backend
venv/bin/python -m bench.perf --runs 10 --warmup 3
venv/bin/python -m bench.perf --no-pace                    # mock: raw throughput (pace_rtf=0)
venv/bin/python -m bench.perf --output bench/results/run1.json
```

Measures, per backend × per prompt (short / medium / long):

| metric | what it means | target |
|--------|---------------|--------|
| **TTFC** p50/p95 | Time-To-First-Chunk: first `TTSAudioRawFrame` after `run_tts` call | < 60ms |
| **synth_ms** p50 | total wall-time from `run_tts` to last frame | informational |
| **audio_s** mean | PCM duration produced | informational |
| **RTF** p50/p95 | `synth_s / audio_s` — real-time factor | < 0.15 |

Table marks ✓/✗ against the spec targets. Today the marks are mostly ✗ because edge-tts is a cloud round-trip and mock pacing is set to RTF=0.15 (intentionally at the target — mock is a placeholder, not a winner). The marks turn ✓ in Phase C with the real GPU backend.

JSON output (`--output`) holds every run, suitable for diffing across runs / commits.

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
├── bench/
│   └── perf.py            # TTFC / synth / RTF harness — runs against any TTSService
├── docs/
│   └── takehome_project.docx   # assignment spec
├── kernels/
│   ├── talker_kernel/     # OUR fork — Qwen3-TTS talker megakernel (TRACKED)
│   │   ├── csrc/kernel.cu
│   │   ├── csrc/torch_bindings.cpp
│   │   └── talker_megakernel/  # model.py, build.py, bench.py
│   ├── qwen_megakernel/   # AlpinDale's Qwen3-0.6B megakernel (gitignored, re-clone per Setup)
│   └── Qwen3-TTS/         # Qwen3-TTS HF reference (gitignored, re-clone per Setup)
├── requirements.txt
├── .env.example
└── README.md
```

`kernels/qwen_megakernel/` and `kernels/Qwen3-TTS/` are upstream clones — gitignored to keep this repo's diff focused on our work. Clone them after first checkout:

```bash
git clone https://github.com/AlpinDale/qwen_megakernel kernels/qwen_megakernel
git clone https://huggingface.co/Qwen/Qwen3-TTS    kernels/Qwen3-TTS  # gated, HF token needed
```

---

## Performance numbers

Measured on **vast.ai RTX 5090 (sm_120), CUDA 13.1, driver 590, Python 3.12**, bf16, single utterance, speaker `aiden`, language English. Per the spec: real numbers, methodology, and the bottleneck stated honestly — including where the integration falls short.

### Kernel microbenchmarks (`talker_megakernel.bench`, synthetic weights)

| Metric | Result | Note |
|--------|--------|------|
| Upstream `qwen_megakernel` sanity (Qwen3-0.6B) | **1029 tok/s** | builds + runs on CUDA 13.1 unmodified |
| Talker kernel throughput | **1141 tok/s** (0.876 ms/step) | beats upstream — smaller LM head (vocab 3072 vs 151936) |
| Talker prefill (32 tokens) | **25.8 ms** (0.805 ms/step) | N×step_embed, no batched-prefill op |
| Correctness vs HF `talker.forward` (cosine, 64 steps) | min **0.830** / p50 **0.878** / mean **0.890** | systematic fp32-vs-bf16 + 1D-RoPE drift |

### End-to-end, real model (`scripts/`)

| Run | Audio | Synth | RTF | EOS |
|-----|-------|-------|-----|-----|
| Stock baseline (`baseline_tts.py`) | 6.24 s | 6735 ms | **1.079** | clean ✓ |
| Kernel-in-loop (`kernel_generate.py`) | 40.88 s* | 33705 ms | 0.824* | **never fired** ✗ |

\* **Not a valid RTF.** The kernel-driven talker hit the 512-token cap without emitting EOS — see the known limitation below. The number is recorded only to show the kernel *does* run inside the stock generate loop and accelerates per-step decode; it is not a real-time-factor claim.

Methodology: `bench/perf.py` (Phase A3) for the Pipecat-path metrics; `talker_megakernel.bench` for raw kernel; `scripts/{baseline_tts,profile_stages,kernel_generate}.py` for stock vs kernel-in-loop. Same harness local (mock) and on GPU (real) — no methodology drift.

### Why we are off the < 0.15 RTF target — honest bottleneck

Two reasons, both measured, neither hidden:

1. **CodePredictor dominates, and it is out of scope.** Per-stage profiling (`profile_stages.py`) of stock generate: CodePredictor **≈62%**, talker decode ≈25%, vocoder <1%. The spec scopes the work to *the talker decoder, not the codebook generator*. Even with the talker decode made free, projected RTF only falls to **~1.18** — still ~8× over target. The talker megakernel is the right target per spec, but it is not the dominant cost. Real-time would require kernelizing CodePredictor too (a Qwen3-shape 5-layer decoder — same technique applies; flagged as next step, not done here).

2. **1D-RoPE drift breaks coherence (the known limitation, below).** Because the kernel-driven talker never emits EOS, its end-to-end run over-generates and the RTF number is meaningless.

### Known limitation — 1D RoPE vs 3D mrope drift → no EOS

**What.** The talker uses Qwen2.5-VL-style **3D multimodal RoPE** (`apply_multimodal_rotary_pos_emb`, `mrope_section`). The kernel keeps **1D RoPE** — a deliberate integration-scope choice (rewriting RoPE in CUDA is *research*, not *integration*, per the spec).

**Measured impact.** Cosine similarity kernel-vs-HF talker hidden = **min 0.830 / mean 0.890** (fp32-vs-bf16 contributes part of this; the spread is tight and systematic, not a growing logic error). Audibly: the kernel produces **real speech-like signal** (kernel.wav rms 0.061, peak 0.72 — not silence, not noise), but the drift shifts the codec-token argmax enough that the **EOS token (2150) never wins**. Generation runs away to the `max_new_tokens` cap (512 steps → 40.88 s of babble for a ~6 s sentence) instead of self-terminating like stock (74 steps → 6.24 s, clean EOS).

**Conclusion.** The integration is sound — kernel weight-load, embed-bypass, prefill, hidden-state hand-off, and the in-loop monkeypatch all work end-to-end, and per-step decode is faster than upstream. The remaining gap is **numerical fidelity**: closing it requires implementing true 3D mrope in `kernel.cu` (extend the rotary stage to 3× cos/sin tables applied per `mrope_section`). That is the documented next step; it was scoped out as research per the brief.

### C9 integration bugs found + fixed (kernel-in-loop bring-up)

The monkeypatch path (`scripts/kernel_generate.py`) surfaced three real integration bugs, all fixed:
1. **Strided embed view → GPU deadlock.** `inputs_embeds[0, i]` is a non-contiguous row; `step_embed` assumes a contiguous `[HIDDEN_SIZE]` buffer → out-of-bounds read → cooperative-kernel hang that ignores SIGINT (needs instance reboot). Fix: `.contiguous().to(bf16)` before every kernel call.
2. **Missing `hidden_states` → `'NoneType' not subscriptable`.** Stock `talker.forward` wraps the inner-model output as `hidden_states=(outputs.hidden_states, codec_ids)`, and `generate()` reads `hid[0][-1][:, -1:]` (`modeling_qwen3_tts.py:2281`). The patched inner forward must return `hidden_states=(out,)`, not just `last_hidden_state`.
3. **Runaway generation.** No EOS (see limitation) → added a `max_new_tokens` cap as a guard so the run terminates and is debuggable.

### Pipecat integration — what is proven, what is not (honest scope)

| Layer | Proven by | Status |
|-------|-----------|--------|
| Pipecat pipeline (STT → LLM → TTS → transport, VAD, interruptions) | `tests/smoke.py` (7 stages), edge-tts end-to-end | ✅ |
| `MegakernelTTSService` + `TalkerBackend` protocol (frame format, sample-rate/channel validation, streaming `AsyncIterator`) | `MockTalkerBackend` streams `TTSAudioRawFrame`s through the real service; `bench/perf.py` measures TTFC/RTF | ✅ |
| Talker megakernel produces audio | `scripts/kernel_generate.py` — kernel swapped into stock `generate` loop, writes `kernel.wav` | ✅ (GPU) |
| **Real megakernel *through* the Pipecat service** | — | ❌ **not wired** |

**The honest gap.** The GPU proof drives the kernel via stock `generate` (monkeypatch), which **bypasses** the Pipecat backend. The Pipecat path itself — `pipeline/talker_backend_megakernel.py::MegakernelTalkerBackend.stream_pcm` — still has two `NotImplementedError` stubs (`_build_prefix`, `_vocoder_decode`) that reproduce stock `generate`'s prefix construction and vocoder call. So `server.py --tts megakernel` does not run yet.

**Why it was not finished.** It is gated by the drift limitation above: the kernel-driven talker never emits EOS, so even a fully wired Pipecat backend would stream the same incoherent, non-terminating babble. Filling the stubs proves only that babble flows frame-by-frame — no additional integration signal — so the effort was deferred behind fixing 3D-mrope. The pipeline's streaming contract is already proven with the mock backend, which is the part Pipecat actually owns. To finish it later: fill the two stubs (the logic is exactly what `kernel_generate.py` already exercises), add a `megakernel` branch to `bench/perf.py::build_service`, and run `--tts megakernel`.

---

## Known issues (Phase A1)

- **edge-tts audio occasionally choppy.** Root cause: `edge_tts.Communicate.stream()` only yields complete mp3 segments, and we wait for the whole mp3 before decoding (miniaudio needs aligned frames). So all PCM chunks land back-to-back in pipecat's output ring buffer — too-small chunks underrun the audio device. Mitigation: bumped frame size to 60ms. **Real fix lands with Phase B/C** — the megakernel TTS streams PCM frame-by-frame from the talker decoder, no mp3 round-trip.
- **TTFC dominated by edge-tts buffering** (~1.8s in smoke test). This is the placeholder backend, not the target. Megakernel target is <60ms.
- **No interruption barge-in fully tested** under noisy mic. Pipecat has `allow_interruptions=True` enabled; Silero VAD threshold may need tuning.

## Notes

- `kernels/qwen_megakernel/` and `kernels/Qwen3-TTS/` are upstream clones, gitignored, not git submodules — re-clone per Setup. Keeps the tracked diff focused on `kernels/talker_kernel/` (our fork).
- Pipecat 0.0.108. API may drift — pin if upgrading.
- bfloat16 only. No quantization (per assignment spec).
- mp3 decode via `miniaudio` (bundles libmpg123) — no ffmpeg system dep.
