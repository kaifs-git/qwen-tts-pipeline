# vast.ai bring-up runbook — Phase C

Step-by-step for taking the local Phase-B repo to a working RTX 5090 demo. Every step has a verify command and an expected output. If a step fails, fix it before moving on — the next step depends on it.

> **Time budget**: ~3-4 hrs of GPU rental if nothing surprises us. Burn rate ~$1/hr on a 5090.

---

## C0. Pick the instance

Vast.ai filters:
- **GPU**: RTX 5090 (sm_120 / Blackwell) — kernel is hardware-locked, no fallback
- **CUDA**: 12.8 or newer (kernel uses cooperative launch + bf16 features)
- **Disk**: ≥ 80 GB (Qwen3-TTS weights ~3.5 GB + qwen_megakernel build cache + python venv)
- **RAM**: ≥ 32 GB
- **Image**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` or `nvidia/cuda:12.8.0-devel-ubuntu22.04` — needs `nvcc` for JIT build

Verify:
```bash
nvidia-smi                                           # expect RTX 5090, driver ≥ 555
nvcc --version                                       # expect Cuda compilation tools, release 12.8+
python3 --version                                    # expect 3.10+
```

If `nvcc` is missing (runtime-only image), install dev toolchain:
```bash
apt update && apt install -y cuda-toolkit-12-8
```

---

## C1. Clone + system deps

```bash
cd /workspace
git clone https://github.com/kaifs-git/qwen-tts-pipeline
cd qwen-tts-pipeline

# Mic/speaker libs (smoke test still imports pipecat audio bits even on the GPU box).
apt update
apt install -y portaudio19-dev libportaudio2 python3-dev build-essential git
```

---

## C2. Python deps

```bash
python3.10 -m venv venv
venv/bin/pip install -U pip wheel
venv/bin/pip install -r requirements.txt

# Qwen3-TTS package (model class + processor + tokenizer)
venv/bin/pip install -U qwen-tts
```

Verify:
```bash
venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True  NVIDIA GeForce RTX 5090
venv/bin/python -c "from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration; print('ok')"
# expect: ok
```

---

## C3. Upstream kernel clones + HF auth

**Important:** point HF cache at the big `/workspace` partition before downloading anything. Vast.ai's root `/` is usually only ~20 GB; Qwen3-TTS (~3.5 GB) + Qwen3-0.6B (~1.2 GB) + pip + torch caches will fill it and crash mid-build.

```bash
# 1. Move HF cache to the rented disk. Set ONCE per shell session;
#    transformers / huggingface_hub / our load_weights / MegakernelTalkerBackend
#    all read this env var.
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"

# 2. Persist for new shells (optional but useful if you reconnect).
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

# 3. Upstream kernel source (gitignored in our repo; pulled fresh here).
git clone https://github.com/AlpinDale/qwen_megakernel kernels/qwen_megakernel

# 4. Qwen/Qwen3-TTS is gated. Need HF token with terms accepted in browser first.
export HUGGINGFACE_HUB_TOKEN=hf_...
huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN"

# 5. Pre-cache (~3.5 GB) — saves 5 min on first model load.
huggingface-cli download Qwen/Qwen3-TTS
```

Verify cache location + gated access:
```bash
# Cache landed where expected
du -sh "$HF_HOME/hub/models--Qwen--Qwen3-TTS"      # expect ~3.5G
df -h /workspace                                   # confirm room left

# Auth + gating works
venv/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Qwen/Qwen3-TTS', 'config.json')
print('gated access OK')
"
```

> Why HF_HOME and not `cache_dir=` in code: env-var is library-wide. Setting it once covers `transformers.from_pretrained` (our loader + the backend) AND the upstream `qwen_megakernel` bench (step C4, which pulls 0.6B weights). No per-call plumbing.

---

## C4. Validate upstream megakernel (sanity baseline)

Confirm the unmodified upstream kernel compiles + hits its claimed perf BEFORE touching our fork. If this fails, the instance is at fault, not our code.

```bash
cd kernels/qwen_megakernel
../../venv/bin/pip install -r requirements.txt
../../venv/bin/python -m qwen_megakernel.bench
```

Expected:
- First run: JIT-compiles the kernel (~30s, lots of nvcc output)
- ~1000 tok/s on Qwen3-0.6B
- Correctness check: HF tokens == MK tokens for `"Hello"` prompt

If tok/s is way under 1000, instance is GPU-throttled or sharing — pick a different machine.

```bash
cd ../..
```

---

## C5. Build our talker kernel (constants-swapped fork)

```bash
cd kernels/talker_kernel
../../venv/bin/pip install -r requirements.txt
# First import triggers JIT build with new constants (NUM_LAYERS=20, NUM_KV_HEADS=2, etc.)
../../venv/bin/python -c "import talker_megakernel; print('built')"
```

Expected: `built`. nvcc may warn about new dims; warnings are fine, errors aren't.

If build fails: check `kernel.cu` constants block matches `model.py` constants — both must be `NUM_LAYERS=20, NUM_KV_HEADS=2, INTERMEDIATE_SIZE=2048, LDG_VOCAB_SIZE=3072` exactly.

```bash
cd ../..
```

---

## C6. Bench the talker kernel — throughput mode

Synthetic random weights, isolates raw kernel perf. No HF model needed.

`talker_megakernel` is not a pip-installed package — `cd` into the kernel dir so Python resolves the module from cwd:

```bash
cd kernels/talker_kernel
../../venv/bin/python -m talker_megakernel.bench --mode throughput --prefill 32 --decode 256 --runs 5
cd ../..
```

Expected (rough): talker dims are smaller than 0.6B (fewer layers, smaller KV, much smaller LM head). Should beat upstream's ~1000 tok/s. Record number for README perf table.

If tok/s is below 500 — kernel is mis-launched or a constant is wrong. Bisect against upstream perf.

---

## C7. Prefill latency

Isolates the TTFC-dominating cost (prefill before first codec emission).

```bash
cd kernels/talker_kernel
../../venv/bin/python -m talker_megakernel.bench --mode prefill --prefill 32 --runs 10
cd ../..
```

Expected: roughly `32 × ms_per_step` from C6. TTFC budget = prefill_time + 1 step + CodePredictor 31 steps + vocoder.

If prefill is >50 ms for 32 prompt tokens, this is the dominant TTFC slug — flag in README perf section.

---

## C8. Correctness — kernel vs HF talker

Quantifies 1D-RoPE-vs-3D-mrope drift. Loads real Qwen3-TTS weights. This is the integration-correctness gate documented in `README.md` "Known bottlenecks anticipated".

```bash
cd kernels/talker_kernel
../../venv/bin/python -m talker_megakernel.bench --mode correctness --prefill 32 --decode 64
cd ../..
```

Interpretation (from bench output):
- **min sim > 0.99** → 1D RoPE is good enough; ship as-is, mrope is academic.
- **min sim 0.90 – 0.99** → drift exists; record number, run audio quality check anyway (next step).
- **min sim < 0.90** → kernel output diverges materially. Document the gap in the README perf table. Decide whether to invest in mrope (out of "integration" scope — flag to interviewer if you do).

Save the output:
```bash
cd kernels/talker_kernel
../../venv/bin/python -m talker_megakernel.bench --mode correctness > ../../bench/results/correctness.txt
cd ../..
```

---

## C9. Wire the real backend — fill the two `NotImplementedError` blocks

`pipeline/talker_backend_megakernel.py` has two TODOs that require running the model:

1. **`_build_prefix(text)`** — diff against stock `Qwen3TTSForConditionalGeneration.generate(non_streaming_mode=False)` for a single utterance with `language="english"`, no speaker. Transcribe the prefix construction (text_projection + tts_bos/eos/pad + codec_bos/pad + codec_think/nothink tokens) from `modeling_qwen3_tts.py` lines ~2068-2220.

2. **`_vocoder_decode(primary_codec_id, sub_codes)`** — confirm the variant in the loaded checkpoint: v1 (25 Hz, `tokenizer_25hz/`) or v2 (12 Hz, `tokenizer_12hz/`). Call its streaming decode signature, convert float32 waveform → s16le bytes:
   ```python
   import numpy as np
   pcm_f32 = self._speech_tokenizer.decode(codes)   # API name TBD per variant
   pcm_s16 = (pcm_f32.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy().tobytes()
   ```

After filling: smoke-test the backend in isolation:
```bash
venv/bin/python -c "
import asyncio
from pipeline.talker_backend_megakernel import MegakernelTalkerBackend

async def main():
    b = MegakernelTalkerBackend()
    chunks = 0
    total_bytes = 0
    async for pcm in b.stream_pcm('Hello world, this is a test.'):
        chunks += 1
        total_bytes += len(pcm)
    print(f'chunks={chunks}  bytes={total_bytes}  audio={total_bytes/(24000*2):.2f}s')

asyncio.run(main())
"
```

Expected: chunks > 0, audio > 0.5s for a 5-word prompt.

---

## C10. End-to-end via Pipecat

```bash
venv/bin/python server.py --tts megakernel
```

Speak into the (vast.ai instance) mic. Verify:
- STT transcribes
- LLM responds
- TTS via megakernel — audio comes out, not silence, not noise
- No frame drops / underruns in pipecat logs

Vast.ai instances usually lack a real mic. Two workarounds:
- Pipe a pre-recorded WAV through the input transport (modify `server.py` to use a file source instead of `LocalAudioInputTransport`)
- Skip end-to-end mic and rely on the per-stage bench numbers (C11)

---

## C11. Run the canonical bench harness

Same `bench/perf.py` we built in Phase A3, now pointed at the real backend. Yields the assignment-grade TTFC + RTF numbers.

First, register `megakernel` in the bench's backend list:

```bash
# bench/perf.py:: DEFAULT_BACKENDS — add "megakernel"
# bench/perf.py:: build_service — add the megakernel branch
```

Then:
```bash
venv/bin/python -m bench.perf --backends megakernel --runs 5 --warmup 2 \
    --output bench/results/c11_megakernel.json
```

Expected table marks: `TTFC p50 < 60 ms ✓`, `RTF p50 < 0.15 ✓` (if not, see C12).

---

## C12. Per-stage breakdown (where the time goes)

If TTFC misses target, instrument the backend to log per-stage latency. Drop the following into `MegakernelTalkerBackend.stream_pcm`:

```python
import time
t = {}
t['t0'] = time.perf_counter()
prefix, trailing, pad = self._build_prefix(text)
t['prefix'] = time.perf_counter()
self._kernel.prefill(prefix[:-1])
t['prefill'] = time.perf_counter()
# first step
primary = self._kernel.step_embed(prefix[-1])
t['first_step'] = time.perf_counter()
sub_codes = self._run_code_predictor(self._kernel.last_hidden_state, primary)
t['codepred'] = time.perf_counter()
pcm = self._vocoder_decode(primary, sub_codes)
t['vocoder'] = time.perf_counter()
print(f"prefix={t['prefix']-t['t0']:.3f} prefill={t['prefill']-t['prefix']:.3f} "
      f"step={t['first_step']-t['prefill']:.3f} cp={t['codepred']-t['first_step']:.3f} "
      f"voc={t['vocoder']-t['codepred']:.3f}")
yield pcm
# ...
```

Likely-culprits in priority order (per `MEMORY.md::project-architecture-research` + `project-talker-arch-findings`):
1. **Vocoder first-chunk** — probably 30-100 ms; if so, that's TTFC. Megakernel speed irrelevant past this point.
2. **CodePredictor 31-step eager loop** — single-batch eager Python — probably 10-30 ms per talker step. Becomes RTF bottleneck.
3. **Prefill N×step_embed loop** — kernel-launch overhead × N. ~10 ms for N=32 at sub-ms step.
4. **Talker megakernel step** — should be sub-millisecond (our optimization target).

---

## C13. Demo recording

Doc deliverable: "Demo recording showing the voice agent working with you talking end to end."

Tools:
- OBS Studio (capture screen + mic + speaker — vast.ai instance won't have OBS; record from your laptop while SSHing into vast.ai for the agent run, but you need the audio on your machine — see below)
- Easier: dump conversation transcripts + a single WAV of one round-trip TTS output and call it a demo if mic loop is non-trivial on vast.ai

Bare minimum demo content the doc asks for:
- Show the agent running (`server.py --tts megakernel`)
- Speak a prompt
- Agent replies via megakernel TTS
- Show bench output (TTFC / RTF / tok/s on screen)

Save the recording somewhere (don't commit binaries to the repo — link via README).

---

## C14. Finalize README

Fill the perf table in `README.md::Performance numbers`:
- Megakernel tok/s (from C6)
- TTFC p50 + p95 (from C11 JSON)
- RTF p50 + p95 (from C11 JSON)
- End-to-end mic→speaker latency (estimate or measure)
- Per-stage breakdown (from C12)

Also document any deviations from the doc's reference numbers — per the spec: *"if you're way off, explain why."* The likely deviation candidate is RTF if vocoder dominates.

Commit + push:
```bash
git add README.md bench/results/
git commit -m "docs: Phase C perf numbers + correctness measurement"
git push origin master
```

---

## Tear-down

When done:
```bash
# Save anything you want off the instance first — vast.ai wipes on shutdown
mkdir -p ~/vastai_archive
cp -r bench/results ~/vastai_archive/
scp -r ~/vastai_archive your-laptop:~/vastai_archive
```

Then destroy the instance from the vast.ai console.

---

## Quick-fail checklist (skip ahead if something obvious breaks)

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvcc: command not found` | Runtime-only image | `apt install cuda-toolkit-12-8` |
| Kernel build: `unrecognized architecture sm_120` | nvcc too old | Need CUDA 12.8+, upgrade toolkit |
| `OSError: Qwen/Qwen3-TTS is gated` | No HF token | `huggingface-cli login`, accept terms on hf.co/Qwen/Qwen3-TTS |
| `import qwen_tts` fails | Package not installed | `pip install -U qwen-tts` |
| `OSError: [Errno 28] No space left on device` mid-download | HF cache on small `/` partition | `export HF_HOME=/workspace/hf_cache` then re-download (see C3) |
| `from_pretrained` re-downloads every run | `HF_HOME` not set in current shell | Re-export, or persist via `~/.bashrc` |
| Kernel runs but tok/s < 200 | Sharing GPU / thermal throttle / wrong instance | Check `nvidia-smi`; pick a different vast.ai node |
| Correctness sim < 0.5 | Weight loader prefix wrong or constants drifted | Diff `_detect_prefix` output keys vs actual state_dict keys |
| `stream_pcm` yields no chunks | `_build_prefix` or `_vocoder_decode` still `NotImplementedError` | Fill in C9 |
| Audio is noise | mrope drift OR wrong codec_head OR vocoder decode mis-shaped | C8 correctness number tells you which |
