#!/usr/bin/env bash
#
# One-shot bring-up for a fresh RTX 5090 / sm_120 box (vast.ai or similar).
# Idempotent: safe to re-run. Replaces the manual C0-C5 steps in
# docs/vastai_runbook.md.
#
# Usage:
#   export HUGGINGFACE_HUB_TOKEN=hf_xxx          # gated model — accept terms first at
#                                                # https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
#   bash scripts/setup_gpu.sh                    # if repo already cloned
#   # — or from scratch —
#   curl -fsSL https://raw.githubusercontent.com/kaifs-git/qwen-tts-pipeline/master/scripts/setup_gpu.sh | bash
#
# Env overrides: WORKDIR (default /workspace), HF_HOME (default $WORKDIR/hf_cache).
#
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_URL="https://github.com/kaifs-git/qwen-tts-pipeline"
MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
export HF_HOME="${HF_HOME:-$WORKDIR/hf_cache}"

step() { echo -e "\n\033[1;34m==> $*\033[0m"; }
warn() { echo -e "\033[1;33mWARN: $*\033[0m"; }

# --- 0. sanity: GPU + CUDA toolchain ---------------------------------------
step "GPU + CUDA toolchain"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || { echo "no GPU visible"; exit 1; }
if ! command -v nvcc >/dev/null; then
  warn "nvcc missing — installing cuda-toolkit (runtime-only image?)"
  apt-get update && apt-get install -y cuda-toolkit-12-8 || \
    { echo "install a CUDA dev toolkit (>=12.8) then re-run"; exit 1; }
fi
nvcc --version | tail -2

# --- 1. system deps --------------------------------------------------------
step "system packages"
apt-get update
apt-get install -y \
  portaudio19-dev libportaudio2 python3-dev python3-venv build-essential git \
  sox libsox-fmt-all ffmpeg

# --- 2. repo ---------------------------------------------------------------
step "repo @ $WORKDIR/qwen-tts-pipeline"
mkdir -p "$WORKDIR" && cd "$WORKDIR"
[ -d qwen-tts-pipeline/.git ] || git clone "$REPO_URL"
cd qwen-tts-pipeline

# --- 3. python venv + deps -------------------------------------------------
step "python venv + deps"
[ -d venv ] || python3 -m venv venv
venv/bin/pip install -U pip wheel
venv/bin/pip install -r requirements.txt
venv/bin/pip install -U qwen-tts

# --- 4. HF cache (on big disk) + auth --------------------------------------
step "HF cache + auth  (HF_HOME=$HF_HOME)"
mkdir -p "$HF_HOME"
grep -q "export HF_HOME=$HF_HOME" ~/.bashrc 2>/dev/null || \
  echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  venv/bin/huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN"
else
  warn "HUGGINGFACE_HUB_TOKEN not set — model is gated, download will fail."
  warn "  export HUGGINGFACE_HUB_TOKEN=hf_xxx  (and accept terms in browser) then re-run."
fi

# --- 5. upstream megakernel (gitignored; for sanity bench) -----------------
step "clone upstream qwen_megakernel"
[ -d kernels/qwen_megakernel/.git ] || \
  git clone https://github.com/AlpinDale/qwen_megakernel kernels/qwen_megakernel

# --- 6. pre-cache the model ------------------------------------------------
step "download model: $MODEL"
venv/bin/huggingface-cli download "$MODEL" \
  || warn "model download failed (token / gated terms?). Fix auth and re-run."

# --- 7. build talker kernel (JIT on first import, ~30s) --------------------
step "build talker kernel (JIT compile)"
venv/bin/pip install -r kernels/talker_kernel/requirements.txt
( cd kernels/talker_kernel && ../../venv/bin/python -c "import talker_megakernel; print('talker kernel built')" )

# --- 8. verify -------------------------------------------------------------
step "verify"
venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'torch-cuda', torch.version.cuda)"
venv/bin/python -c "from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration; print('qwen_tts ok')"

step "DONE. Next:"
cat <<EOF
  cd $WORKDIR/qwen-tts-pipeline
  # stock reference audio + RTF
  venv/bin/python scripts/baseline_tts.py
  # kernel driving the talker in-loop (writes kernel.wav)
  venv/bin/python scripts/kernel_generate.py
  # kernel perf + correctness vs HF
  cd kernels/talker_kernel && ../../venv/bin/python -m talker_megakernel.bench --mode correctness --prefill 32 --decode 64
  # optional: upstream sanity (~1000 tok/s)
  cd kernels/qwen_megakernel && ../../venv/bin/pip install -r requirements.txt && ../../venv/bin/python -m qwen_megakernel.bench
EOF
