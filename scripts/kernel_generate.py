"""
Kernel-in-the-loop integration test (Phase C, step C9.3).

Monkeypatches the talker's 28-layer decode (`Qwen3TTSTalkerModel.forward`) to
run through OUR megakernel instead of the torch layers, then calls the stock
`generate_custom_voice`. Stock prefix-construction, CodePredictor, and vocoder
are reused unchanged — only the talker backbone decode is swapped.

This proves the kernel produces valid audio end-to-end and measures the real
RTF with the talker accelerated (vs the stock baseline ~1.55).

Run on the box:
    cd /workspace/qwen-tts-pipeline
    python scripts/kernel_generate.py

Writes kernel.wav. Compare against baseline.wav (scripts/baseline_tts.py).
"""

import sys
import time

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

sys.path.insert(0, "kernels/talker_kernel")
from talker_megakernel.model import (  # noqa: E402
    Decoder,
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_KV_HEADS,
    NUM_LAYERS,
)

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TEXT = "Real time voice agents live and die by latency. This is a baseline test."
LANGUAGE = "English"
MAX_NEW_TOKENS = 512   # runaway guard: kernel drift may never emit EOS

_decode_steps = 0      # counts single-token decode calls through the kernel


def main():
    from qwen_tts import Qwen3TTSModel

    print(f"loading {MODEL} ...")
    tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda", dtype=torch.bfloat16)
    m = tts.model
    speaker = list(tts.get_supported_speakers())[0]

    print("building megakernel decoder (loads talker weights) ...")
    kernel = Decoder(model_name=MODEL, verbose=True)

    # ---- monkeypatch the 28-layer talker decode ----
    # talker.forward composes inputs_embeds (codec + trailing text) then calls
    # self.model(inputs_embeds=...). We replace that .model.forward.
    #
    # Prefill vs decode is detected by sequence length (seq>1 = prefill). The
    # kernel keeps its own RoPE position counter, so HF's cache_position is
    # irrelevant to OUR math — but we still populate a dummy KV cache so HF's
    # generate loop advances and feeds one token per step (otherwise it would
    # re-feed the whole prompt every step).
    from transformers import DynamicCache

    def patched_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **kw,
    ):
        embeds = inputs_embeds
        bs, seq, h = embeds.shape
        assert bs == 1, "kernel path is batch-size 1 only"

        # CUDA step_embed assumes a contiguous [HIDDEN_SIZE] bf16 buffer; row
        # views from `embeds[0, i]` are strided → out-of-bounds read → GPU
        # deadlock. Force contiguous bf16 before every kernel call.
        embeds = embeds.contiguous().to(torch.bfloat16)

        if seq > 1:               # prefill
            print(f"  [prefill seq={seq}] ...", flush=True)
            kernel.reset()
            for i in range(seq):
                kernel.step_embed(embeds[0, i].contiguous())
            print(f"  [prefill done]", flush=True)
            last = kernel.last_hidden_state.clone()
            # only the final position's hidden is consumed by the LM head;
            # fill earlier positions with zeros (unused by generation).
            out = torch.zeros(1, seq, h, dtype=embeds.dtype, device=embeds.device)
            out[0, -1] = last.to(embeds.dtype)
        else:                     # single-token decode
            global _decode_steps
            _decode_steps += 1
            if _decode_steps % 25 == 0:
                print(f"  [decode step {_decode_steps}]", flush=True)
            kernel.step_embed(embeds[0, 0].contiguous())
            out = kernel.last_hidden_state.clone().view(1, 1, h).to(embeds.dtype)

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            dummy = torch.zeros(
                bs, NUM_KV_HEADS, seq, HEAD_DIM, dtype=embeds.dtype, device=embeds.device
            )
            for layer_idx in range(NUM_LAYERS):
                past_key_values.update(dummy, dummy, layer_idx, {})

        return BaseModelOutputWithPast(
            last_hidden_state=out, past_key_values=past_key_values
        )

    m.talker.model.forward = patched_forward

    # ---- warmup + timed run ----
    print("warmup ...")
    tts.generate_custom_voice(
        text="warm up.", speaker=speaker, language=LANGUAGE, max_new_tokens=32
    )

    global _decode_steps
    _decode_steps = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wavs, sr = tts.generate_custom_voice(
        text=TEXT, speaker=speaker, language=LANGUAGE, max_new_tokens=MAX_NEW_TOKENS
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    wav = wavs[0]
    audio_s = len(wav) / sr
    rtf = dt / audio_s if audio_s > 0 else float("nan")

    print(f"\n--- kernel-in-loop ---")
    print(f"  audio_len  = {audio_s:.2f} s")
    print(f"  synth_time = {dt*1000:.0f} ms")
    print(f"  RTF        = {rtf:.3f}   (baseline stock ~1.55; target < 0.15)")

    try:
        import soundfile as sf
        sf.write("kernel.wav", wav, sr)
        print("  wrote kernel.wav  — compare with baseline.wav")
    except ImportError:
        import numpy as np
        (np.clip(wav, -1, 1) * 32767).astype("<i2").tofile("kernel.pcm")
        print("  wrote kernel.pcm (raw s16le)")


if __name__ == "__main__":
    main()
