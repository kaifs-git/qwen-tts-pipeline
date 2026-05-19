"""Benchmark: talker megakernel.

Two modes:

    --mode throughput     Synthetic random-weight kernel, random embed input.
                          Measures per-step latency + sustained tok/s on
                          `Decoder.step_embed(...)` — the talker production path.
                          No HF weights needed; kernel-only perf.

    --mode correctness    Loads real Qwen3-TTS talker weights, runs the same
                          `inputs_embeds` through (a) the talker megakernel and
                          (b) the stock HuggingFace
                          `Qwen3TTSTalkerForConditionalGeneration.forward(
                              inputs_embeds=..., past_key_values=...)`.
                          Compares last_hidden_state cosine similarity per step.
                          Quantifies the impact of keeping 1D RoPE (kernel)
                          while talker spec uses 3D mrope — the integration
                          tradeoff documented in README.

Replaces the upstream Qwen3-0.6B bench (broken after the talker constants
swap — 0.6B weights no longer fit the 20-layer / 2-KV-head / 3072-vocab
kernel).

Run:
    python -m talker_megakernel.bench --mode throughput
    python -m talker_megakernel.bench --mode correctness
"""

import argparse
import gc
import time
import warnings

import torch

from talker_megakernel.model import (
    Decoder,
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    KV_SIZE,
    MAX_SEQ_LEN,
    NUM_LAYERS,
    Q_SIZE,
    VOCAB_SIZE,
)

warnings.filterwarnings("ignore")

DEFAULT_PREFILL = 32   # text-encoder hidden-state prefix length
DEFAULT_DECODE = 256   # codec tokens to generate per run
DEFAULT_WARMUP = 3
DEFAULT_RUNS = 5


def synthetic_weights(seed: int = 0):
    """Build a weight dict shaped like `load_weights` output, but filled with
    random bf16 tensors. Kernel runs end-to-end; outputs are garbage. Used
    only for throughput timing — no semantic correctness needed.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    bf16 = dict(dtype=torch.bfloat16, device="cuda")

    def randn(*shape):
        return (
            torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32)
            .to(torch.bfloat16)
            .contiguous()
        )

    # RoPE tables (1D — talker uses 3D mrope; tradeoff noted in README).
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(**bf16).contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(**bf16).contiguous()

    layer_weights = []
    for _ in range(NUM_LAYERS):
        layer_weights.extend(
            [
                randn(HIDDEN_SIZE),                          # input_layernorm
                randn(Q_SIZE, HIDDEN_SIZE),                  # q_proj
                randn(KV_SIZE, HIDDEN_SIZE),                 # k_proj
                randn(KV_SIZE, HIDDEN_SIZE),                 # v_proj
                randn(HEAD_DIM),                             # q_norm
                randn(HEAD_DIM),                             # k_norm
                randn(HIDDEN_SIZE, Q_SIZE),                  # o_proj
                randn(HIDDEN_SIZE),                          # post_attn_layernorm
                randn(INTERMEDIATE_SIZE, HIDDEN_SIZE),       # gate_proj
                randn(INTERMEDIATE_SIZE, HIDDEN_SIZE),       # up_proj
                randn(HIDDEN_SIZE, INTERMEDIATE_SIZE),       # down_proj
            ]
        )

    return dict(
        embed_weight=randn(VOCAB_SIZE, HIDDEN_SIZE),  # codec_embedding stand-in
        layer_weights=layer_weights,
        final_norm_weight=randn(HIDDEN_SIZE),
        lm_head_weight=randn(VOCAB_SIZE, HIDDEN_SIZE),  # codec_head stand-in (NOT tied)
        cos_table=cos_table,
        sin_table=sin_table,
    )


def bench_throughput(prefill: int, decode: int, warmup: int, runs: int):
    weights = synthetic_weights()
    dec = Decoder(weights=weights, tokenizer=None, verbose=False)

    # Random text-encoder-style hidden prefix + per-step input embeds.
    prefix = torch.randn(prefill, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    step_embeds = torch.randn(decode, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    def run():
        dec.reset()
        dec.prefill(prefix)
        for i in range(decode):
            dec.step_embed(step_embeds[i])

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Measured
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p95 = sorted(times)[min(len(times) - 1, int(len(times) * 0.95))]

    # Per-step latency from the avg run, attributing prefill linearly.
    # (Prefill is N × step_embed, so total steps = prefill + decode.)
    total_steps = prefill + decode
    ms_per_step = avg * 1000.0 / total_steps

    return dict(
        prefill=prefill,
        decode=decode,
        runs=runs,
        avg_s=avg,
        p50_s=p50,
        p95_s=p95,
        tok_per_s=total_steps / avg,
        ms_per_step=ms_per_step,
    )


def bench_prefill_only(prefill: int, warmup: int, runs: int):
    """Isolate prefill cost — relevant to TTFC budget."""
    weights = synthetic_weights()
    dec = Decoder(weights=weights, tokenizer=None, verbose=False)
    prefix = torch.randn(prefill, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    for _ in range(warmup):
        dec.reset()
        dec.prefill(prefix)
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        dec.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dec.prefill(prefix)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    return dict(
        prefill=prefill,
        runs=runs,
        avg_ms=avg * 1000.0,
        ms_per_step=avg * 1000.0 / prefill,
    )


def correctness_check(prefill: int, decode: int, model_name: str = "Qwen/Qwen3-TTS"):
    """Compare kernel last_hidden_state vs stock HF talker forward, per step.

    Reports cosine similarity per step and aggregate min/mean. A score near
    1.0 means kernel output matches HF reference; lower scores indicate
    drift, primarily from kernel's 1D RoPE vs talker's spec'd 3D mrope.
    """
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )

    tts = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    tts.eval()
    talker = tts.talker

    # Real weights (uses our talker loader).
    dec = Decoder(model_name=model_name, verbose=False)

    # Random inputs_embeds — independent of any text path, isolates the
    # decoder forward itself.
    g = torch.Generator(device="cuda").manual_seed(0)
    seq = prefill + decode
    embeds = torch.randn(
        1, seq, HIDDEN_SIZE, generator=g, device="cuda", dtype=torch.bfloat16
    )

    # HF reference: one full forward, output_hidden_states at the model
    # output (post final norm).
    with torch.no_grad():
        hf_out = talker.model(
            inputs_embeds=embeds, use_cache=False, output_hidden_states=False
        )
        hf_hidden = hf_out.last_hidden_state[0]  # [seq, HIDDEN_SIZE]

    # Kernel: prefill + per-step.
    dec.reset()
    dec.prefill(embeds[0, :prefill])
    cos = torch.nn.functional.cosine_similarity

    sims = []
    for i in range(decode):
        dec.step_embed(embeds[0, prefill + i])
        mk_hidden = dec.last_hidden_state.float()
        ref_hidden = hf_hidden[prefill + i].float()
        sim = cos(mk_hidden.unsqueeze(0), ref_hidden.unsqueeze(0)).item()
        sims.append(sim)

    return dict(
        steps=decode,
        sim_min=min(sims),
        sim_mean=sum(sims) / len(sims),
        sim_p50=sorted(sims)[len(sims) // 2],
        per_step=sims,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=("throughput", "correctness", "prefill"), default="throughput")
    p.add_argument("--prefill", type=int, default=DEFAULT_PREFILL)
    p.add_argument("--decode", type=int, default=DEFAULT_DECODE)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    p.add_argument("--model", default="Qwen/Qwen3-TTS")
    args = p.parse_args()

    print("=" * 64)
    print("Talker Megakernel Benchmark")
    print(
        f"layers={NUM_LAYERS}  hidden={HIDDEN_SIZE}  kv_size={KV_SIZE}  "
        f"intermediate={INTERMEDIATE_SIZE}  vocab={VOCAB_SIZE}"
    )
    print("=" * 64)

    if args.mode == "throughput":
        r = bench_throughput(args.prefill, args.decode, args.warmup, args.runs)
        print(f"\n[throughput]  prefill={r['prefill']}  decode={r['decode']}  runs={r['runs']}")
        print(f"  avg = {r['avg_s']*1000:7.1f} ms / utterance")
        print(f"  p50 = {r['p50_s']*1000:7.1f} ms")
        print(f"  p95 = {r['p95_s']*1000:7.1f} ms")
        print(f"  tok/s         = {r['tok_per_s']:7.1f}")
        print(f"  ms / step     = {r['ms_per_step']:7.3f}")
        print("\nReference: AlpinDale's upstream Qwen3-0.6B megakernel hits ~1000 tok/s")
        print("on RTX 5090. Talker dims are smaller (20L, 2KV, 2048 MLP, 3072 vocab),")
        print("so this number should beat that.")

    elif args.mode == "prefill":
        r = bench_prefill_only(args.prefill, args.warmup, args.runs)
        print(f"\n[prefill]  N={r['prefill']}  runs={r['runs']}")
        print(f"  avg          = {r['avg_ms']:6.2f} ms")
        print(f"  ms / step    = {r['ms_per_step']:6.3f} ms")
        print("\nNote: prefill is N × step_embed (no batched kernel — see B1.4).")
        print("TTFC budget = prefill_time + 1 × step + codepredictor + vocoder.")

    elif args.mode == "correctness":
        r = correctness_check(args.prefill, args.decode, args.model)
        print(f"\n[correctness]  cosine sim (kernel vs HF talker.forward), {r['steps']} steps")
        print(f"  min  = {r['sim_min']:.4f}")
        print(f"  p50  = {r['sim_p50']:.4f}")
        print(f"  mean = {r['sim_mean']:.4f}")
        print("\nInterpretation:")
        print("  > 0.99  kernel matches HF — RoPE difference negligible for this input")
        print("  0.9-0.99 small drift — likely safe, audible-quality TBD")
        print("  < 0.9   significant drift — likely from 1D RoPE vs 3D mrope mismatch;")
        print("          document as known integration limitation per README")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
