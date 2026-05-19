"""Weight loading and high-level decode API for Qwen3-TTS talker decoder.

Forked from AlpinDale/qwen_megakernel (Qwen3-0.6B). The Qwen3-TTS-0.6B talker
decoder is architecturally IDENTICAL to Qwen3-0.6B except its LM head emits
codec tokens instead of text. Verified against the actual config.json of
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (talker_config: 28 layers, 1024 hidden,
8 KV heads, 3072 intermediate, 3072 vocab, 128 head_dim).

Per take-home spec ("If the talker decoder's backbone is a different size than
0.6B, document what you changed in the kernel and why"), the ONLY constant
that changes vs upstream is:

    VOCAB_SIZE  151936 → 3072   (LM head: codec tokens, not text)

All other dims match upstream Qwen3-0.6B exactly — no other kernel change.
"""

import math
import struct

import torch

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = 16 * HEAD_DIM  # 2048
KV_SIZE = 8 * HEAD_DIM  # 1024
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 3072

_decode = torch.ops.talker_megakernel_C.decode


def _detect_prefix(state: dict) -> str:
    """Detect whether state_dict is the standalone talker (`model.layers.0...`)
    or the full Qwen3-TTS model (`talker.model.layers.0...`). Returns the
    prefix that resolves to the talker submodel's `model.` root."""
    if "talker.model.layers.0.input_layernorm.weight" in state:
        return "talker.model."
    if "model.layers.0.input_layernorm.weight" in state:
        return "model."
    raise RuntimeError(
        "Could not locate talker decoder layers in state_dict. "
        "Expected `talker.model.layers.0.*` or `model.layers.0.*` keys."
    )


def load_weights(model_name="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", verbose: bool = True):
    """Load Qwen3-TTS talker decoder weights from HuggingFace into GPU tensors.

    Loads the full Qwen3-TTS checkpoint (gated repo — needs HF auth) via the
    qwen_tts package's `Qwen3TTSForConditionalGeneration` class, then extracts
    the talker submodel's state_dict and maps it to the megakernel's flat
    layer-weight layout.

    Key mapping (after `_detect_prefix` resolves to either `model.` for a
    standalone talker checkpoint or `talker.model.` for the full Qwen3-TTS
    checkpoint):
        {prefix}layers.{i}.input_layernorm.weight
        {prefix}layers.{i}.self_attn.{q,k,v,o}_proj.weight
        {prefix}layers.{i}.self_attn.{q,k}_norm.weight
        {prefix}layers.{i}.post_attention_layernorm.weight
        {prefix}layers.{i}.mlp.{gate,up,down}_proj.weight
        {prefix}norm.weight                  → final_norm
        {prefix}codec_embedding.weight       → embed table (codec tokens, NOT text)
        talker.codec_head.weight             → LM head (NOT tied to embedding)
    """
    if not verbose:
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers.utils import logging as hf_logging

    if not verbose:
        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except AttributeError:
            pass
        try:
            from huggingface_hub import logging as hf_hub_logging

            hf_hub_logging.set_verbosity_error()
        except Exception:
            pass

    if verbose:
        print(f"Loading {model_name}...")

    # Qwen3-TTS uses a custom class shipped in the qwen_tts package, not the
    # vanilla transformers AutoModelForCausalLM.
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )

    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    state = model.state_dict()
    prefix = _detect_prefix(state)

    # codec_head is on the talker root, NOT inside `talker.model.*`. Locate it.
    codec_head_key = (
        "talker.codec_head.weight" if prefix == "talker.model." else "codec_head.weight"
    )

    # RoPE tables (kept 1D — talker uses 3D mrope per spec; mrope correctness
    # impact documented in README as known integration limitation).
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    # Per-layer weight list (11 tensors per layer, flattened — same struct
    # layout as upstream qwen_megakernel; only the source keys differ).
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"{prefix}layers.{i}."
        layer_weights.extend(
            [
                state[p + "input_layernorm.weight"].contiguous(),
                state[p + "self_attn.q_proj.weight"].contiguous(),
                state[p + "self_attn.k_proj.weight"].contiguous(),
                state[p + "self_attn.v_proj.weight"].contiguous(),
                state[p + "self_attn.q_norm.weight"].contiguous(),
                state[p + "self_attn.k_norm.weight"].contiguous(),
                state[p + "self_attn.o_proj.weight"].contiguous(),
                state[p + "post_attention_layernorm.weight"].contiguous(),
                state[p + "mlp.gate_proj.weight"].contiguous(),
                state[p + "mlp.up_proj.weight"].contiguous(),
                state[p + "mlp.down_proj.weight"].contiguous(),
            ]
        )

    embed_weight = state[prefix + "codec_embedding.weight"].contiguous()
    lm_head_weight = state[codec_head_key].contiguous()

    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=state[prefix + "norm.weight"].contiguous(),
        lm_head_weight=lm_head_weight,  # codec_head — NOT tied to embedding
        cos_table=cos_table,
        sin_table=sin_table,
    )

    del model
    torch.cuda.empty_cache()
    # Talker doesn't take a text tokenizer at its boundary (input is
    # inputs_embeds composed by the caller). Tokenizer left as None; the
    # text encoder + processor live one level up in the TTS pipeline.
    return weights, None


def _pack_layer_weights(layer_weights: list[torch.Tensor]) -> torch.Tensor:
    """Pack 11-tensor-per-layer flat list into a device blob of LDGLayerWeights structs."""
    ptr_size = 8  # 64-bit pointers
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_bytes)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    t = torch.frombuffer(buf, dtype=torch.uint8).cuda()
    return t


class Decoder:
    """Stateful decoder wrapping the Talker Megakernel torch ops."""

    def __init__(
        self,
        weights=None,
        tokenizer=None,
        model_name="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        verbose: bool = True,
    ):
        if weights is None:
            weights, tokenizer = load_weights(model_name, verbose=verbose)
        self.tokenizer = tokenizer
        self._position = 0

        # Keep references so tensors stay alive (prevents GC of weight memory).
        self._weights = weights

        # Model weights (read-only, shared across calls)
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # KV cache
        self._k_cache = torch.zeros(
            NUM_LAYERS,
            NUM_KV_HEADS,
            MAX_SEQ_LEN,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers (single-token decode)
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

        # Embed-bypass scratch: kernel reads `embed_weight + token_id * HIDDEN`
        # at layer 0. By passing this single-row buffer as `embed_weight` and
        # token_id=0, we inject arbitrary `inputs_embeds` without editing CUDA.
        # Required for the talker decoder, whose inputs are not token IDs but
        # composed embeddings (codec_embed sum + trailing text hidden + pad).
        self._embed_scratch = torch.empty(HIDDEN_SIZE, **bf16)

    def step(self, token_id: int) -> int:
        """Decode one token (token-id input — only useful for non-talker tests).

        Talker production path uses `step_embed`; this method preserves the
        upstream qwen_megakernel API for smoke-testing the kernel against a
        token-input checkpoint."""
        _decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    def step_embed(self, inputs_embeds: torch.Tensor) -> int:
        """Decode one talker step. Input is a pre-composed embedding vector
        of shape `[HIDDEN_SIZE]` in bf16. Returns next codec token id.

        Caller composes `inputs_embeds` per the talker spec:
            inputs_embeds = sum(codec_token_embeds) + trailing_text_hidden[step]
                            + tts_pad_embed
        Output is `argmax(codec_head(final_hidden))` over codec vocab (3072).

        After this call, `self.last_hidden_state` holds the bf16
        post-final-norm hidden vector, suitable to feed into CodePredictor.
        """
        if inputs_embeds.dtype != torch.bfloat16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)
        if inputs_embeds.device.type != "cuda":
            inputs_embeds = inputs_embeds.cuda(non_blocking=True)
        if inputs_embeds.shape != (HIDDEN_SIZE,):
            raise ValueError(
                f"inputs_embeds must be shape [{HIDDEN_SIZE}], got {tuple(inputs_embeds.shape)}"
            )

        # Embed-bypass: write into row 0 of scratch, pass scratch as
        # embed_weight, token_id=0. Kernel reads `scratch + 0 * HIDDEN`.
        self._embed_scratch.copy_(inputs_embeds, non_blocking=True)
        _decode(
            self._out_token,
            0,
            self._embed_scratch,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    def prefill(self, inputs_embeds_seq: torch.Tensor) -> None:
        """Run a prefix of `inputs_embeds` rows through the talker to seed
        the KV cache. Shape: `[seq_len, HIDDEN_SIZE]` bf16. Discards outputs.

        Used to ingest text-encoder hidden states before autoregressive
        codec-token generation begins. Implemented as N single-step calls —
        the upstream megakernel has no batched-prefill entrypoint, and
        adding one is out of scope for this integration ("integration, not
        research" — see take-home spec).
        """
        if inputs_embeds_seq.dim() != 2 or inputs_embeds_seq.shape[1] != HIDDEN_SIZE:
            raise ValueError(
                f"inputs_embeds_seq must be [seq, {HIDDEN_SIZE}], got {tuple(inputs_embeds_seq.shape)}"
            )
        for i in range(inputs_embeds_seq.shape[0]):
            self.step_embed(inputs_embeds_seq[i])

    @property
    def last_hidden_state(self) -> torch.Tensor:
        """Final post-norm hidden vector from the most recent step. Shape
        `[HIDDEN_SIZE]` bf16. Feed into the CodePredictor's input embedding
        for the 31-step sub-codebook autoregressive loop."""
        return self._hidden

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)
        _gen = torch.ops.talker_megakernel_C.generate_nosync
        output_ids = _gen(
            ids[-1],
            max_tokens,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += max_tokens
        out = output_ids.cpu().tolist()
        eos = self.tokenizer.eos_token_id
        if eos in out:
            out = out[: out.index(eos)]
        return self.tokenizer.decode(out, skip_special_tokens=True)


def generate(prompt: str, max_tokens: int = 100, verbose: bool = True) -> str:
    """One-shot convenience: load model, generate, return text."""
    return Decoder(verbose=verbose).generate(prompt, max_tokens)
