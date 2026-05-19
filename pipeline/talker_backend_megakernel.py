"""
MegakernelTalkerBackend — real `TalkerBackend` implementation using the
adapted megakernel for the Qwen3-TTS talker decode step.

Architecture (Phase B/C):

    text
      │
      ▼
    Qwen3TTSProcessor (tokenize, build instruct/role tokens)
      │
      ▼  input_ids (text vocab)
    talker.text_embedding ─► talker.text_projection (2048 → 1024)
      │
      ▼  text_hiddens [seq, HIDDEN_SIZE=1024]
    talker_kernel.Decoder.prefill(...)            ◄── kernel fast path
      │
      ▼  KV cache populated
    autoregressive loop, step = 0..max_new_tokens-1:
        ┌────────────────────────────────────────────────────────┐
        │ talker_kernel.Decoder.step_embed(inputs_embeds[step])  │  ◄── kernel
        │   → primary codec_id  (argmax over codec_head, V=3072) │
        │   → last_hidden_state [HIDDEN_SIZE]                    │
        │                                                        │
        │ CodePredictor.generate(                                │  ◄── stock PyTorch
        │     inputs_embeds = cat(last_hidden, codec_embed(id)), │      (5-layer model,
        │     n=31)                                              │       eager mode)
        │   → 31 sub-codebook codes                              │
        │                                                        │
        │ all_codes = [primary] + [31 sub_codes]   (32-way)      │
        │                                                        │
        │ speech_tokenizer.decode(all_codes)                     │  ◄── stock vocoder
        │   → 24kHz mono float waveform chunk                    │
        │                                                        │
        │ → s16le bytes                                          │
        │ → yield                                                │
        │                                                        │
        │ compose next inputs_embeds:                            │
        │   sum(embed(all_codes))                                │
        │   + trailing_text_hidden[step]                         │
        │   + tts_pad_embed                                      │
        └────────────────────────────────────────────────────────┘

Local box has no GPU; this module is import-only safe (no torch/qwen_tts at
module top-level). Real instantiation requires CUDA 12.8 + sm_120 +
`Qwen/Qwen3-TTS` weights on disk.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

SAMPLE_RATE = 24000
CHANNELS = 1
BYTES_PER_SAMPLE = 2


class MegakernelTalkerBackend:
    """Real `TalkerBackend` — drives the adapted talker megakernel.

    Lazy-imports torch / qwen_tts / talker_megakernel so this module loads on
    the local laptop. Calling `__init__` here will fail without CUDA.
    """

    sample_rate = SAMPLE_RATE
    channels = CHANNELS

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-TTS",
        language: str = "english",
        speaker: Optional[str] = None,
        max_new_tokens: int = 4096,
        device: str = "cuda",
        verbose: bool = True,
    ):
        # Lazy imports keep this module loadable on non-GPU dev machines.
        import torch  # noqa: F401  (used by attr stashes below)
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        # Talker-side kernel; the package is built under `kernels/talker_kernel`.
        from talker_megakernel.model import Decoder as TalkerKernelDecoder

        self._device = device
        self._language = language
        self._speaker = speaker
        self._max_new_tokens = max_new_tokens

        if verbose:
            print(f"[MegakernelTalkerBackend] loading {model_name} ...")

        # Full Qwen3-TTS model: text encoder pieces (text_embedding +
        # text_projection), talker config (eos / bos / pad ids), code_predictor,
        # and speech_tokenizer (vocoder). The talker decoder's `model.layers.*`
        # weights are pulled from this state_dict by the kernel loader below;
        # the rest stays in stock PyTorch.
        self._tts = Qwen3TTSForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=device
        )
        self._tts.eval()

        # Convenience handles.
        self._talker = self._tts.talker
        self._talker_cfg = self._tts.config.talker_config
        self._code_predictor = self._talker.code_predictor
        self._codec_embedding = self._talker.get_input_embeddings()
        self._text_embedding = self._talker.get_text_embeddings()
        self._text_projection = self._talker.text_projection
        self._codec_head = self._talker.codec_head
        self._speech_tokenizer = self._tts.speech_tokenizer  # vocoder

        # Kernel: load talker weights into the adapted megakernel.
        self._kernel = TalkerKernelDecoder(model_name=model_name, verbose=verbose)

        # Cache constants from talker config for the generate loop.
        self._codec_eos = self._talker_cfg.codec_eos_token_id
        self._codec_bos = self._talker_cfg.codec_bos_id
        self._codec_pad = self._talker_cfg.codec_pad_id
        self._num_code_groups = self._talker_cfg.num_code_groups  # 32

    # -------------------------------------------------------------------------
    # TalkerBackend protocol
    # -------------------------------------------------------------------------

    async def stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        """Tokenize, build talker prefix embeds, prefill kernel, then yield
        24kHz mono s16le PCM chunks as codec frames are decoded.

        Status: scaffold. Requires vast.ai correctness validation against
        stock `Qwen3TTSForConditionalGeneration.generate(non_streaming_mode=
        False)`. Lands in Phase C.
        """
        import torch

        if not text.strip():
            return

        # ---------------------------------------------------------------------
        # 1. Build the initial talker_input_embed using the same construction
        #    as upstream's `generate()` (modeling_qwen3_tts.py lines ~2068-2202).
        #    Simplified: no voice clone, no instruct, no ICL — single utterance,
        #    `language` + optional `speaker`.
        # ---------------------------------------------------------------------
        input_ids, trailing_text_hidden, tts_pad_embed = self._build_prefix(text)
        # `input_ids` is the talker-side composed prefix embedding tensor of
        # shape [1, seq, HIDDEN_SIZE]. `trailing_text_hidden` is the per-step
        # text conditioning consumed in the generate loop. `tts_pad_embed` is
        # the constant residual once we run out of trailing text.

        prefix = input_ids[0]  # [seq, HIDDEN_SIZE]
        seq_len = prefix.shape[0]

        # ---------------------------------------------------------------------
        # 2. Prefill the megakernel KV cache with the prefix.
        # ---------------------------------------------------------------------
        self._kernel.reset()
        self._kernel.prefill(prefix[:-1])  # all but last; last drives first step

        current_embed = prefix[-1]  # [HIDDEN_SIZE]

        # ---------------------------------------------------------------------
        # 3. Autoregressive loop. Each iteration emits one codec frame
        #    → one PCM chunk (12 Hz frame rate → ~83ms audio per step on the
        #    12hz tokenizer variant; matches the streaming target).
        # ---------------------------------------------------------------------
        codec_history: list[int] = []
        for step in range(self._max_new_tokens):
            # 3a. Kernel: one talker step. Argmax over codec_head is done
            #     inside the kernel; primary codec id returned directly.
            primary_codec_id = self._kernel.step_embed(current_embed)
            if primary_codec_id == self._codec_eos:
                break

            talker_hidden = self._kernel.last_hidden_state  # [HIDDEN_SIZE] bf16

            # 3b. CodePredictor: 31-step inner autoregressive loop over
            #     sub-codebooks. Stock PyTorch (eager). Returns 31 codes.
            sub_codes = self._run_code_predictor(talker_hidden, primary_codec_id)

            # 3c. All 32 codes → vocoder → PCM chunk.
            pcm_chunk_bytes = self._vocoder_decode(primary_codec_id, sub_codes)
            if pcm_chunk_bytes:
                yield pcm_chunk_bytes

            # 3d. Compose next `inputs_embeds`:
            #     sum(embed(code) for code in [primary, *sub_codes])
            #     + trailing_text_hidden[step] (or tts_pad_embed past the end)
            #     + tts_pad_embed
            current_embed = self._compose_next_embed(
                primary_codec_id, sub_codes, step, trailing_text_hidden, tts_pad_embed
            )

            codec_history.append(primary_codec_id)

    # -------------------------------------------------------------------------
    # Internals (require GPU + Qwen3-TTS weights)
    # -------------------------------------------------------------------------

    def _build_prefix(self, text: str):
        """Tokenize + build talker prefix per upstream `generate()`.

        Returns:
            input_ids:            [1, seq, HIDDEN_SIZE] bf16  — talker-input embeds
            trailing_text_hidden: [1, T_text, HIDDEN_SIZE]    — per-step text cond
            tts_pad_embed:        [1, 1, HIDDEN_SIZE]         — pad residual

        See `modeling_qwen3_tts.py::Qwen3TTSForConditionalGeneration.generate`
        for the upstream construction (lines ~2068-2220). The minimum-viable
        path here covers: language tag, optional speaker, single text turn,
        no voice clone, no instruct prefix.
        """
        raise NotImplementedError(
            "Prefix construction needs validation against upstream generate(). "
            "Implement on vast.ai with the model loaded — diff against the "
            "first `talker_input_embed` produced by stock generate() for the "
            "same text + language + speaker, then transcribe the construction."
        )

    def _run_code_predictor(self, talker_hidden, primary_codec_id) -> list[int]:
        """31-step inner loop over CodePredictor for the 31 sub-codebooks."""
        import torch

        with torch.inference_mode():
            last_id_hidden = self._codec_embedding(
                torch.tensor([[primary_codec_id]], device=self._device)
            )
            predictor_result = self._code_predictor.generate(
                inputs_embeds=torch.cat(
                    (talker_hidden.view(1, 1, -1).to(last_id_hidden.dtype), last_id_hidden),
                    dim=1,
                ),
                max_new_tokens=self._num_code_groups - 1,
                do_sample=False,
                output_hidden_states=False,
                return_dict_in_generate=True,
            )
        return predictor_result.sequences.view(-1).tolist()

    def _vocoder_decode(self, primary_codec_id: int, sub_codes: list[int]) -> bytes:
        """Decode 32-way codec codes → 24kHz mono PCM bytes.

        Stock `speech_tokenizer.decode(...)` API needs vast.ai validation —
        Qwen3-TTS ships a frame-by-frame streaming decoder in the v2 (12Hz)
        tokenizer. Should produce one ~83ms chunk per call.
        """
        raise NotImplementedError(
            "speech_tokenizer.decode signature varies between v1 (25Hz) and v2 "
            "(12Hz) variants. Confirm against the variant in the loaded "
            "checkpoint on vast.ai, then convert float32 waveform → s16le bytes."
        )

    def _compose_next_embed(
        self, primary_codec_id, sub_codes, step, trailing_text_hidden, tts_pad_embed
    ):
        """Build next-step inputs_embeds per upstream's generate inner loop
        (modeling_qwen3_tts.py ~lines 1670-1692).

        codec_hiddens = sum(embed(code) for code in [primary, *sub_codes])
        next         = codec_hiddens + (trailing_text_hidden[step]
                                        if step < T_text else tts_pad_embed)
                                     + tts_pad_embed
        """
        import torch

        with torch.inference_mode():
            primary_emb = self._codec_embedding(
                torch.tensor([primary_codec_id], device=self._device)
            )  # [1, HIDDEN_SIZE]
            sub_embs = []
            sub_embed_tables = self._code_predictor.get_input_embeddings()
            for i, sc in enumerate(sub_codes):
                sub_embs.append(
                    sub_embed_tables[i](torch.tensor([sc], device=self._device))
                )
            codec_sum = primary_emb + sum(sub_embs)  # [1, HIDDEN_SIZE]

            T_text = trailing_text_hidden.shape[1]
            if step < T_text:
                text_cond = trailing_text_hidden[:, step]
            else:
                text_cond = tts_pad_embed[:, 0]

            next_embed = codec_sum + text_cond + tts_pad_embed[:, 0]
        return next_embed.view(-1)  # [HIDDEN_SIZE]
