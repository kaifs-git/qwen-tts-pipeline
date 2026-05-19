"""Talker Megakernel — single-kernel Qwen3-TTS talker decoder for RTX 5090.

Forked from AlpinDale/qwen_megakernel; constants adapted to talker dims
(20 layers, 2 KV heads, 2048 MLP, 3072 codec vocab).
"""

from talker_megakernel.build import get_extension as _get_ext

_get_ext()

from talker_megakernel.model import load_weights, Decoder, generate  # noqa: E402

__all__ = ["load_weights", "Decoder", "generate"]
