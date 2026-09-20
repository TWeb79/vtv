# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""VTV - Vector-Time Video, research codec (beta)."""
from .config import EncoderConfig, VERSION
from .encoder import encode, Encoder
from .decoder import decode, decode_segment

__all__ = ["EncoderConfig", "VERSION", "encode", "Encoder", "decode", "decode_segment"]
__version__ = "%d.%d-beta" % VERSION
