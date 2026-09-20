# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""VTV container: header | segments | index (index at the end, offset in header).

    FILE_HEADER (40 bytes, little endian)
      magic 'VTVB' | ver_major u8 | ver_minor u8 | flags u16 | width u16 | height u16
      fps_num u32 | fps_den u32 | n_frames u32 | n_segments u32 | index_offset u64
      chroma_format u8 (1 = 4:2:0) | bit_depth u8 | reserved u16
    SEGMENT
      n_frames u16 | qp_i u8 | qp_p u8 | payload_len u32 | payload | crc32 u32
      (crc over header+payload).  Every segment starts with an I frame and fresh
      entropy-coder contexts, so it decodes without any other segment.
    INDEX  (n_segments entries)
      frame_start u32 | n_frames u16 | offset u64 | size u32

Writing the index last lets an encoder stream segments out in one pass; an HTTP
client fetches the 40-byte header, then the index via a range request, then any
segment it needs.
"""
import struct
import zlib

from .config import FLAG_GEO, FLAG_INTERP_SHIFT, FLAG_PHOTO, VERSION
from .rangecoder import BitstreamError

MAGIC = b"VTVB"
HDR = struct.Struct("<4sBBHHHIIIIQBBH")
SEG = struct.Struct("<HBBI")
IDX = struct.Struct("<IHQI")
CRC = struct.Struct("<I")

MAX_DIM = 8192
MAX_FRAMES = 1 << 24


def write_file(width, height, fps, flags, segments):
    """segments: list of (n_frames, qp_i, qp_p, payload_bytes).  Returns the file bytes."""
    n_frames = sum(s[0] for s in segments)
    body = bytearray()
    index = []
    start = 0
    for nf, qi, qp, payload in segments:
        off = HDR.size + len(body)
        rec = SEG.pack(nf, qi, qp, len(payload)) + payload
        rec += CRC.pack(zlib.crc32(rec) & 0xFFFFFFFF)
        index.append((start, nf, off, len(rec)))
        body += rec
        start += nf
    index_offset = HDR.size + len(body)
    hdr = HDR.pack(MAGIC, VERSION[0], VERSION[1], flags, width, height, fps[0], fps[1],
                   n_frames, len(segments), index_offset, 1, 8, 0)
    idx = b"".join(IDX.pack(*e) for e in index)
    return hdr + bytes(body) + idx


def read_header(data):
    if len(data) < HDR.size:
        raise BitstreamError("file too short")
    (magic, vmaj, vmin, flags, w, h, fn, fd, nfr, nseg, ioff, cf, bd, _r) = HDR.unpack_from(data, 0)
    if magic != MAGIC:
        raise BitstreamError("bad magic")
    if vmaj != VERSION[0]:
        raise BitstreamError(f"unsupported major version {vmaj}")
    if not (0 < w <= MAX_DIM and 0 < h <= MAX_DIM) or w % 2 or h % 2:
        raise BitstreamError("invalid dimensions")
    if cf != 1 or bd != 8:
        raise BitstreamError("only 4:2:0 8-bit supported in this version")
    if nfr > MAX_FRAMES or fd == 0 or nseg > nfr:
        raise BitstreamError("invalid frame/segment count")
    if ioff + nseg * IDX.size > len(data):
        raise BitstreamError("index outside file")
    return dict(version=(vmaj, vmin), flags=flags, width=w, height=h, fps=(fn, fd), n_frames=nfr,
                n_segments=nseg, index_offset=ioff,
                geo=bool(flags & FLAG_GEO), photo=bool(flags & FLAG_PHOTO),
                interp=(flags >> FLAG_INTERP_SHIFT) & 3)


def read_index(data, hdr):
    idx = []
    for i in range(hdr["n_segments"]):
        fs_, nf, off, size = IDX.unpack_from(data, hdr["index_offset"] + i * IDX.size)
        if off < HDR.size or off + size > hdr["index_offset"] or size < SEG.size + CRC.size:
            raise BitstreamError(f"segment {i}: bad index entry")
        idx.append(dict(frame_start=fs_, n_frames=nf, offset=off, size=size))
    return idx


def read_segment(data, entry):
    """Verify CRC and return (n_frames, qp_i, qp_p, payload).  Raises BitstreamError."""
    rec = data[entry["offset"]: entry["offset"] + entry["size"]]
    if len(rec) != entry["size"]:
        raise BitstreamError("truncated segment")
    body, (crc,) = rec[:-4], CRC.unpack(rec[-4:])
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise BitstreamError("segment CRC mismatch")
    nf, qi, qp, plen = SEG.unpack_from(body, 0)
    if plen != len(body) - SEG.size or nf == 0 or nf != entry["n_frames"] or qi > 51 or qp > 51:
        raise BitstreamError("inconsistent segment header")
    return nf, qi, qp, body[SEG.size:]
