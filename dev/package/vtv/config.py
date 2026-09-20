# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Shared constants and the encoder configuration for VTV beta."""
from dataclasses import dataclass

VERSION = (0, 2)          # bitstream version (major, minor)
CTU = 32                  # quadtree root size (luma pixels)
MIN_LEAF = 8              # smallest leaf; also the residual-transform block size
UNIT = 16                 # luma area that maps to one 8x8 chroma block

# header flag bits
FLAG_GEO = 1              # global affine trajectory present
FLAG_PHOTO = 2            # photometric (gain/offset) trajectory present
FLAG_INTERP_SHIFT = 2     # bits 2-3: interpolation filter id
INTERP_BILINEAR = 0
INTERP_CATMULL = 1


@dataclass
class EncoderConfig:
    qp: int = 32                  # base quantizer (P frames)
    qp_i_offset: int = -3         # I frames use qp + offset
    gop: int = 32                 # max frames per independently decodable segment
    use_global: bool = True       # global affine transform trajectory
    use_traj: bool = True         # polynomial temporal trajectories (False = per-frame params)
    use_photo: bool = True        # gain/offset trajectory
    interp: int = INTERP_CATMULL
    search_range: int = 8         # integer full-search range for regional deltas (pixels)
    subpel: bool = True           # 1/2 and 1/4 pel refinement
    traj_max_len: int = 16        # max frames per trajectory segment
    tol_geo: float = -1.0         # pixels; <0 = derived from qp
    tol_photo: float = -1.0       # gray levels; <0 = derived from qp
    lam_scale: float = 0.13       # lambda = lam_scale * qstep^2
    cut_mad: float = 18.0         # scene-cut threshold on compensated MAD
    rdoq: bool = True             # block-level zero-out decision
    stats: bool = True            # collect per-category bit statistics

    def qp_i(self) -> int:
        return max(0, min(51, self.qp + self.qp_i_offset))
