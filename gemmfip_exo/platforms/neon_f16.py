# example.py
from __future__ import annotations
import exo
from exo.platforms.neon import *


@exo.instr("vst1q_lane_f16(&{dst_data}, {src_data}, 7);")
def neon_vst_lane7_f16(dst: [f16][1] @ DRAM, src: [f16][8] @ Neon):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[0] = src[i]

class FMA:
    prec = 'f16'
    prefix = 'sh'
    vlen = 8
    vld = neon_vld_8xf16
    vst = neon_vst_8xf16
    vld_broadcast = neon_broadcast_8xf16
    vst_uniform = neon_vst_lane7_f16
    vfma = neon_vfmadd_8xf16_8xf16

