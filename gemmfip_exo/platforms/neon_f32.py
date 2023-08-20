# example.py
from __future__ import annotations
import exo
from exo.platforms.neon import *


@exo.instr("vst1q_lane_f32(&{dst_data}, {src_data}, 3);")
def neon_vst_lane3_f32(dst: [f32][1] @ DRAM, src: [f32][4] @ Neon):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[0] = src[i]

class FMA:
    prefix = 's'
    vlen = 4
    vld = neon_vld_4xf32
    vst = neon_vst_4xf32
    vld_broadcast = neon_broadcast_4xf32
    vst_uniform = neon_vst_lane3_f32
    vfma = neon_vfmadd_4xf32_4xf32

