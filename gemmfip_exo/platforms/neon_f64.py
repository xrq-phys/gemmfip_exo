# example.py
from __future__ import annotations
import exo
from exo.platforms.neon import *


@exo.instr("vst1q_lane_f64(&{dst_data}, {src_data}, 1);")
def neon_vst_lane1_f64(dst: [f64][1] @ DRAM, src: [f64][2] @ Neon):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 2):
        dst[0] = src[i]

class FMA:
    prec = 'f64'
    prefix = 'd'
    vlen = 2
    vld = neon_vld_2xf64
    vst = neon_vst_2xf64
    vld_broadcast = neon_broadcast_2xf64
    vst_uniform = neon_vst_lane1_f64
    vfma = neon_vfmadd_2xf64_2xf64

