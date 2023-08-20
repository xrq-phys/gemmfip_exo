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

