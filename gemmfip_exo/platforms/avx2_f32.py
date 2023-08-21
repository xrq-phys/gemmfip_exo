# example.py
from __future__ import annotations
import exo
from exo.platforms.x86 import *


@exo.instr("_mm_mask_store_ss(&{dst_data}, __mmask8(1), __m128({src_data}));") # TODO: Pack & store?
def mm_mask_store_ss(dst: [f32][1] @ DRAM, src: [f32][8] @ AVX2):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[0] = src[i]

class FMA:
    Reg = AVX2
    prec = 'f32'
    prefix = 's'
    vlen = 8
    vld = mm256_loadu_ps # TODO: This is ups. How to use aps?
    vst = mm256_storeu_ps
    vld_broadcast = mm256_broadcast_ss
    vst_uniform = mm_mask_store_ss
    vfma = mm256_fmadd_ps

