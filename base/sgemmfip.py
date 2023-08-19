# example.py
from __future__ import annotations
import exo
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *

from exo.stdlib.scheduling import *

@exo.instr("vst1q_lane_f32(&{dst_data}, {src_data}, 3);")
def neon_vst_lane3_f32(dst: [f32][1] @ DRAM, src: [f32][4] @ Neon):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[0] = src[i]

# @proc
def sgemmfip_mcxnr_getref(
    mr: size,
    nr: size,
    a_packed: bool,
    b_packed: bool,
):
    def sgemmfip_mcxnr_ref(
        num_mr: size,
        k: size,
        rsC: size,
        rsA: size,
        rsB: size,
        alpha: f32[1] @ DRAM,
        C: f32[num_mr * mr, rsC] @ DRAM,
        A: f32[num_mr * mr, rsA] @ DRAM,
        B: f32[k, rsB] @ DRAM,
        beta: f32[1] @ DRAM,
        Abuffer: f32[num_mr, k, mr] @ DRAM,
        Bbuffer: f32[k, nr] @ DRAM,
    ):
        assert rsA >= k
        assert rsB >= nr
        assert rsC >= nr

        for ic in seq(0, num_mr):
            for l in seq(0, k):
                for jr in seq(0, nr):
                    for ir in seq(0, mr):
                        if a_packed:
                            if b_packed: # or ic > 0:
                                C[ic * mr + ir, jr] += Abuffer[ic, l, ir] * Bbuffer[l, jr] 
                            else:
                                C[ic * mr + ir, jr] += Abuffer[ic, l, ir] * B[l, jr] 
                                Bbuffer[l, jr] = B[l, jr]
                        else:
                            if b_packed: # or ic > 0:
                                C[ic * mr + ir, jr] += A[ic * mr + ir, l] * Bbuffer[l, jr] 
                            else:
                                C[ic * mr + ir, jr] += A[ic * mr + ir, l] * B[l, jr] 
                                Bbuffer[l, jr] = B[l, jr]
                            Abuffer[ic, l, ir] = A[ic * mr + ir, l]
    return proc(sgemmfip_mcxnr_ref)


def generate_sgemm(mr, nr, a_packed, b_packed):
    p = simplify(sgemmfip_mcxnr_getref(mr, nr, a_packed, b_packed))
    # p = p.partial_eval(mr, nr, a_packed, b_packed)
    # print(p)

    p = rename(p, "sgemmfip_{}x{}_{}{}".format(mr, nr, a_packed, b_packed))
    p = divide_loop(p, 'jr', 4, ['jr', 'jvec'], perfect=True)
    p = reorder_loops(p, 'jvec ir')
    print(p)

    p = stage_mem(p, 'C[_] += _', 'C[ic * {} + ir, jr * 4 + jvec]'.format(mr), 'C_reg')
    p = expand_dim(p, 'C_reg', 4, 'jvec', unsafe_disable_checks=True)
    p = expand_dim(p, 'C_reg', mr, 'ir', unsafe_disable_checks=True)
    p = expand_dim(p, 'C_reg', nr // 4, 'jr', unsafe_disable_checks=True)
    print(p)

    n_unpacked = [a_packed, b_packed].count(False)

    p = lift_alloc(p, 'C_reg', n_lifts=4)
    p = autofission(p, p.find('C_reg = _ #0').after(), n_lifts=4)
    for i in range(n_unpacked):
        p = reorder_stmts(p, p.find('C[_] = C_reg[_]').expand(0, 1))
    p = autofission(p, p.find('C[_] = _ #0').before(), n_lifts=4)
    print(p)

    p = set_memory(p, 'C_reg', Neon)
    p = replace(p, 'for jvec in _: _ #0', neon_vld_4xf32)
    p = replace(p, 'for jvec in _: _ #1', neon_vst_4xf32)
    p = simplify(p)
    print(p)


    if a_packed:
        p = bind_expr(p, 'Abuffer[ic, l, _]', 'A_vec')
    else:
        p = bind_expr(p, 'A[ir + {} * ic, l]'.format(mr), 'A_vec', cse=True)
    p = set_memory(p, 'A_vec', Neon)
    p = expand_dim(p, 'A_vec:_', '4', 'jvec')
    p = lift_alloc(p, 'A_vec:_', n_lifts=1)
    p = autofission(p, p.find('A_vec = _ #0').after(), n_lifts=1)
    if not a_packed:
        p = autofission(p, p.find('Abuffer[_] = _ #0').before(), n_lifts=1)
    print(p)

    if b_packed:
        p = bind_expr(p, 'Bbuffer[l, _]', 'B_vec')
    else:
        p = bind_expr(p, 'B[l, _]', 'B_vec', cse=True)
    p = set_memory(p, 'B_vec', Neon)
    p = autolift_alloc(p, 'B_vec:_', keep_dims=True)
    p = autofission(p, p.find('B_vec[_] = _').after())
    if not b_packed:
        p = autofission(p, p.find('Bbuffer[_] = _ #0').before(), n_lifts=1)
    print(p)

    p = divide_loop(p, 'ir #1', 4, ['iro', 'iri'], perfect=True)
    p = reorder_loops(p, 'jr iro')
    p = expand_dim(p, 'A_vec', 4, 'iri', unsafe_disable_checks=True)
    p = expand_dim(p, 'B_vec', 3, 'jr', unsafe_disable_checks=True)
    p = lift_alloc(p, 'A_vec:_', n_lifts=4)
    p = lift_alloc(p, 'B_vec:_', n_lifts=4)
    print(p)

    p = autofission(p, p.find('for jvec in _: _ #0').after(), n_lifts=2)
    p = autofission(p, p.find('for jvec in _: _ #1').after(), n_lifts=2)
    if n_unpacked > 0:
        p = autofission(p, p.find('for jvec in _: _ #2').after(), n_lifts=2)
    if n_unpacked > 1:
        p = autofission(p, p.find('for jvec in _: _ #3').after(), n_lifts=2)
    print(p)

    p = replace_all(p, neon_vld_4xf32)
    p = replace_all(p, neon_broadcast_4xf32)
    p = replace_all(p, neon_vfmadd_4xf32_4xf32)
    if not b_packed:
        p = replace_all(p, neon_vst_4xf32)
    if not a_packed:
        p = replace(p, 'for jvec in _: _ #0', neon_vst_lane3_f32)
    print(p)

    # LD
    p = unroll_loop(p, 'ir #0')
    p = unroll_loop(p, 'jr #0')
    # FMA
    p = unroll_loop(p, 'iri #0') # LD A
    p = unroll_loop(p, 'jr #0')  # LD B
    p = unroll_loop(p, 'iri #0') # FMA
    p = unroll_loop(p, 'jr #0')  # FMA
    if not a_packed:
        p = unroll_loop(p, 'iri #0') # ST A
    if not b_packed:
        p = unroll_loop(p, 'jr #0') # ST B
    p = unroll_loop(p, 'iro #0') # Outside
    # ST
    p = unroll_loop(p, 'ir #0')
    p = unroll_loop(p, 'jr #0')
    return p


if __name__ == "__main__":
    M=6
    N=8
    K=2

    p=generate_sgemm(M,N,K,True ,True ); print(p)
    p=generate_sgemm(M,N,K,True ,False); print(p)
    p=generate_sgemm(M,N,K,False,True ); print(p)
    p=generate_sgemm(M,N,K,False,False); print(p)
