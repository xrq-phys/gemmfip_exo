# example.py
from __future__ import annotations
from exo import DRAM, proc

def sgemmfip_mcxnr_getref(
    mr: size,
    nr: size,
    a_transpose: bool,
    a_packed: bool,
    b_packed: bool,
):
    def sgemmfip_notrans_notrans_mcxnr_ref(
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

    def sgemmfip_trans_notrans_mcxnr_ref(
        num_mr: size,
        k: size,
        rsC: size,
        rsA: size,
        rsB: size,
        alpha: f32[1] @ DRAM,
        C: f32[num_mr * mr, rsC] @ DRAM,
        A: f32[k, rsA] @ DRAM,
        B: f32[k, rsB] @ DRAM,
        beta: f32[1] @ DRAM,
        Abuffer: f32[num_mr, k, mr] @ DRAM,
        Bbuffer: f32[k, nr] @ DRAM,
    ):
        assert rsA >= num_mr * mr
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
                                C[ic * mr + ir, jr] += A[l, ic * mr + ir] * Bbuffer[l, jr] 
                            else:
                                C[ic * mr + ir, jr] += A[l, ic * mr + ir] * B[l, jr] 
                                Bbuffer[l, jr] = B[l, jr]
                            Abuffer[ic, l, ir] = A[l, ic * mr + ir]

    if not a_transpose:
        return proc(sgemmfip_notrans_notrans_mcxnr_ref)
    else:
        return proc(sgemmfip_trans_notrans_mcxnr_ref)

