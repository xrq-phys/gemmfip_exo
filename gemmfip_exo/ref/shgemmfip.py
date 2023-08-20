# example.py
from __future__ import annotations
from exo import DRAM, proc

def shgemmfip_mcxnr_getref(
    mr: size,
    nr: size,
    a_packed: bool,
    b_packed: bool,
):
    def shgemmfip_mcxnr_ref(
        num_mr: size,
        k: size,
        rsC: size,
        rsA: size,
        rsB: size,
        alpha: f16[1] @ DRAM,
        C: f16[num_mr * mr, rsC] @ DRAM,
        A: f16[num_mr * mr, rsA] @ DRAM,
        B: f16[k, rsB] @ DRAM,
        beta: f16[1] @ DRAM,
        Abuffer: f16[num_mr, k, mr] @ DRAM,
        Bbuffer: f16[k, nr] @ DRAM,
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
    return proc(shgemmfip_mcxnr_ref)

