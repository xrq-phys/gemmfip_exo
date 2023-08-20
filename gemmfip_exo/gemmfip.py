# example.py
from __future__ import annotations
import exo
from exo import proc
from exo.platforms.neon import Neon
from exo.stdlib.scheduling import *


class GEMMFIP:
    def __init__(self, getref, FMA):
        self.getref = getref
        self.FMA = FMA

    def generate(self, mr, nr, a_packed, b_packed, mr_inner=None):
        p = simplify(self.getref(mr, nr, a_packed, b_packed))
        # p = p.partial_eval(mr, nr, a_packed, b_packed)
        # print(p)
        FMA = self.FMA

        p = rename(p, "{}gemmfip_{}x{}_{}{}".format(FMA.prefix, mr, nr, a_packed, b_packed))
        p = divide_loop(p, 'jr', FMA.vlen, ['jr', 'jvec'], perfect=True)
        p = reorder_loops(p, 'jvec ir')
        print(p)

        p = stage_mem(p, 'C[_] += _', 'C[ic * {} + ir, jr * {} + jvec]'.format(mr, FMA.vlen), 'C_reg')
        p = expand_dim(p, 'C_reg', FMA.vlen, 'jvec', unsafe_disable_checks=True)
        p = expand_dim(p, 'C_reg', mr, 'ir', unsafe_disable_checks=True)
        p = expand_dim(p, 'C_reg', nr // FMA.vlen, 'jr', unsafe_disable_checks=True)
        print(p)

        n_unpacked = [a_packed, b_packed].count(False)

        p = lift_alloc(p, 'C_reg', n_lifts=4)
        p = autofission(p, p.find('C_reg = _ #0').after(), n_lifts=4)
        for i in range(n_unpacked):
            p = reorder_stmts(p, p.find('C[_] = C_reg[_]').expand(0, 1))
        p = autofission(p, p.find('C[_] = _ #0').before(), n_lifts=4)
        print(p)

        p = set_memory(p, 'C_reg', Neon)
        p = replace(p, 'for jvec in _: _ #0', FMA.vld)
        p = replace(p, 'for jvec in _: _ #1', FMA.vst)
        p = simplify(p)
        print(p)


        if a_packed:
            p = bind_expr(p, 'Abuffer[ic, l, _]', 'A_vec')
        else:
            p = bind_expr(p, 'A[ir + {} * ic, l]'.format(mr), 'A_vec', cse=True)
        p = set_memory(p, 'A_vec', Neon)
        p = set_precision(p, 'A_vec', FMA.prec)
        p = expand_dim(p, 'A_vec:_', str(FMA.vlen), 'jvec')
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
        p = set_precision(p, 'B_vec', FMA.prec)
        p = autolift_alloc(p, 'B_vec:_', keep_dims=True)
        p = autofission(p, p.find('B_vec[_] = _').after())
        if not b_packed:
            p = autofission(p, p.find('Bbuffer[_] = _ #0').before(), n_lifts=1)
        print(p)

        # Have to mr % mr_inner == 0.
        # Inner loop is fully allocated to A-regs.
        if mr_inner is None:
            # TODO: Find the largest denominator smaller than remaining registers.
            raise Exception("Not implemented: automatic mr_inner picking")
        p = divide_loop(p, 'ir #1', mr_inner, ['iro', 'iri'], perfect=True)
        p = reorder_loops(p, 'jr iro')
        p = expand_dim(p, 'A_vec', mr_inner, 'iri', unsafe_disable_checks=True)
        p = expand_dim(p, 'B_vec', nr // FMA.vlen, 'jr', unsafe_disable_checks=True) # jr loops over range(nr // vlen)
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

        p = replace_all(p, FMA.vld)
        p = replace_all(p, FMA.vld_broadcast)
        p = replace_all(p, FMA.vfma)
        if not b_packed:
            p = replace_all(p, FMA.vst)
        if not a_packed:
            p = replace(p, 'for jvec in _: _ #0', FMA.vst_uniform)
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

