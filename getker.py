from gemmfip_exo import GEMMFIP

from gemmfip_exo.ref.shgemmfip import shgemmfip_mcxnr_getref
from gemmfip_exo.ref.sgemmfip import sgemmfip_mcxnr_getref
from gemmfip_exo.ref.dgemmfip import dgemmfip_mcxnr_getref

from gemmfip_exo.platforms import neon_f16
from gemmfip_exo.platforms import neon_f32
from gemmfip_exo.platforms import neon_f64
from gemmfip_exo.platforms import avx2_f32


dtype, mr, nr, a_transpose, a_packed, b_packed, mr_inner = tuple(input().split())
dtype = dtype.lower()
mr = int(mr)
nr = int(nr)
a_transpose = bool(int(a_transpose))
a_packed = int(a_packed)
b_packed = int(b_packed)
mr_inner = int(mr_inner)

a_packed = bool(int(a_packed))
b_packed = bool(int(b_packed))

if dtype == 'sh':
    generator = GEMMFIP(shgemmfip_mcxnr_getref, neon_f16.FMA)
elif dtype == 's':
    # generator = GEMMFIP(sgemmfip_mcxnr_getref, avx2_f32.FMA)
    generator = GEMMFIP(sgemmfip_mcxnr_getref, neon_f32.FMA)
elif dtype == 'd':
    generator = GEMMFIP(dgemmfip_mcxnr_getref, neon_f64.FMA)

p = generator.generate(
    mr, nr,
    a_transpose=a_transpose,
    a_packed=a_packed,
    b_packed=b_packed,
    mr_inner=mr_inner,
)

