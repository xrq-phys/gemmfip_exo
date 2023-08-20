from gemmfip_exo import GEMMFIP

from gemmfip_exo.ref.sgemmfip import sgemmfip_mcxnr_getref
from gemmfip_exo.ref.dgemmfip import dgemmfip_mcxnr_getref

from gemmfip_exo.platforms import neon_f32
from gemmfip_exo.platforms import neon_f64


mr, nr, a_packed, b_packed, mr_inner = ( int(x) for x in input().split() )

a_packed = bool(a_packed)
b_packed = bool(b_packed)

generator = GEMMFIP(sgemmfip_mcxnr_getref, neon_f32.FMA)

p = generator.generate(mr, nr, a_packed, b_packed, mr_inner=mr_inner)


