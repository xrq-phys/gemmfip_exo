from base.sgemmfip import generate_sgemm

mr, nr, a_packed, b_packed = ( int(x) for x in input().split() )

a_packed = bool(a_packed)
b_packed = bool(b_packed)

p = generate_sgemm(mr, nr, a_packed, b_packed)


