import scipy
import sys
# Stroud conical quadrature points/weights
N = int(sys.argv[1])
for n in range(1, N):
    rule0 = scipy.special.roots_jacobi(n, 0, 0)
    rule1 = scipy.special.roots_jacobi(n, 1, 0)
    rule0 = ((rule0[0] + 1) / 2, rule0[1] / 2)
    rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)

    with open(f"gpu/quadrules/{n}.txt", "w") as f:
        for x in rule0[0]:
            f.write("%f " % x)
        f.write("\n")
        for w in rule0[1]:
            f.write("%f " % w)
        f.write("\n")
        for x in rule1[0]:
            f.write("%f " % x)
        f.write("\n")
        for w in rule1[1]:
            f.write("%f " % w)
        f.write("\n")
