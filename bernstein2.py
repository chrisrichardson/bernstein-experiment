import numpy as np
import scipy.special
from scipy.special import comb


# Pack index
def idx(i, j, n): return ((2 * n + 3) * j - j * j) // 2 + i


def compute_moments_triangle(n, f, fdegree):
    """Compute the Bernstein moments.

    These are defined in equation (12) of https://doi.org/10.1137/11082539X
    (Ainsworth, Andriamaro, Davydov, 2011).

    Args:
      n: The polynomial degree of the Bernstein polynomials.
      f: The function to take moments with.
      fdegree: The polynomial degree of the function f.

    Returns:
      A two-dimensional array containing the Bernstein moments.
    """
    # if fdegree == 0:
    #     return np.array(
    #         [[f(0, 0) / (n + 1) / (n + 2) for _ in range(n + 1)] for _ in range(n + 1)])

    rule1 = scipy.special.roots_jacobi((n + fdegree) // 2 + 1, 1, 0)
    rule2 = scipy.special.roots_jacobi((n + fdegree) // 2 + 1, 0, 0)
    rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
    rule2 = ((rule2[0] + 1) / 2, rule2[1] / 2)

    q = len(rule1[0])
    assert len(rule2[0]) == len(rule1[0])

    f0 = np.array([
        [f(p1, p2 * (1 - p1)) for p2 in rule2[0]]
        for p1 in rule1[0]])

    f1 = np.zeros((n+1, q))
    for i1, (p, w) in enumerate(zip(*rule1)):
        s = 1 - p
        r = p / (1 - p)
        ww = w
        for _ in range(n):
            ww *= s
        for alpha1 in range(n + 1):
            for i2 in range(q):
                f1[alpha1, i2] += ww * f0[i1, i2]
            ww *= r * (n - alpha1) / (1 + alpha1)

    f2 = np.zeros(((n + 1), (n + 1)))
    for i2, (p, w) in enumerate(zip(*rule2)):
        s = 1 - p
        r = p / s
        s0 = 1.0
        for alpha1 in range(n + 1):
            ww = w * s0
            s0 *= s
            for alpha2 in range(alpha1 + 1):
                print('idx=', i2, n-alpha1, alpha2)
                f2[n-alpha1, alpha2] += ww * f1[n-alpha1, i2]
                ww *= r * (alpha1 - alpha2) / (1 + alpha2)

    return f2


def compute_mass_matrix_triangle(n, f=None, fdegree=0):
    """
    Compute the mass matrix with a weight function.

    These method is described in section 4.1 of https://doi.org/10.1137/11082539X
    (Ainsworth, Andriamaro, Davydov, 2011).

    Args:
      n: The polynomial degree of the Bernstein polynomials.
      f: The function to take moments with. This is the function c from the paper.
      fdegree: The polynomial degree of the function f.

    Returns:
      A mass matrix.
    """
    moments = compute_moments_triangle(2 * n, f, fdegree)

    mat = np.zeros(((n + 1) * (n + 2) // 2, (n + 1) * (n + 2) // 2))

    # Copy over correct moments
    for a in range(n + 1):
        for a2 in range(n + 1 - a):
            i = idx(a2, a, n)
            for b in range(n + 1):
                for b2 in range(n + 1 - b):
                    j = idx(b2, b, n)
                    mat[i, j] = moments[a + b, a2 + b2]

    # First scaling (a outer)
    r = np.ones(n+1, dtype=int)
    w = np.empty(mat.shape[0], dtype=int)
    for a in range(n + 1):
        k = 0
        for b in range(n + 1):
            for b2 in range(n + 1 - b):
                w[k] = r[b]
                k += 1
        for a2 in range(n + 1 - a):
            i = idx(a2, a, n)
            print('first = ', i, w)
            mat[i, :] *= w
        r = np.cumsum(r)

    print(w[-1])

    print()
    # Second scaling (a2 outer)
    r = np.ones(n+1, dtype=int)
    for a2 in range(n + 1):
        k = 0
        for b in range(n + 1):
            for b2 in range(n + 1 - b):
                w[k] = r[b2]
                k += 1
        i = a2
        ia = n + 1
        for a in range(n + 1 - a2):
            # i = ((2 * n + 3) * a - a * a) // 2 + a2
            print('2nd row =', i, (2*n+3 - a)*a//2 + a2)
            mat[i, :] *= w
            i += ia
            ia -= 1
        r = np.cumsum(r)

    print()
    # Final scaling (a+a2 outer)
    r = np.ones(n+1, dtype=int)
    for asum in range(n,-1,-1):
        k = 0
        for b in range(n + 1):
            for b2 in range(n - b + 1):
                w[k] = r[n-b-b2]
                k += 1
        ww = 0
        w2 = n + 1
        for a2 in range(asum, -1, -1):
            i = ww + a2
            print('3rd row = ', i)
            mat[i, :] *= w
            ww += w2
            w2 -= 1

        r = np.cumsum(r)

    mat /= w[0]

    return mat


np.set_printoptions(linewidth=120)
print(compute_mass_matrix_triangle(3, lambda x,y: 1.0, 4))
