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

    f2 = np.zeros((n + 1)*(n + 2)//2)
    for i2, (p, w) in enumerate(zip(*rule2)):
        s = 1 - p
        r = p / s
        s0 = 1.0
        for alpha1 in range(n + 1):
            ww = w * s0
            s0 *= s
            for alpha2 in range(alpha1 + 1):
                f2[idx(n-alpha1, alpha2, n)] += ww * f1[n-alpha1, i2]
                ww *= r * (alpha1 - alpha2) / (1 + alpha2)

    ccode = f"""

void tabulate_bernstein_mass_tri(double *f0, double *A)
{{
  // Input: f0 at quadrature points (size {q} x {q})
  // Output: A - mass matrix (size {(n+1)*(n+2)//2} x {(n+1)*(n+2)//2})
  double rule1p[{q}] = {{{', '.join([str(p) for p in rule1[0]])}}};
  double rule1w[{q}] = {{{', '.join([str(p) for p in rule1[1]])}}};
  double rule2p[{q}] = {{{', '.join([str(p) for p in rule2[0]])}}};
  double rule2w[{q}] = {{{', '.join([str(p) for p in rule2[1]])}}};

  double f1[{(2*n+1)}][{q}] = {{}};

  for (int i1 = 0; i1 < {q}; ++i1)
  {{
    double s = 1.0 - rule1p[i1];
    double r = rule1p[i1] / s;
    double ww = rule1w[i1];
    for (int alpha1 = 0; alpha1 < {2*n}; ++alpha1)
      ww *= s;
    for (int alpha1 = 0; alpha1 < {2*n+1}; ++alpha1)
    {{
      for (int i2 = 0; i2 < {q}; ++i2)
        f1[alpha1][i2] += ww * f0[i1*{q} + i2];
      ww *= r *({2*n}-alpha1)/(1.0 + alpha1);
    }}
  }}

  double f2[{(2*n+1)}][{(2*n+1)}] = {{0}};

  for (int i2 = 0; i2 < {q}; ++i2)
  {{
    double s = 1.0 - rule2p[i2];
    double r = rule2p[i2]/s;
    double s0 = 1.0;
    for (int alpha1 = 0; alpha1 < {2*n+1}; ++alpha1)
    {{
      double ww = rule2w[i2] * s0;
      s0 *= s;
      for (int alpha2 = 0; alpha2 < alpha1+1; ++alpha2)
      {{
        f2[{2*n}-alpha1][alpha2] += ww * f1[{2*n}-alpha1][i2];
        ww *= r * (alpha1 - alpha2) / (1.0 + alpha2);
      }}
    }}
  }}


    // double A[{(n + 1) * (n + 2) // 2}][{(n + 1) * (n + 2) // 2}] = {{0}};
    double cmat[{n+1}][{n+1}] = {{{', '.join([str(comb(p+q, p)) for p in range(n+1) for q in range(n+1)])}}};

    for (int a = 0; a < {n+1}; ++a)
    {{
      for (int a2 = 0; a2 < ({n + 1} - a); ++a2)
      {{
        int i = ({(2 * n + 3)} * a2 - a2 * a2) / 2 + a;
        for (int b = 0; b < {n + 1}; ++b)
        {{
          for (int b2 = 0; b2 < {n + 1} - b; ++b2)
          {{
          int j = ({(2 * n + 3)} * b2 - b2 * b2) / 2 + b;
          A[i*{(n+1)*(n+2)//2} + j] = cmat[a][b] * cmat[a2][b2] * cmat[{n} - a - a2][{n} - b - b2] * f2[a + b][a2 + b2] / cmat[{n}][{n}];
          }}
        }}
      }}
    }}

}}

"""

    return ccode


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

    cmat = np.ones((n+1, n+1))
    for p in range(1, n+1):
        for q in range(1, n+1):
            cmat[p, q] = comb(p+q, p)

    for a in range(n + 1):
        for a2 in range(n + 1 - a):
            i = idx(a, a2, n)
            for b in range(n + 1):
                for b2 in range(n + 1 - b):
                    j = idx(b, b2, n)
                    mat[i, j] = cmat[a, b] * cmat[a2, b2] \
                        * cmat[n - a - a2, n - b - b2] \
                        * moments[idx(a + b, a2 + b2, 2*n)]

    mat /= cmat[n, n]
    return mat

print(compute_moments_triangle(5, lambda x,y:x*y, 5))
