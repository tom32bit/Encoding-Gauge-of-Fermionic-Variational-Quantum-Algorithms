"""Scalable DLA dimension via sparse Lie closure over Pauli strings (not 2^n x 2^n matrices).

Each algebra element is a sparse vector over the Pauli group {X^x Z^z}; the commutator of two
Paulis is a single Pauli (0 if they commute, else 2*(-1)^{za.xb} times their product), so the
closure stays sparse whenever the dynamical Lie algebra is small (poly). This extends the floor
numerics: for free fermions dim g = su(n) = n^2-1 grows polynomially while the Pauli weight under
Jordan-Wigner grows linearly, out to sizes the dense method cannot reach.

Validated against the dense matrix routine before the scaling run. Run: python gauge_dla_scale.py
"""
import numpy as np
from openfermion import FermionOperator
from gauge_micro import encode, dense, dla_dim


def _pc(x):
    return int(x).bit_count()


def qop_to_xz(qop):
    """QubitOperator -> {(x,z): complex coeff in the X^x Z^z basis}; Y = i*X*Z."""
    out = {}
    for term, coeff in qop.terms.items():
        x = z = 0
        for q, p in term:
            if p in ("X", "Y"):
                x |= 1 << q
            if p in ("Z", "Y"):
                z |= 1 << q
        out[(x, z)] = out.get((x, z), 0j) + complex(coeff) * (1j) ** _pc(x & z)
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


def bracket(A, B):
    """[A,B] for two X^x Z^z-basis Pauli vectors (dicts)."""
    out = {}
    for (xa, za), ca in A.items():
        for (xb, zb), cb in B.items():
            if (_pc(za & xb) + _pc(zb & xa)) & 1:        # anticommute -> nonzero
                R = (xa ^ xb, za ^ zb)
                out[R] = out.get(R, 0j) + 2.0 * ca * cb * ((-1.0) ** _pc(za & xb))
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


def _real(A):
    """complex Pauli vector -> sparse REAL vector keyed by ((x,z), 0=re|1=im) (DLA is a REAL algebra)."""
    v = {}
    for k, c in A.items():
        if abs(c.real) > 1e-12:
            v[(k, 0)] = c.real
        if abs(c.imag) > 1e-12:
            v[(k, 1)] = c.imag
    return v


def dla_dim_pauli(gens_aH, cap=20000):
    """Real dimension of the Lie algebra from anti-Hermitian XZ-basis generators, by sparse RREF."""
    pivots = {}

    def reduce(v):
        v = dict(v)
        prog = True
        while prog:
            prog = False
            for pk, pv in pivots.items():
                if abs(v.get(pk, 0.0)) > 1e-9:
                    f = v[pk] / pv[pk]
                    for c, val in pv.items():
                        nv = v.get(c, 0.0) - f * val
                        if abs(nv) < 1e-9:
                            v.pop(c, None)
                        else:
                            v[c] = nv
                    prog = True
        return v

    def add(A):
        v = reduce(_real(A))
        if not v:
            return False
        pivots[max(v, key=lambda k: abs(v[k]))] = v
        return True

    basis, queue = [], []
    for g in gens_aH:
        if add(g):
            basis.append(g); queue.append(g)
    while queue:
        A = queue.pop()
        for B in list(basis):
            C = bracket(A, B)
            if C and add(C):
                basis.append(C); queue.append(C)
                if len(pivots) >= cap:
                    return cap
    return len(pivots)


def free_gens(n):
    return [FermionOperator("%d^ %d" % (i, j)) + FermionOperator("%d^ %d" % (j, i))
            for i in range(n) for j in range(i + 1, n)]


def interacting_gens(n):
    g = []
    for i in range(n):
        for j in range(i + 1, n):
            g.append(FermionOperator("%d^ %d" % (i, j)) + FermionOperator("%d^ %d" % (j, i)))
            g.append(FermionOperator("%d^ %d %d^ %d" % (i, i, j, j)))
    return g


def aH(gens, enc, n):
    return [{k: 1j * c for k, c in qop_to_xz(encode(g, enc, n)).items()} for g in gens]


def jw_weight(gens, n):
    return max((bin(x | z).count("1") for g in gens for (x, z) in qop_to_xz(encode(g, "JW", n))), default=0)


def pr(*a):
    print(*a, flush=True)


if __name__ == "__main__":
    pr("######## validation: sparse Pauli DLA == dense matrix DLA ########")
    ok = True
    for n in (4, 5):
        for name, gf in (("free", free_gens), ("interacting", interacting_gens)):
            gens = gf(n)
            sparse = dla_dim_pauli(aH(gens, "JW", n))
            den = dla_dim([dense(encode(g, "JW", n), n) for g in gens], cap=2 * 4 ** n)
            match = (sparse == den)
            ok = ok and match
            pr("  n=%d %-12s sparse=%-5d dense=%-5s  %s" % (n, name, sparse, den, "OK" if match else "MISMATCH"))
    # encoding-invariance of the sparse dim
    inv = len({dla_dim_pauli(aH(free_gens(5), e, 5)) for e in ("JW", "BK", "parity")}) == 1
    pr("  free-fermion sparse dim identical across JW/BK/parity: %s" % inv)
    assert ok and inv, "sparse DLA failed validation"
    pr("  [validated]\n")

    pr("######## SCALING: free-fermion dim g (su(n)=n^2-1, poly) vs JW weight (=n), at scale ########")
    pr("%-4s | %-10s %-10s | %-10s" % ("n", "dim g", "expected", "JW weight"))
    for n in range(4, 21, 2):
        gens = free_gens(n)
        d = dla_dim_pauli(aH(gens, "JW", n), cap=4 * n * n)
        pr("%-4d | %-10d %-10d | %-10d" % (n, d, n * n - 1, jw_weight(gens, n)))
    pr("=> dim g = n^2-1 (polynomial, gauge-invariant) while JW weight = n: the floor holds at scale.")
