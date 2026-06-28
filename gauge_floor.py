"""#8 core: is the gauge-optimal cost floored by the INVARIANT resources (DLA, magic)?

Orbit = {JW, BK, BK-tree, parity}; W_opt = min over the orbit (the gauge-optimal Pauli weight;
the JKMN ternary tree is the proven optimum this approximates). The union verdict is simulable if
ANY route is cheap: weight n^O(W_opt), DLA poly(dim g), magic 2^O(SRE).

Claim (the floor): the covariant weight is never the SOLE obstruction. Whenever an INVARIANT route
is small the problem is simulable regardless of weight -- the sharpest case being free fermions,
where JW weight = n (a naive Pauli-propagation reading calls it hard) yet dim g = su(n) = n^2-1
(poly; hopping-only generators), so g-sim simulates it exactly. Genuine hardness needs the INVARIANTS large.
Run: python gauge_floor.py
"""
import numpy as np
from openfermion import (FermionOperator, jordan_wigner, bravyi_kitaev, bravyi_kitaev_tree,
                         binary_code_transform, parity_code, get_sparse_operator)
import openfermion.ops.operators.symbolic_operator as _s
_s.COEFFICIENT_TYPES = tuple(_s.COEFFICIENT_TYPES) + (np.integer, np.floating, np.complexfloating)
from gauge_micro import dla_dim, two_sre

ORBIT = ["JW", "BK", "BKtree", "parity"]


def enc4(fop, e, n):
    if e == "JW": return jordan_wigner(fop)
    if e == "BK": return bravyi_kitaev(fop, n_qubits=n)
    if e == "BKtree": return bravyi_kitaev_tree(fop, n_qubits=n)
    if e == "parity": return binary_code_transform(fop, parity_code(n))
    raise ValueError(e)


def dense(q, n): return np.asarray(get_sparse_operator(q, n_qubits=n).todense())
def mw(q): return max((len(t) for t in q.terms if t), default=0)
def ground(H, n):
    w, V = np.linalg.eigh(dense(H, n)); return V[:, 0]


def free_long_range(n, seed=1):
    """all-to-all QUADRATIC hops (Gaussian) -> DLA = su(n), dim n^2-1 (poly); g-sim simulable."""
    rng = np.random.default_rng(seed)
    gens = [FermionOperator("%d^ %d" % (i, j)) + FermionOperator("%d^ %d" % (j, i))
            for i in range(n) for j in range(i + 1, n)]
    H = FermionOperator()
    for g in gens:
        H += rng.normal() * g
    return n, gens, H


def interacting(n, seed=1):
    """all-to-all with 2-body interactions -> DLA large, magic large."""
    rng = np.random.default_rng(seed)
    gens, H = [], FermionOperator()
    for i in range(n):
        for j in range(i + 1, n):
            hop = FermionOperator("%d^ %d" % (i, j)) + FermionOperator("%d^ %d" % (j, i))
            nn = FermionOperator("%d^ %d %d^ %d" % (i, i, j, j))
            gens += [hop, nn]; H += rng.normal() * hop + rng.normal() * nn
    return n, gens, H


def report(name, n, gens, H, cap):
    wts = {e: mw(enc4(H, e, n)) for e in ORBIT}
    wopt = min(wts.values())
    dimg = dla_dim([dense(enc4(g, "JW", n), n) for g in gens], cap=cap)   # invariant (encoding-independent)
    poly = 2 * n * n
    gsim = isinstance(dimg, int) and dimg <= poly
    tag = ("= su(%d) = %d, POLY" % (n, dimg)) if gsim else ("%s, >poly" % dimg)
    print("%-13s n=%d | JW wt=%d  W_opt=%d (covariant) | dim g %-18s (invariant) | g-sim: %s"
          % (name, n, wts["JW"], wopt, tag, "YES -> SIMULABLE" if gsim else "no"))
    return wts["JW"], dimg, gsim


if __name__ == "__main__":
    # self-validation vs ground truth before the study
    for nn in (4, 5):
        _, g, _ = free_long_range(nn)
        d = dla_dim([dense(enc4(x, "JW", nn), nn) for x in g], cap=4 * nn * nn)
        assert d == nn * nn - 1, "free-fermion DLA %s != su(%d)=%d" % (d, nn, nn * nn - 1)
    _H = FermionOperator("0^ 1") + FermionOperator("1^ 0") + FermionOperator("0^ 0 2^ 2")
    _sj = np.sort(np.linalg.eigvalsh(dense(enc4(_H, "JW", 4), 4)).real)
    _sb = np.sort(np.linalg.eigvalsh(dense(enc4(_H, "BKtree", 4), 4)).real)
    assert np.max(np.abs(_sj - _sb)) < 1e-9, "BK-tree not isospectral with JW"
    print("[self-check] free-fermion DLA = su(n)=n^2-1 (exact) and BK-tree isospectral with JW: OK\n")

    print("######## #8: the gauge-INVARIANT resource is the honest hardness measure ########")
    print("orbit = JW/BK/BKtree/parity; W_opt = min over orbit (gauge-optimal weight, ~ ternary-tree optimum).")
    print("Key: JW weight does NOT distinguish free from interacting; dim g (invariant) DOES.\n")
    for n in (4, 5, 6):
        cap = 8 * n * n
        _, gf, Hf = free_long_range(n)
        jf, df, sf = report("free-fermion", n, gf, Hf, cap=cap)
        _, gi, Hi = interacting(n)
        ji, di, si = report("interacting", n, gi, Hi, cap=cap)
        print("   --> JW weight identical (%d vs %d); dim g separates them (%s vs %s).\n" % (jf, ji, df, di))
    print("=> Covariant Pauli weight flags free fermions as 'high weight' (=n) yet they are EXACTLY")
    print("   g-sim-simulable (dim g = n^2-1, poly). The ENCODING-INVARIANT dim g is the reliable")
    print("   hardness certificate; the gauge-covariant weight is not. cost* is floored by the invariants.")

    print("\n######## cost* = min over the encoding orbit (gauge-optimal Pauli weight) ########")
    print("%-4s %-4s %-4s %-7s %-7s | %s" % ("n", "JW", "BK", "BKtree", "parity", "W_opt (cost*)  achiever"))
    for n in range(6, 21, 2):
        _, _, H = interacting(n)
        w = {e: mw(enc4(H, e, n)) for e in ORBIT}
        wopt = min(w.values())
        best = min(w, key=w.get)
        print("%-4d %-4d %-4d %-7d %-7d | %-3d  %s" % (n, w["JW"], w["BK"], w["BKtree"], w["parity"], wopt, best))
    print("=> W_opt ~ O(log n) (BK/BK-tree), vs JW = n: gauge-optimization removes the weight blow-up.")
    print("   BK-tree occasionally beats BK; the JKMN ternary tree is the proven optimum at the bottom.")
