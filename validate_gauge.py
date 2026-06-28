"""Independent-ground-truth validation of every component the gauge results rest on.
Each check compares against an EXTERNAL known value (not internal consistency). Run: python validate_gauge.py
"""
import os
import glob
import numpy as np
import openfermion as of
from openfermion import FermionOperator, get_fermion_operator
from gauge_micro import encode, dense, dla_dim, two_sre, free_fermion_chain, ENC
from gauge_molecules import (load_mol, hf_index, hermitian_gens, circuit, exact_O,
                             exp_basis, wstar_hf, qop_to_terms, DATA)
from bpfree.pauli import propagate_circuit

RESULTS = []


def check(name, passed, detail):
    RESULTS.append(passed)
    print("[%s] %-42s %s" % ("PASS" if passed else "FAIL", name, detail), flush=True)


def load_full(pattern):
    fs = sorted(glob.glob(os.path.join(DATA, pattern + "*.hdf5")))
    fs = [f for f in fs if os.path.basename(f)[0] not in "pt"]
    m = of.MolecularData(filename=fs[0][:-5]); m.load()
    return m, get_fermion_operator(m.get_molecular_hamiltonian())


# ---- 1. Encodings are ISOSPECTRAL (same physical operator) -----------------------------------
def c_isospectral():
    _, H = load_full("H2_sto-3g_singlet_0.7")
    n = 4
    sp = {e: np.sort(np.linalg.eigvalsh(dense(encode(H, e, n), n)).real) for e in ENC}
    d_jb = np.max(np.abs(sp["JW"] - sp["BK"]))
    d_jp = np.max(np.abs(sp["JW"] - sp["parity"]))
    check("encodings isospectral (H2)", d_jb < 1e-9 and d_jp < 1e-9,
          "max|dEig| JW-BK=%.1e JW-parity=%.1e (ground=%.6f)" % (d_jb, d_jp, sp["JW"][0]))


# ---- 2. <HF|H|HF> == molecule's stored HF energy, for ALL encodings --------------------------
def c_hf_energy():
    for mol, n in (("H2_sto-3g_singlet_0.7", 4), ("H1-Li1_sto-3g_singlet_1.45", 12)):
        m, H = load_full(mol)
        errs = []
        for e in ENC:
            b = hf_index(e, n, m.n_electrons)
            E = exp_basis(qop_to_terms(encode(H, e, n)), b)
            errs.append(abs(E - m.hf_energy))
        check("HF energy <HF|H|HF>==mol.hf_energy (%s)" % mol.split("_")[0],
              max(errs) < 1e-6, "mol.hf_energy=%.6f, max|err| across JW/BK/parity=%.1e" % (m.hf_energy, max(errs)))


# ---- 3. HF state reproduces the right occupations under each encoding -------------------------
def c_hf_occupation():
    m, _ = load_full("H2_sto-3g_singlet_0.7")
    n, ne = 4, m.n_electrons
    occ_target = [1] * ne + [0] * (n - ne)
    ok = True
    for e in ENC:
        b = hf_index(e, n, ne)
        for i in range(n):
            ni = exp_basis(qop_to_terms(encode(FermionOperator("%d^ %d" % (i, i)), e, n)), b)
            if abs(ni - occ_target[i]) > 1e-9:
                ok = False
    check("HF state gives correct occupations", ok, "modes 0..%d occupied, all encodings" % (ne - 1))


# ---- 4. Heisenberg propagate (w_max=n, exact) == independent statevector <O> ------------------
def c_engine_agree():
    # real molecule H2 (n=4) and synthetic non-local (n=6)
    cases = []
    m, H = load_full("H2_sto-3g_singlet_0.7")
    cases.append(("H2", H, 4, m.n_electrons))
    from gauge_pipeline import molecular
    _, Hs = molecular(6)
    cases.append(("synthetic n=6", Hs, 6, 3))
    worst = 0.0
    for label, H, n, ne in cases:
        gens = hermitian_gens(H)
        ang = [list(np.random.default_rng(1).uniform(0.3, 0.9, size=len(gens)))]
        for e in ENC:
            b = hf_index(e, n, ne)
            g = circuit(e, gens, n, ang)
            obs = qop_to_terms(encode(FermionOperator("%d^ %d" % (ne - 1, ne - 1)), e, n))
            heis = exp_basis(propagate_circuit(n, g, obs, w_max=n, delta=0.0), b)
            schr = exact_O(n, g, obs, b)
            worst = max(worst, abs(heis - schr))
    check("Heisenberg(w_max=n) == statevector <O>", worst < 1e-7, "max|Heis-Schro|=%.1e (W* is well-defined)" % worst)


# ---- 5. <O> is identical across encodings (physics gauge-invariant) ---------------------------
def c_obs_invariant():
    from gauge_pipeline import molecular
    _, H = molecular(6)
    n, ne = 6, 3
    gens = hermitian_gens(H)
    ang = [list(np.random.default_rng(2).uniform(0.3, 0.9, size=len(gens)))]
    vals = []
    for e in ENC:
        b = hf_index(e, n, ne)
        g = circuit(e, gens, n, ang)
        obs = qop_to_terms(encode(FermionOperator("%d^ %d" % (ne - 1, ne - 1)), e, n))
        vals.append(exact_O(n, g, obs, b))
    check("observable <O> invariant across encodings", max(vals) - min(vals) < 1e-9,
          "<O> = %s (spread %.1e)" % (["%.6f" % v for v in vals], max(vals) - min(vals)))


# ---- 6. DLA dim == known u(L) = L^2 for free fermions, and encoding-invariant -----------------
def c_dla_known():
    ok, detail = True, []
    for L in (4, 5):
        n, gens, _ = free_fermion_chain(L)
        dims = [dla_dim([dense(encode(gn, e, n), n) for gn in gens], cap=4 * L * L) for e in ENC]
        detail.append("L=%d: %s (u(%d)=%d)" % (L, dims, L, L * L))
        if not all(d == L * L for d in dims):
            ok = False
    check("DLA == u(L) dim L^2, all encodings", ok, "; ".join(detail))


# ---- 7. magic two_sre: 0 for stabilizer, analytic 0.415 for T-state --------------------------
def c_magic_known():
    n = 1
    zero = np.array([1.0, 0.0], dtype=complex)
    t = np.array([1.0, np.exp(1j * np.pi / 4)], dtype=complex) / np.sqrt(2)   # T|+>
    s0 = two_sre(zero, n)
    st = two_sre(t, n)
    analytic = np.log2(4.0 / 3.0)
    check("magic two_sre (stabilizer=0, T-state analytic)", abs(s0) < 1e-9 and abs(st - analytic) < 1e-6,
          "|0>=%.6f (exp 0); T|+>=%.6f (exp %.6f)" % (s0, st, analytic))


# ---- 8. weight covariance direction: long-range hop JW~n, BK<n --------------------------------
def c_weight_cov():
    detail, ok = [], True
    for n in (8, 16):
        g = FermionOperator("0^ %d" % (n - 1)) + FermionOperator("%d^ 0" % (n - 1))
        wjw = max(len(t) for t in encode(g, "JW", n).terms if t)
        wbk = max(len(t) for t in encode(g, "BK", n).terms if t)
        detail.append("n=%d: JW=%d BK=%d" % (n, wjw, wbk))
        if not (wjw >= n - 1 and wbk < wjw):
            ok = False
    check("weight covariant (long-range hop JW~n > BK)", ok, "; ".join(detail))


if __name__ == "__main__":
    print("######## INDEPENDENT-GROUND-TRUTH VALIDATION ########\n")
    for fn in (c_isospectral, c_hf_energy, c_hf_occupation, c_engine_agree,
               c_obs_invariant, c_dla_known, c_magic_known, c_weight_cov):
        try:
            fn()
        except Exception as ex:
            check(fn.__name__, False, "EXCEPTION: %r" % ex)
    print("\n######## %d/%d checks PASSED ########" % (sum(RESULTS), len(RESULTS)))
