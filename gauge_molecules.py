"""Real-molecule, proper-Hartree-Fock-reference version of the gauge demonstration.

Loads REAL molecular Hamiltonians (H2, LiH; OpenFermion-bundled STO-3G integrals), builds a
Trotterized Hamiltonian-variational ansatz, and computes the circuit-level absolute-error
truncation weight W* of an observable back-propagated through it, with the proper encoded
Hartree-Fock reference state (a basis state computed per encoding), under JW / BK / parity.

Prediction: W* covariant (JW high -> verdict HARD; BK low -> simulable); DLA dim & magic invariant.
Run: python gauge_molecules.py
"""
import os
import glob
import numpy as np
import openfermion as of
from openfermion import FermionOperator, get_fermion_operator
from gauge_micro import encode, dla_dim, two_sre, dense, ENC
from gauge_pipeline import qop_to_terms
from bpfree.pauli import propagate_circuit, _popcount
from bpfree.statevec import apply_rotation, _pauli_apply

DATA = os.path.join(os.path.dirname(of.__file__), "testing", "data")


def load_mol(pattern):
    fs = sorted(glob.glob(os.path.join(DATA, pattern + "*.hdf5")))
    fs = [f for f in fs if os.path.basename(f)[0] not in "pt"]   # skip phdm_/tqdm_ density-matrix files
    assert fs, "no file matching " + pattern
    f = fs[0][:-5]
    m = of.MolecularData(filename=f)
    m.load()
    H = get_fermion_operator(m.get_molecular_hamiltonian())
    return H, m.n_qubits, m.n_electrons, os.path.basename(f)


def hf_index(enc, n, n_elec):
    """Encoded Hartree-Fock computational-basis state: occupation = modes 0..n_elec-1 occupied."""
    occ = [1] * n_elec + [0] * (n - n_elec)
    nterms = [qop_to_terms(encode(FermionOperator("%d^ %d" % (i, i)), enc, n)) for i in range(n)]
    for b in range(1 << n):
        ok = True
        for i in range(n):
            val = sum(c * (1.0 if not (_popcount(z & b) & 1) else -1.0) for (x, z), c in nterms[i].items() if x == 0)
            if abs(val - occ[i]) > 1e-6:
                ok = False
                break
        if ok:
            return b
    raise RuntimeError("HF basis state not found for %s" % enc)


def hermitian_gens(H):
    """Hermitian fermionic generators of H (term + its conjugate), deduped -> a real molecular HVA."""
    gens, seen = [], set()
    for term in H.terms:
        if not term:
            continue
        td = list(of.hermitian_conjugated(FermionOperator(term)).terms.keys())[0]
        key = tuple(sorted([term, td]))
        if key in seen:
            continue
        seen.add(key)
        gens.append(FermionOperator(term) + FermionOperator(td))
    return gens


def circuit(enc, gens, n, angles):
    """Expressive HVA: independent per-generator angle per layer, SAME across encodings (physical)."""
    gates = []
    for layer in angles:
        for theta, g in zip(layer, gens):
            for (x, z), c in qop_to_terms(encode(g, enc, n)).items():
                if (x, z) != (0, 0):
                    gates.append((x, z, theta * c))
    return gates


def exp_basis(terms, b):
    return float(sum(np.real(c) * (1.0 if not (_popcount(z & b) & 1) else -1.0)
                     for (x, z), c in terms.items() if x == 0))


def exact_O(n, gates, obs_terms, b):
    psi = np.zeros(1 << n, dtype=complex)
    psi[b] = 1.0
    for (x, z, th) in gates:
        psi = apply_rotation(psi, x, z, th, n)
    val = 0j
    for (x, z), a in obs_terms.items():
        val += a * ((1j) ** _popcount(x & z)) * np.vdot(psi, _pauli_apply(psi, x, z, n))
    return val.real


def wstar_hf(n, gates, obs_terms, b, eps=0.05):
    exact = exact_O(n, gates, obs_terms, b)
    for W in range(0, n + 1):
        approx = exp_basis(propagate_circuit(n, gates, obs_terms, w_max=W, delta=0.0), b)
        if abs(approx - exact) < eps:
            return W, exact
    return n + 1, exact


def pr(*a):
    print(*a, flush=True)


def run(name, pattern, layers=2, full=True, amp=1.2):
    H, n, ne, fname = load_mol(pattern)
    gens = hermitian_gens(H)
    pr("\n=== %s : %s  (n=%d qubits, %d electrons, %d HVA generators, %d layers) ===" %
       (name, fname, n, ne, len(gens), layers))
    obs_mode = ne - 1                       # a frontier (HOMO) number operator
    angles = [list(np.random.default_rng(100 + L).uniform(0.3, amp, size=len(gens))) for L in range(layers)]
    thr = 2 * np.log2(n)
    pr("%-8s %6s %-30s %-9s %-9s" % ("enc", "HFidx", "W* (thr=%.2f)" % thr, "DLAdim", "2-SRE"))
    for enc in ENC:
        b = hf_index(enc, n, n_elec=ne)
        gates = circuit(enc, gens, n, angles)
        obs = qop_to_terms(encode(FermionOperator("%d^ %d" % (obs_mode, obs_mode)), enc, n))
        W, ex = wstar_hf(n, gates, obs, b)
        wtxt = (">%d" % n) if W > n else str(W)
        verdict = "HARD" if W > thr else "simulable"
        if full and n <= 4:
            d = dla_dim([dense(encode(g, enc, n), n) for g in gens], cap=400)
            psi = np.zeros(1 << n, dtype=complex); psi[b] = 1.0
            for (x, z, th) in gates:
                psi = apply_rotation(psi, x, z, th, n)
            sre = "%.5f" % two_sre(psi, n)
        else:
            d, sre = "(inv.)", "(inv.)"
        pr("%-8s %6d %-30s %-9s %-9s" %
           (enc, b, "%-6s -> %-9s (<O>=%+.3f, %dg)" % (wtxt, verdict, ex, len(gates)), str(d), sre))


if __name__ == "__main__":
    pr("######## REAL MOLECULES: circuit-level W* with proper HF reference, JW/BK/parity ########")
    run("H2 (STO-3G, invariance check)", "H2_sto-3g_singlet_0.7", layers=2, full=True)
    run("H2 (6-31G, the flip)", "H2_6-31g_singlet_0.75", layers=2, full=False)
    pr("\nLook for: JW W* > threshold (HARD) but BK W* <= threshold (simulable) => FLIP;")
    pr("          DLA dim & 2-SRE identical across JW/BK/parity (invariant); <O> ~ encoding-invariant.")
