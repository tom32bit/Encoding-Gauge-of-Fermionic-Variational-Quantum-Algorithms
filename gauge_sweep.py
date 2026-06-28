"""Operating-point sweep: locate the W* flip window for a real molecule (H2/6-31g, n=8).

W* depends on the ansatz operating point (depth x angle). We sweep the angle amplitude and
show that the JW truncation-weight curve dominates BK/parity and crosses the 'hard' threshold
at a SMALLER amplitude than BK -> there is an operating window where the real fermionic VQE is
JW-HARD but BK-SIMULABLE (the gauge flip), while <O> stays encoding-invariant.
"""
import numpy as np
from openfermion import FermionOperator
from gauge_micro import encode, ENC
from gauge_molecules import load_mol, hermitian_gens, circuit, hf_index, wstar_hf, qop_to_terms, pr

if __name__ == "__main__":
    H, n, ne, fname = load_mol("H2_6-31g_singlet_0.75")
    gens = hermitian_gens(H)
    thr = 2 * np.log2(n)
    pr("=== %s  n=%d, %d electrons, %d generators, threshold(hard)=%.2f ===" % (fname, n, ne, len(gens), thr))
    pr("%-6s %-8s | %-10s %-10s %-10s | %s" % ("amp", "layers", "JW W*", "BK W*", "parity W*", "flip? (<O> inv.)"))
    for layers in (1,):
        for amp in (0.3, 0.5, 0.7, 0.9, 1.1, 1.4):
            angles = [list(np.random.default_rng(100 + L).uniform(0.15, amp, size=len(gens))) for L in range(layers)]
            row, Os = {}, []
            for enc in ENC:
                b = hf_index(enc, n, n_elec=ne)
                gates = circuit(enc, gens, n, angles)
                obs = qop_to_terms(encode(FermionOperator("%d^ %d" % (ne - 1, ne - 1)), enc, n))
                W, ex = wstar_hf(n, gates, obs, b)
                row[enc] = W
                Os.append(ex)
            flip = (row["JW"] > thr) and (row["BK"] <= thr)
            oinv = (max(Os) - min(Os) < 1e-3)
            pr("%-6.2f %-8d | %-10s %-10s %-10s | %s  (<O>=%+.3f%s)" % (
                amp, layers,
                "%d%s" % (row["JW"], " HARD" if row["JW"] > thr else " sim"),
                "%d%s" % (row["BK"], " HARD" if row["BK"] > thr else " sim"),
                "%d%s" % (row["parity"], " HARD" if row["parity"] > thr else " sim"),
                "*** FLIP ***" if flip else "no flip",
                Os[0], "" if oinv else " (!inv)"))
