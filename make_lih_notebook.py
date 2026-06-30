"""Generate lih_chemistry_kaggle.ipynb: an HONEST test for a molecular W* verdict flip.

LiH/STO-3G (n=12), standard UCCSD ansatz. We sweep the operating point (angle amplitude) and
seeds, and at each point test the verdict by truncating the back-propagated observable at
W = floor(2 log2 n) = 7: error > eps => that encoding is HARD for truncated Pauli propagation.
A flip is JW HARD AND BK simulable, at a point where <O> is encoding-invariant. The full sweep is
reported so the flip's robustness (or absence) is visible -- no cherry-picking, threshold fixed.
"""
import json
import make_notebook as MN   # reuse the validated, inlined engine + self-validation

LIH = r'''
# ===================== LiH/STO-3G (n=12): an honest molecular W* flip test =====================
import numpy as np, time
from joblib import Parallel, delayed

def uccsd_gens(n, ne):
    """Standard UCCSD Hermitian generators: singles + doubles (occupied <-> virtual)."""
    occ, virt = list(range(ne)), list(range(ne, n)); g = []
    for i in occ:
        for a in virt:
            g.append(FermionOperator("%d^ %d" % (a, i)) + FermionOperator("%d^ %d" % (i, a)))
    for ii in range(len(occ)):
        for jj in range(ii + 1, len(occ)):
            for aa in range(len(virt)):
                for bb in range(aa + 1, len(virt)):
                    i, j, a, b = occ[ii], occ[jj], virt[aa], virt[bb]
                    g.append(FermionOperator("%d^ %d^ %d %d" % (a, b, j, i))
                             + FermionOperator("%d^ %d^ %d %d" % (i, j, b, a)))
    return g

m, H = load_mol("H1-Li1_sto-3g_singlet_1.45")
n, ne = m.n_qubits, m.n_electrons
thr = 2 * np.log2(n); Wchk = int(np.floor(thr)); obs_mode = ne - 1; eps = 0.05
gens = uccsd_gens(n, ne)
print("LiH/STO-3G: n=%d qubits, %d electrons, %d UCCSD generators, threshold=%.2f (verdict at W=%d), eps=%.2f"
      % (n, ne, len(gens), thr, Wchk, eps), flush=True)

# --- ground-truth check: encoded Hartree-Fock energy matches the stored molecular value ---
for e in ENC:
    hf = exp_basis(qop_to_terms(encode(H, e, n)), hf_index(e, n, ne))
    print("  [HF check %-7s] <HF|H|HF>=%.6f  stored=%.6f  |err|=%.1e" % (e, hf, m.hf_energy, abs(hf - m.hf_energy)), flush=True)

def verdict_point(args):
    _ofpatch(); enc, amp, seed, layers = args
    g = uccsd_gens(n, ne)
    ang = [list(np.random.default_rng(seed + L).uniform(0.15, amp, size=len(g))) for L in range(layers)]
    b = hf_index(enc, n, ne); gates = circuit(enc, g, n, ang)
    obs = qop_to_terms(encode(FermionOperator("%d^ %d" % (obs_mode, obs_mode)), enc, n))
    ex = exact_O(n, gates, obs, b)
    err = abs(exp_basis(propagate_circuit(n, gates, obs, w_max=Wchk), b) - ex)
    return (enc, amp, seed, layers, ex, err)

# --- the sweep: operating point x seed, all three encodings, verdict at the fixed threshold ---
AMPS = [0.5, 0.7, 0.9, 1.1, 1.3]
SEEDS = [100, 200]
LAYERS = 1
tasks = [(e, a, s, LAYERS) for a in AMPS for s in SEEDS for e in ENC]
t0 = time.time()
res = Parallel(n_jobs=-1)(delayed(verdict_point)(t) for t in tasks)
print("\nsweep done in %.0f s\n" % (time.time() - t0), flush=True)
R = {(e, a, s): (ex, err) for (e, a, s, L, ex, err) in res}

print("amp   seed | <O> spread | JW err  verdict | BK err  verdict | par err verdict | FLIP?")
nflip = 0
for a in AMPS:
    for s in SEEDS:
        os_ = [R[(e, a, s)][0] for e in ENC]; spread = max(os_) - min(os_)
        v = {e: ("HARD" if R[(e, a, s)][1] > eps else "sim ") for e in ENC}
        flip = (v["JW"] == "HARD" and v["BK"] == "sim " and spread < 1e-3)
        nflip += int(flip)
        print("%-5.2f %-4d | %.1e   | %.4f %s | %.4f %s | %.4f %s | %s"
              % (a, s, spread, R[("JW", a, s)][1], v["JW"], R[("BK", a, s)][1], v["BK"],
                 R[("parity", a, s)][1], v["parity"], "*** FLIP ***" if flip else ""))
print("\nflip points: %d / %d (JW HARD, BK simulable, <O> invariant). Threshold fixed at 2 log2 n." % (nflip, len(AMPS) * len(SEEDS)))

# --- if a flip is robust, report the actual integer W* at a representative point ---
def wstar_point(args):
    _ofpatch(); enc, amp, seed, layers = args
    g = uccsd_gens(n, ne)
    ang = [list(np.random.default_rng(seed + L).uniform(0.15, amp, size=len(g))) for L in range(layers)]
    b = hf_index(enc, n, ne); gates = circuit(enc, g, n, ang)
    obs = qop_to_terms(encode(FermionOperator("%d^ %d" % (obs_mode, obs_mode)), enc, n))
    ex = exact_O(n, gates, obs, b)
    Wcap = min(n, Wchk + 3)                       # bound the expensive high-W propagation for HARD encodings
    for W in range(0, Wcap + 1):
        if abs(exp_basis(propagate_circuit(n, gates, obs, w_max=W), b) - ex) < eps:
            return (enc, W, ex)
    return (enc, Wcap + 1, ex)                     # ">Wcap" : still HARD past the cap

if nflip > 0:
    bestamp = 0.9 if 0.9 in AMPS else AMPS[len(AMPS) // 2]
    print("\nfull integer W* at amp=%.2f, seed=%d (representative):" % (bestamp, SEEDS[0]))
    ws = Parallel(n_jobs=-1)(delayed(wstar_point)((e, bestamp, SEEDS[0], LAYERS)) for e in ENC)
    for (e, W, ex) in ws:
        print("  %-7s W*=%-4s (%s)  <O>=%+.4f" % (e, ("%d" % W if W <= n else ">%d" % n), "HARD" if W > thr else "simulable", ex))
print("\n=> A flip here is a real molecular instance where the simulability verdict is encoding-relative;")
print("   no flip => the H2/LiH molecular data remain an invariance check, as the paper already states.")
'''

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                           "source": s.strip("\n").splitlines(keepends=True)})

md("""# LiH chemistry test: is the W* verdict flip reachable for a real molecule?

Honest, pre-committed test. We take **LiH/STO-3G (n=12)** with a standard **UCCSD** ansatz and ask
whether there is an operating point where the *same* physical circuit is Pauli-propagation-HARD under
Jordan-Wigner but classically simulable under Bravyi-Kitaev (a verdict flip), with the physical
expectation value identical across encodings. The threshold is fixed at 2 log2 n; the full
operating-point x seed sweep is reported, so a flip (or its absence) is visible without cherry-picking.
Self-validating: the engine is checked against independent ground truth before the test.
""")
code('print(">>> LiH chemistry flip test <<<")\n!pip install -q openfermion joblib')
code(MN.ENGINE)
md("## Step 1 - certify the engine (independent ground truth)")
code(MN.VALIDATE)
md("## Step 2 - LiH/STO-3G UCCSD: operating-point sweep and the verdict flip test")
code(LIH)
md("""## Reading the result
- **If FLIP points appear** (JW HARD, BK simulable, `<O>` spread < 1e-3): a real molecule exhibits the
  encoding-relative verdict -- promote the chemistry from a check to a demonstration.
- **If no flip**: the molecular instances stay an invariance check, exactly as the paper states; the
  verdict-flip demonstration remains the synthetic structured family. Either outcome is reported as-is.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
with open("lih_chemistry_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote lih_chemistry_kaggle.ipynb with", len(cells), "cells")
