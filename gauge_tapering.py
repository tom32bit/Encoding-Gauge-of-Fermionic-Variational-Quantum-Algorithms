"""T-A: faithfulness is the trainability boundary (corrected, exact-dynamics version).

Re-encoding (JW <-> BK <-> parity) is a FAITHFUL *-isomorphism: it preserves the dynamical Lie
algebra and hence the gradient variance exactly.

The barren-plateau variance is governed by the dimension of the algebra ON THE SUBSPACE THE
DYNAMICS EXPLORES. A symmetry-respecting (Hartree-Fock) reference lives in one charge sector, so
that sector's algebra rho_S(g) -- not the full-Fock algebra g -- sets the variance. Therefore:

  * Part 1 (algebra dimensions): g acts reducibly. dim g on the full Fock space is far larger than
    dim rho_S(g) on the half-filling sector (interacting: 66->36 at n=4, 918->400 at n=6); the free
    (Gaussian) family is faithful on the sector, so there it does not drop.

  * Part 2 (variance): with EXACT number-conserving gates (no Trotter leakage),
      - HF reference (respects the symmetry): tapering is a NO-OP, Var ratio = 1.0000;
      - symmetry-BREAKING reference: tapering CHANGES Var, by the dimension ratio
        (n=6: ratio ~ 2.29 ~ 918/400).

So trainability is invariant under BOTH re-encoding and symmetry tapering of a respected symmetry;
it moves only when one changes what the dynamics explores (a symmetry-breaking reference that is
then tapered, or a restricted/equivariant ansatz).

Run: python gauge_tapering.py
"""
import numpy as np
from scipy.linalg import expm
from openfermion import FermionOperator
from gauge_micro import encode, dense, dla_dim, ENC
from gauge_dla_scale import free_gens, interacting_gens, dla_dim_pauli, aH
from gauge_molecules import hf_index

def popcount(x): return bin(x).count("1")
def sector(n, m): return np.array([b for b in range(1 << n) if popcount(b) == m])
def pr(*a): print(*a, flush=True)


def part1_dimensions():
    pr("######## Part 1: dim g on the full Fock space vs the half-filling sector ########\n")
    pr("%-14s %-4s | %-26s | %-12s %-16s" %
       ("family", "n", "dim g across {JW,BK,parity}", "full Fock", "sector (N=n/2)"))
    pr("-" * 80)
    for label, gf in (("free/Gaussian", free_gens), ("interacting", interacting_gens)):
        for n in (4, 6):
            gens = gf(n)
            denc = {e: dla_dim_pauli(aH(gens, e, n), cap=20000) for e in ENC}
            inv = "(invariant)" if len(set(denc.values())) == 1 else "(DIFFER!)"
            idx = sector(n, n // 2)
            full = dla_dim_pauli(aH(gens, "JW", n), cap=20000)
            mats = [dense(encode(g, "JW", n), n)[np.ix_(idx, idx)] for g in gens]
            sec = dla_dim(mats, cap=20000)
            pr("%-14s %-4d | %-13s %-12s | %-12s %-16s" %
               (label, n, str(set(denc.values())), inv, str(full), str(sec)))
        pr("")


def part2_variance(n=6, S=80, eps=1e-3, layers=2, seed=11):
    gens = interacting_gens(n)
    Gmats = [dense(encode(g, "JW", n), n) for g in gens]
    Omat = dense(encode(FermionOperator("0^ 0"), "JW", n), n)
    keep = sector(n, n // 2); mask = np.zeros(1 << n); mask[keep] = 1.0

    def variance(psi0, project):
        rng = np.random.default_rng(seed); ncomp = min(len(gens), 8); comps = [[] for _ in range(ncomp)]
        def Oexp(theta):
            psi = psi0.astype(complex).copy()
            if project: psi = psi * mask; psi /= np.linalg.norm(psi)
            for L in range(layers):
                for k, G in enumerate(Gmats):
                    psi = expm(-1j * theta[L][k] * G) @ psi
                    if project: psi = psi * mask; psi /= np.linalg.norm(psi)
            return float(np.real(np.vdot(psi, Omat @ psi)))
        for _ in range(S):
            th = [[float(rng.uniform(0, 2 * np.pi)) for _ in range(len(gens))] for _ in range(layers)]
            for k in range(ncomp):
                tp = [l[:] for l in th]; tp[0][k] += eps; tm = [l[:] for l in th]; tm[0][k] -= eps
                comps[k].append((Oexp(tp) - Oexp(tm)) / (2 * eps))
        return float(np.mean([np.var(c) for c in comps]))

    psiHF = np.zeros(1 << n, complex); psiHF[hf_index("JW", n, n // 2)] = 1.0
    psiB = np.ones(1 << n, complex); psiB /= np.linalg.norm(psiB)
    pr("######## Part 2: variance under tapering, exact number-conserving gates (n=%d) ########\n" % n)
    hf_u, hf_t = variance(psiHF, False), variance(psiHF, True)
    b_u, b_t = variance(psiB, False), variance(psiB, True)
    pr("HF reference (respects symmetry):  Var untapered=%.4e  tapered=%.4e  ratio=%.4f  -> no-op"
       % (hf_u, hf_t, hf_t / hf_u))
    pr("symmetry-breaking reference:       Var untapered=%.4e  tapered=%.4e  ratio=%.4f  (~ dim_full/dim_sector)"
       % (b_u, b_t, b_t / b_u))


if __name__ == "__main__":
    part1_dimensions()
    part2_variance(n=6)
    pr("\n=> Re-encoding: dim g and variance invariant (faithful). Tapering of the respected symmetry:")
    pr("   a no-op for the variance (ratio 1). Trainability moves only by changing the explored algebra.")
