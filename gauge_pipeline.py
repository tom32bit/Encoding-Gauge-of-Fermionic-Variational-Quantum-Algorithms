"""Circuit-level upgrade of the gauge-relativity demonstration.

Replaces the Hamiltonian-max-weight PROXY with the genuine absolute-error truncation
weight W* of the back-propagated observable through an actual (Trotterized) encoded
ansatz circuit, computed with the project's own Pauli-propagation engine.

For a NON-local (chemistry-like all-to-all) fermionic ansatz, under JW / BK / parity:
  - W*        (circuit-level truncation weight)  -> COVARIANT (JW high, BK low) : the verdict flips
  - DLA dim   (matrix Lie closure of encoded gens) -> INVARIANT
  - magic 2-SRE (of the encoded ground state)       -> INVARIANT
All four resources computed end-to-end; |0> reference (the encoded Pauli-rotation circuit
acts non-trivially on |0>). Run: python gauge_pipeline.py
"""
import numpy as np
from openfermion import FermionOperator
from gauge_micro import encode, dla_dim, two_sre, dense
from gauge_engine.pauli import propagate_circuit, expectation_zero

ENC = ["JW", "BK", "parity"]


def qop_to_terms(qop):
    out = {}
    for term, coeff in qop.terms.items():
        x = z = 0
        for q, p in term:
            if p in ("X", "Y"):
                x |= 1 << q
            if p in ("Z", "Y"):
                z |= 1 << q
        out[(x, z)] = out.get((x, z), 0j) + complex(coeff)
    return {k: v.real for k, v in out.items() if abs(v) > 1e-12}


def molecular(N, seed=1):
    """all-to-all 2-body fermionic Hamiltonian; returns hermitian generators + full H."""
    rng = np.random.default_rng(seed)
    gens, H = [], FermionOperator()
    for p in range(N):
        for q in range(p + 1, N):
            hop = FermionOperator("%d^ %d" % (p, q)) + FermionOperator("%d^ %d" % (q, p))
            nn = FermionOperator("%d^ %d %d^ %d" % (p, p, q, q))
            gens += [hop, nn]
            H += rng.normal() * hop + rng.normal() * nn
    for m in range(N):
        H += 0.1 * (m - (N - 1) / 2.0) * FermionOperator("%d^ %d" % (m, m))
    return gens, H


def sample_angles(n_gens, layers, seed=7):
    rng = np.random.default_rng(seed)
    return [[float(rng.uniform(0.2, 0.8)) for _ in range(n_gens)] for _ in range(layers)]


def build_circuit(gens, enc, n, angles):
    """Trotterized encoded HVA: SAME per-generator angle across encodings (same physical ansatz)."""
    gates = []
    for layer in angles:
        for theta, g in zip(layer, gens):
            for (x, z), c in qop_to_terms(encode(g, enc, n)).items():
                gates.append((x, z, theta * c))
    return gates


def exp0(terms):
    return float(np.real(expectation_zero(terms)))


def wstar(n, gates, obs_terms, eps=0.05):
    exact = exp0(propagate_circuit(n, gates, obs_terms, w_max=n, delta=0.0))   # w_max=n -> no truncation -> exact
    for W in range(0, n + 1):
        approx = exp0(propagate_circuit(n, gates, obs_terms, w_max=W, delta=0.0))
        if abs(approx - exact) < eps:
            return W, exact, len(gates)
    return n + 1, exact, len(gates)   # never converged below n -> report as > n


def ground_state(Hq, n):
    M = dense(Hq, n)
    _, V = np.linalg.eigh(M)
    return V[:, 0]


def pr(*a):
    print(*a, flush=True)


if __name__ == "__main__":
    pr("######## END-TO-END: circuit-level W*, DLA, magic under JW/BK/parity ########")
    pr("Non-local molecular-like ansatz; obs = number operator n_mid; |0> reference.")
    pr("(DLA & magic exact at N=6; their INVARIANCE already confirmed in gauge_micro.)\n")
    hdr = "%-4s %7s | %-26s | %-9s | %-10s"
    pr(hdr % ("N", "enc", "W* (truncation weight)", "DLA dim", "magic SRE"))
    for N in (6, 8):
        gens, H = molecular(N)
        mid = N // 2
        angles = sample_angles(len(gens), layers=2)
        thr = 2 * np.log2(N)
        for enc in ENC:
            gates = build_circuit(gens, enc, N, angles)
            obs = qop_to_terms(encode(FermionOperator("%d^ %d" % (mid, mid)), enc, N))
            W, ex, ng = wstar(N, gates, obs)
            wtxt = (">%d" % N) if W > N else str(W)
            verdict = "HARD" if W > thr else "simulable"
            if N <= 6:
                d = dla_dim([dense(encode(g, enc, N), N) for g in gens], cap=400)
                sre = "%.5f" % two_sre(ground_state(encode(H, enc, N), N), N)
            else:
                d, sre = "(inv.)", "(inv.)"
            pr(hdr % (str(N), enc, "%-5s (thr=%.1f -> %-9s)" % (wtxt, thr, verdict), str(d), sre))
        pr("")

    pr("######## what to look for ########")
    pr("W* COVARIANT: JW W* > threshold (HARD) but BK W* <= threshold (simulable) => VERDICT FLIPS.")
    pr("DLA dim & magic SRE INVARIANT: identical across JW/BK/parity.")
