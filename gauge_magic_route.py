"""(2b) The MAGIC route of the floor: weight & dim g large, yet magic=0 -> stabilizer-simulable.

Complements the DLA route (gauge_floor.py: free fermions, dim g poly). Here we take the SAME
high-weight, high-dim-g interacting ansatz and evaluate it at CLIFFORD angles (each Pauli rotation
= pi/4, which is a Clifford gate). The output from the stabilizer reference |HF> is then a stabilizer
state, so its 2-SRE (magic) = 0 and it is exactly stabilizer-simulable (Gottesman-Knill) -- even
though the Pauli weight and the DLA dimension are both large. Magic is encoding-invariant, so this
certificate holds in every encoding. Run: python gauge_magic_route.py
"""
import numpy as np
from openfermion import FermionOperator
from gauge_micro import encode, two_sre, dla_dim, dense, ENC
from gauge_molecules import hf_index, qop_to_terms, circuit
from gauge_engine.statevec import apply_rotation


def interacting_gens(n):
    g = []
    for i in range(n):
        for j in range(i + 1, n):
            g.append(FermionOperator("%d^ %d" % (i, j)) + FermionOperator("%d^ %d" % (j, i)))
            g.append(FermionOperator("%d^ %d %d^ %d" % (i, i, j, j)))
    return g


def paulis_of(enc, gens, n):
    P = []
    for g in gens:
        for (x, z), c in qop_to_terms(encode(g, enc, n)).items():
            if (x, z) != (0, 0):
                P.append((x, z))
    return P


def state(gates, n, b):
    psi = np.zeros(1 << n, dtype=complex); psi[b] = 1.0
    for (x, z, th) in gates:
        psi = apply_rotation(psi, x, z, th, n)
    return psi


def maxw(enc, gens, n):
    return max((bin(x | z).count("1") for g in gens for (x, z) in qop_to_terms(encode(g, enc, n))
                if (x, z) != (0, 0)), default=0)


if __name__ == "__main__":
    n = 4
    gens = interacting_gens(n)
    ne = n // 2
    rng = np.random.default_rng(2)
    dimg = dla_dim([dense(encode(g, "JW", n), n) for g in gens], cap=8 * n * n)
    print("Interacting ansatz, n=%d: JW max weight=%d, dim g=%s (both LARGE -> weight & DLA routes do NOT help).\n"
          % (n, maxw("JW", gens, n), dimg))
    gen_angles = [[float(rng.uniform(0.3, 1.2)) for _ in gens]]    # per-GENERATOR, same fermionic ansatz
    print("%-7s | SRE @ Clifford (pi/4)   SRE @ generic angles (same fermionic ansatz)" % "enc")
    for enc in ENC:
        b = hf_index(enc, n, ne)
        cliff = [(x, z, np.pi / 4) for (x, z) in paulis_of(enc, gens, n)]
        sc = two_sre(state(cliff, n, b), n)
        sg = two_sre(state(circuit(enc, gens, n, gen_angles), n, b), n)
        print("%-7s | %.6f                %.6f" % (enc, sc, sg))
    print("\n=> At Clifford angles SRE=0 (stabilizer state) -> the MAGIC route simulates it EXACTLY,")
    print("   though weight and dim g are large; at generic angles SRE>0 and is encoding-INVARIANT")
    print("   (same value across JW/BK/parity). This third certificate (with DLA) floors cost* in")
    print("   every gauge. Completes the floor: hardness needs ALL of weight, DLA, AND magic large.")
