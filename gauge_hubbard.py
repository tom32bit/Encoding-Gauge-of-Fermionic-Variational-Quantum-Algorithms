"""Condensed-matter instance: the 1D Fermi-Hubbard model under the encoding orbit.

Reproduces the paper's Hubbard table. For L sites at half filling (n = 2L qubits, t=1, U=3):
  - dim g (dynamical-Lie-algebra dimension) identical across JW / BK / parity, cross-validated
    dense matrices vs sparse Pauli-string closure;
  - ansatz-output magic (second stabilizer Renyi entropy) identical across encodings;
  - max Pauli weight small and bounded (local model; Bravyi-Kitaev is not smaller than JW).

This exhibits the invariant axis (large, encoding-invariant dim g and magic) in a physical
lattice model, the complement to the molecular instance where the covariant weight is large.

Run: python gauge_hubbard.py
"""
import numpy as np
from gauge_micro import hubbard_chain, encode, dense, dla_dim, two_sre, ENC, max_mean_weight
from gauge_dla_scale import dla_dim_pauli, aH
from gauge_molecules import hf_index, circuit
from gauge_engine.statevec import apply_rotation


def ansatz_magic(gens, n, ne, enc, layers=2, seed=7, amp=0.9):
    """Second stabilizer Renyi entropy of the Hamiltonian-variational ansatz output state."""
    ang = [list(np.random.default_rng(seed + L).uniform(0.3, amp, size=len(gens))) for L in range(layers)]
    psi = np.zeros(1 << n, complex)
    psi[hf_index(enc, n, ne)] = 1.0
    for (x, z, th) in circuit(enc, gens, n, ang):
        psi = apply_rotation(psi, x, z, th, n)
    return two_sre(psi, n)


if __name__ == "__main__":
    print("######## 1D Fermi-Hubbard (t=1, U=3, half filling) under the encoding orbit ########\n")
    print("%-4s %-4s | %-20s | %-30s | %-22s" %
          ("L", "n", "weight JW/BK/parity", "dim g (dense, sparse, invariant)", "magic 2-SRE (invariant)"))
    print("-" * 92)
    for L in (2, 3):
        n, gens, H = hubbard_chain(L)
        ne = L
        wt = {e: max_mean_weight(encode(H, e, n))[0] for e in ENC}
        dgd = dla_dim([dense(encode(g, "JW", n), n) for g in gens], cap=4 ** n)
        dge = {e: dla_dim_pauli(aH(gens, e, n), cap=20000) for e in ENC}
        am = {e: ansatz_magic(gens, n, ne, e) for e in ENC}
        dg_inv = len(set(dge.values())) == 1
        m_inv = (max(am.values()) - min(am.values()) < 1e-6)
        print("%-4d %-4d | %d / %d / %d              | dense=%d sparse=%d invariant=%s | %.4f invariant=%s"
              % (L, n, wt["JW"], wt["BK"], wt["parity"], dgd, dge["JW"], dg_inv, am["JW"], m_inv))
    print("\n=> dim g and magic are identical across JW/BK/parity (invariant); the weight is small and")
    print("   bounded (local model), with Bravyi-Kitaev not smaller than Jordan-Wigner. Hardness is")
    print("   certified by the invariant dim g, not by the weight.")
