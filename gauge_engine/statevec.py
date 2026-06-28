"""General statevector simulator for circuits of Pauli-rotation gates.

A gate is exp(-i theta G_herm) with G_herm the HERMITIAN Pauli encoded by (x, z)
(G_herm = i^{#Y} X^x Z^z). Used as ground truth for the light-cone DAG ansatz and
to validate the Pauli engine on gates containing Y (e.g. ZY gates).

Bit convention: qubit q <-> bit q of the basis index k (little-endian), MATCHING
gauge_engine/pauli.py (mask bit q = 1<<q). (This differs from qaoa.py's big-endian core;
the two are never mixed.)
"""
from __future__ import annotations
import numpy as np


def _pauli_apply(state, x, z, n):
    """Return (X^x Z^z) |state>  (the bare encoded string, no i^{#Y})."""
    k = np.arange(2 ** n)
    sign = np.ones(2 ** n)
    zz = z
    while zz:
        b = (zz & -zz).bit_length() - 1
        sign = sign * (1 - 2 * ((k >> b) & 1))
        zz &= zz - 1
    tmp = state * sign            # Z^z |state>
    return tmp[k ^ x]             # X^x permutation


def apply_rotation(state, x, z, theta, n):
    """Apply exp(-i theta G_herm), G_herm = i^{#Y} X^x Z^z (Hermitian, G^2=I)."""
    gamma = (1j) ** (int(x & z).bit_count())   # i^{#Y}
    g_state = gamma * _pauli_apply(state, x, z, n)
    return np.cos(theta) * state - 1j * np.sin(theta) * g_state


def init_state(n, init="zero"):
    if init == "zero":
        s = np.zeros(2 ** n, dtype=np.complex128); s[0] = 1.0; return s
    if init == "plus":
        return np.full(2 ** n, 1.0 / np.sqrt(2 ** n), dtype=np.complex128)
    raise ValueError(init)


def run(n, gates, init="zero"):
    s = init_state(n, init)
    for (x, z, theta) in gates:
        s = apply_rotation(s, x, z, theta, n)
    return s


def run_with_grad(n, gates, m, init="zero"):
    """Return (psi, dpsi/dtheta_m). Generator of gate m is G_herm (inserts -i G_herm)."""
    s = init_state(n, init)
    d = None
    for idx, (x, z, theta) in enumerate(gates):
        s = apply_rotation(s, x, z, theta, n)
        if d is not None:
            d = apply_rotation(d, x, z, theta, n)
        if idx == m:
            gamma = (1j) ** (int(x & z).bit_count())
            d = -1j * gamma * _pauli_apply(s, x, z, n)   # -i G_herm psi (after gate m)
    return s, d


def zz_diag(n, i, j):
    k = np.arange(2 ** n)
    return (1 - 2 * ((k >> i) & 1)) * (1 - 2 * ((k >> j) & 1))


def zz_expectation(state, i, j, n):
    d = zz_diag(n, i, j)
    return float(np.real(np.sum(np.conj(state) * d * state)))


def zz_grad(n, gates, m, i, j, init="zero"):
    """Exact d<Z_i Z_j>/dtheta_m."""
    s, d = run_with_grad(n, gates, m, init)
    diag = zz_diag(n, i, j)
    return 2.0 * np.real(np.sum(np.conj(d) * diag * s))


# ------------------------------- self-test ---------------------------------- #
def _selftest():
    """Validate Pauli engine vs statevector on random ZY-gate circuits, and the
    exact gradient vs finite differences."""
    from .pauli import propagate_circuit, expectation_zero
    rng = np.random.default_rng(7)
    n = 6
    # random circuit of ZY gates on random pairs (generator Z_a Y_b: x=bit_b, z=bit_a|bit_b)
    gates = []
    for _ in range(12):
        a, b = rng.choice(n, size=2, replace=False)
        x = (1 << int(b))
        z = (1 << int(a)) | (1 << int(b))
        gates.append((x, z, float(rng.uniform(0, 2 * np.pi))))
    i, j = 0, 1
    # statevector <Z0 Z1>
    s = run(n, gates, init="zero")
    ref = zz_expectation(s, i, j, n)
    # pauli engine: O_H = U^dag (Z0 Z1) U, measure <0|.|0>
    zmask = (1 << i) | (1 << j)
    terms = propagate_circuit(n, gates, {(0, zmask): 1.0 + 0j})
    got = expectation_zero(terms)
    print(f"  <Z0Z1>: statevec={ref:+.6f}  pauli={got:+.6f}  |err|={abs(ref-got):.2e}")
    assert abs(ref - got) < 1e-9, "Pauli engine vs statevector mismatch on ZY gates"
    # gradient check
    m = 5
    g_exact = zz_grad(n, gates, m, i, j)
    eps = 1e-6
    gp = list(gates); gp[m] = (gp[m][0], gp[m][1], gp[m][2] + eps)
    gm = list(gates); gm[m] = (gm[m][0], gm[m][1], gm[m][2] - eps)
    g_fd = (zz_expectation(run(n, gp), i, j, n) - zz_expectation(run(n, gm), i, j, n)) / (2 * eps)
    print(f"  d<Z0Z1>/dtheta_{m}: exact={g_exact:+.6f}  fd={g_fd:+.6f}  |err|={abs(g_exact-g_fd):.2e}")
    assert abs(g_exact - g_fd) < 1e-5, "gradient mismatch"
    print("  [OK] statevector <-> Pauli engine agree on ZY gates; gradient exact.")


if __name__ == "__main__":
    _selftest()
