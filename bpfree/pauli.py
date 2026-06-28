"""Pauli-propagation engine for QAOA/MaxCut (the keystone for the simulability axis).

Back-propagates the observable O_e = Z_i Z_j through U to form the Heisenberg
operator O_H = U^dag O_e U = sum_P a_P(theta) P, as a dict {(x,z): coeff}.

A Pauli string is encoded by two integer bitmasks over qubits 0..n-1, in the
X^x Z^z convention (single-qubit: I=(0,0), X=(1,0), Z=(0,1), Y=(1,1) meaning X.Z).

This engine plays a DOUBLE role:
  * it is the classical simulator whose truncated cost defines the simulability axis;
  * it measures the shared order parameter: the Pauli-WEIGHT distribution of O_H.

Conjugation rule for a gate g = exp(-i theta G) (G a Pauli, G^2=I), Heisenberg
(O -> g^dag O g):   if [P,G]=0: P unchanged;  else: cos(2 theta) P - i sin(2 theta) (P.G).
For QAOA: mixer gate exp(-i beta X_v) -> theta=beta, G=X_v;
          cost  gate exp(+i gamma Z_a Z_b /2) -> theta=-gamma/2, G=Z_a Z_b.
Initial/final state |+>^n: <+|P|+> = 1 if P has no Z/Y (z-mask==0) else 0.
"""
from __future__ import annotations
import numpy as np


def _popcount(x: int) -> int:
    return int(x).bit_count()


def _commute(xp, zp, xg, zg) -> bool:
    """Two Paulis commute iff symplectic inner product is even."""
    return (_popcount(xp & zg) + _popcount(zp & xg)) % 2 == 0


def _pmul(xp, zp, xg, zg):
    """P.G in X^x Z^z convention: returns (x, z, sign) with sign=+-1.
    (X^xp Z^zp)(X^xg Z^zg) = (-1)^{popcount(zp & xg)} X^{xp^xg} Z^{zp^zg}."""
    x = xp ^ xg
    z = zp ^ zg
    sign = -1.0 if (_popcount(zp & xg) & 1) else 1.0
    return x, z, sign


def zz_pauli(n, a, b):
    """Encode Z_a Z_b as (x=0, z=bit_a|bit_b). Qubit q -> bit q."""
    return 0, (1 << a) | (1 << b)


def _conj_gate(terms, xg, zg, theta):
    """Heisenberg conjugation O -> g^dag O g for a gate g = exp(-i theta G_herm),
    where G_herm is the HERMITIAN Pauli with encoding (xg, zg). For a generator
    containing Y factors, G_herm = i^{#Y} X^xg Z^zg; the i^{#Y} phase is folded in.
    Rule (anticommuting):  P -> cos(2 theta) P - i sin(2 theta) (P . G_herm)."""
    c = np.cos(2.0 * theta)
    s = np.sin(2.0 * theta)
    gamma = (1j) ** (_popcount(xg & zg))   # i^{number of Y in the generator}
    out = {}
    for (xp, zp), coeff in terms.items():
        if _commute(xp, zp, xg, zg):
            out[(xp, zp)] = out.get((xp, zp), 0j) + coeff
        else:
            out[(xp, zp)] = out.get((xp, zp), 0j) + coeff * c
            xr, zr, sign = _pmul(xp, zp, xg, zg)
            out[(xr, zr)] = out.get((xr, zr), 0j) + coeff * (-1j) * s * gamma * sign
    return out


def _truncate(terms, w_max, delta):
    if w_max is None and delta <= 0.0:
        return terms
    out = {}
    for (x, z), c in terms.items():
        if abs(c) < delta:
            continue
        if w_max is not None and _popcount(x | z) > w_max:
            continue
        out[(x, z)] = c
    return out


def propagate_circuit(n, gates, obs_terms, w_max=None, delta=0.0, trunc_every=1):
    """Back-propagate observable through a general circuit.

    `gates` is a list of (xg, zg, theta) in CIRCUIT order (first gate acts first
    on the state). We form O_H = U^dag O U by conjugating O by the gates in
    REVERSE order. Optional weight/coefficient truncation every `trunc_every`
    gates defines the truncated classical (Pauli-path) simulator.
    """
    terms = dict(obs_terms)
    for idx, (xg, zg, theta) in enumerate(reversed(gates)):
        terms = _conj_gate(terms, xg, zg, theta)
        if (idx + 1) % trunc_every == 0:
            terms = _truncate(terms, w_max, delta)
    return _truncate(terms, w_max, delta)


def _commutator_i(terms, xg, zg):
    """Return i[G_herm, terms] in Pauli form, G_herm = i^{#Y} X^xg Z^zg.
    For anticommuting P: i[G,P] = -2i (P.G_herm)."""
    gamma = (1j) ** (_popcount(xg & zg))
    out = {}
    for (xp, zp), c in terms.items():
        if not _commute(xp, zp, xg, zg):
            xr, zr, sign = _pmul(xp, zp, xg, zg)
            out[(xr, zr)] = out.get((xr, zr), 0j) + c * (-2j) * gamma * sign
    return out


def gradient_measure(n, gates, m, obs_terms, init="zero", w_max=None, delta=0.0):
    """Exact (or weight-truncated) d<O>/dtheta_m via direct gradient-OPERATOR propagation:
    O' = backprop O through gates after m; form i[G_m, O']; backprop through gates 0..m;
    measure. With w_max set, this is the truncated classical estimator of the gradient."""
    after = gates[m + 1:]
    o_prime = propagate_circuit(n, after, obs_terms, w_max=w_max, delta=delta)
    xm, zm, _ = gates[m]
    comm = _truncate(_commutator_i(o_prime, xm, zm), w_max, delta)
    before = gates[: m + 1]
    full = propagate_circuit(n, before, comm, w_max=w_max, delta=delta)
    return (expectation_zero if init == "zero" else expectation_plus)(full)


def qaoa_gates(n, edges, betas, gammas):
    """QAOA U as a circuit-order gate list: per layer, cost edges then mixer qubits.
    cost gate exp(+i gamma Z_aZ_b/2) -> (xg,zg)=Z_aZ_b, theta=-gamma/2;
    mixer gate exp(-i beta X_v)      -> (xg,zg)=X_v,    theta= beta."""
    gates = []
    for l in range(len(betas)):
        for (a, b) in edges:
            xg, zg = zz_pauli(n, a, b)
            gates.append((xg, zg, -gammas[l] / 2.0))
        for v in range(n):
            gates.append(((1 << v), 0, betas[l]))
    return gates


def propagate_obs(n, edges, betas, gammas, obs_terms, w_max=None, delta=0.0):
    """Back-propagate `obs_terms` through QAOA U (thin wrapper over propagate_circuit)."""
    gates = qaoa_gates(n, edges, betas, gammas)
    return propagate_circuit(n, gates, obs_terms, w_max=w_max, delta=delta)


def expectation_plus(terms):
    """<+|^n O_H |+>^n = sum of coeffs of terms with z-mask == 0 (only I/X)."""
    val = 0j
    for (x, z), c in terms.items():
        if z == 0:
            val += c
    return float(np.real(val))


def expectation_zero(terms):
    """<0|^n O_H |0>^n = sum of coeffs of terms with x-mask == 0 (only I/Z)."""
    val = 0j
    for (x, z), c in terms.items():
        if x == 0:
            val += c
    return float(np.real(val))


def weight_profile(terms):
    """Return array w_mass[k] = sum |coeff|^2 over terms of Pauli weight k."""
    prof = {}
    for (x, z), c in terms.items():
        w = _popcount(x | z)
        prof[w] = prof.get(w, 0.0) + abs(c) ** 2
    if not prof:
        return np.zeros(1)
    out = np.zeros(max(prof) + 1)
    for k, m in prof.items():
        out[k] = m
    return out


def edge_obs(n, i, j):
    """Observable terms for Z_i Z_j."""
    x, z = zz_pauli(n, i, j)
    return {(x, z): 1.0 + 0j}


# ------------------------------- self-test ---------------------------------- #
def _selftest():
    import networkx as nx
    from .qaoa import cut_diagonal, qaoa_state
    rng = np.random.default_rng(3)
    for (n, D, p) in [(5, 2, 2), (6, 3, 2), (8, 3, 3), (10, 3, 2)]:
        G = nx.random_regular_graph(D, n, seed=n)
        edges = [(int(a), int(b)) for a, b in G.edges()]
        cut = cut_diagonal(n, edges)
        betas = rng.uniform(0, 2 * np.pi, p)
        gammas = rng.uniform(0, 2 * np.pi, p)
        # statevector reference for <Z_i Z_j> on the first edge
        i, j = edges[0]
        state = qaoa_state(n, cut, betas, gammas)
        # <Z_i Z_j> via diagonal (qubit q -> big-endian bit (n-1-q))
        k = np.arange(2 ** n)
        zi = 1 - 2 * ((k >> (n - 1 - i)) & 1)
        zj = 1 - 2 * ((k >> (n - 1 - j)) & 1)
        ref = float(np.real(np.sum(np.conj(state) * (zi * zj) * state)))
        # pauli propagation (exact, no truncation)
        terms = propagate_obs(n, edges, betas, gammas, edge_obs(n, i, j))
        got = expectation_plus(terms)
        err = abs(ref - got)
        print(f"  n={n} D={D} p={p}: <ZiZj> ref={ref:+.6f} pauli={got:+.6f} |err|={err:.2e} "
              f"(#terms={len(terms)})")
        assert err < 1e-9, f"Pauli propagation mismatch: {err}"
    print("  [OK] Pauli propagation matches statevector exactly.")


if __name__ == "__main__":
    _selftest()
