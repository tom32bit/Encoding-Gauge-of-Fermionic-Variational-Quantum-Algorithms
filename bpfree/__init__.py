"""bpfree (engine subset) — numpy-only classical-simulation primitives used by the
encoding-gauge study.

Only the two modules required by the gauge code are included here:

  pauli    : truncated Pauli (Heisenberg) propagation through a circuit of Pauli
             rotations; the weight-controlled simulator whose cost is gauge-covariant.
  statevec : exact statevector reference (apply_rotation, _pauli_apply) used as the
             independent ground truth that every gauge result is validated against.
"""
__all__ = ["pauli", "statevec"]
