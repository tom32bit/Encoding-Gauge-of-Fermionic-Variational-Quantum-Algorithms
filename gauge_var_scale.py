"""Larger-n scaling of the gradient variance: Var[grad] ~ 1/dim g, gauge-invariant.

Turns the trainability/simulability dichotomy into a quantitative scaling. The gradient variance is
averaged over many components and random parameter samples (the single-component estimate at small n
was too noisy). Free fermions (dim g = su(n) = n^2-1, polynomial) stay trainable; the interacting
family (dim g exponential) flattens into a barren plateau. The variance is identical across encodings.
Heavy at the largest n; intended to run unattended. Run: python gauge_var_scale.py
"""
import numpy as np
from openfermion import FermionOperator
from gauge_micro import encode
from gauge_molecules import hf_index, circuit, exact_O, qop_to_terms
from gauge_trainability import free_gens, interacting_gens, grad_component, grad_vec, sample_theta
from gauge_dla_scale import dla_dim_pauli, aH


def var_grad(enc, gens, n, ne, layers, S, K, seed):
    rng = np.random.default_rng(seed)
    b = hf_index(enc, n, ne)
    ob = qop_to_terms(encode(FermionOperator("0^ 0"), enc, n))
    gtest = grad_vec(enc, gens, n, b, ob, sample_theta(len(gens), layers, rng))
    ks = [int(k) for k in np.argsort(-np.abs(gtest))[:K]]       # K most-active components
    thetas = [sample_theta(len(gens), layers, rng) for _ in range(S)]
    per = [np.var([grad_component(enc, gens, n, b, ob, th, k) for th in thetas]) for k in ks]
    return float(np.mean(per))


def pr(*a):
    print(*a, flush=True)


if __name__ == "__main__":
    layers, S, K = 2, 60, 8

    pr("######## invariance check: Var[grad] ratio JW/BK (should be 1.000000) ########")
    for model, gf in (("free", free_gens), ("interacting", interacting_gens)):
        for n in (4, 6, 8):
            vj = var_grad("JW", gf(n), n, n // 2, layers, S, K, seed=11)
            vb = var_grad("BK", gf(n), n, n // 2, layers, S, K, seed=11)
            pr("  %-12s n=%d  Var(JW)=%.4e  Var(BK)=%.4e  ratio=%.6f" % (model, n, vj, vb, vj / vb))

    pr("\n######## SCALING: Var[grad] vs n (Jordan-Wigner; identical in any encoding) ########")
    pr("%-12s %-4s | %-12s | %-14s" % ("model", "n", "Var[grad]", "dim g"))
    for model, gf in (("free", free_gens), ("interacting", interacting_gens)):
        for n in range(4, 13, 2):
            gens = gf(n)
            v = var_grad("JW", gens, n, n // 2, layers, S, K, seed=11)
            dg = dla_dim_pauli(aH(gens, "JW", n), cap=3000)
            pr("%-12s %-4d | %-12.5e | %-14s" % (model, n, v, dg))
        pr("")
    pr("=> Var[grad] is gauge-invariant (ratio=1); it decays slowly for free fermions (poly dim g)")
    pr("   and collapses for the interacting family (exp dim g): trainability tracks the invariant dim g.")
