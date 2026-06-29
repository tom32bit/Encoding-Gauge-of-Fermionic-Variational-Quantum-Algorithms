"""Gradient-variance estimates with bootstrap confidence intervals and documented sample counts.

Produces the data for the trainability figure: Var_theta[d<O>/dtheta] vs n for the free-fermion
(poly dim g) and interacting (exp dim g) ansatze, identical across encodings, with 95% bootstrap
CIs over S random parameter sets. The estimator averages the per-component gradient variance over
the most active components; the CI is a percentile bootstrap over the S samples.

Run: python gauge_var_errorbars.py
"""
import numpy as np
from openfermion import FermionOperator
from gauge_micro import encode, dla_dim, dense
from gauge_molecules import hf_index, circuit, exact_O, qop_to_terms
from gauge_trainability import free_gens, interacting_gens, grad_component, sample_theta

S = 400           # random parameter sets
B = 2000          # bootstrap resamples
LAYERS = 2
NCOMP = 8         # average the variance over this many gradient components
RNG = 7


def var_with_ci(model_gens, n):
    gens, ne = model_gens(n), n // 2
    rng = np.random.default_rng(RNG)
    b = hf_index("JW", n, ne)
    ob = qop_to_terms(encode(FermionOperator("0^ 0"), "JW", n))
    thetas = [sample_theta(len(gens), LAYERS, rng) for _ in range(S)]
    ncomp = min(NCOMP, len(gens))
    # G[s, k] = d<O>/dtheta_k at sample s  (k over the first ncomp components)
    G = np.array([[grad_component("JW", gens, n, b, ob, th, k) for k in range(ncomp)]
                  for th in thetas])                       # shape (S, ncomp)
    point = float(np.mean(np.var(G, axis=0)))              # mean over components of Var over samples
    brng = np.random.default_rng(123)
    boots = []
    for _ in range(B):
        idx = brng.integers(0, S, S)
        boots.append(np.mean(np.var(G[idx], axis=0)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    dimg = dla_dim([dense(encode(g, "JW", n), n) for g in gens], cap=8 * n * n)
    return point, float(lo), float(hi), dimg


if __name__ == "__main__":
    print("# Var[grad] with 95%% bootstrap CI. S=%d random theta, %d layers, mean over %d components, B=%d.\n"
          % (S, LAYERS, NCOMP, B))
    for name, gf, sizes in (("free", free_gens, (4, 6, 8, 10, 12)),
                            ("interacting", interacting_gens, (4, 6, 8))):
        print("# %s" % name)
        print("# n   Var          CI_low       CI_high      dim_g")
        for n in sizes:
            v, lo, hi, d = var_with_ci(gf, n)
            print("%-4d  %.6e  %.6e  %.6e  %s" % (n, v, lo, hi, d), flush=True)
        print()
