import numpy as np
from openfermion import FermionOperator
from gauge_micro import encode
from gauge_molecules import hf_index, qop_to_terms
from gauge_trainability import interacting_gens, grad_vec, grad_component, sample_theta

def var_grad(n, S=60, K=10, layers=2, seed=11):
    gens = interacting_gens(n); ne = n // 2; rng = np.random.default_rng(seed)
    b = hf_index("JW", n, ne); ob = qop_to_terms(encode(FermionOperator("0^ 0"), "JW", n))
    gt = grad_vec("JW", gens, n, b, ob, sample_theta(len(gens), layers, rng))
    ks = [int(k) for k in np.argsort(-np.abs(gt))[:K]]
    thetas = [sample_theta(len(gens), layers, rng) for _ in range(S)]
    per = [np.var([grad_component("JW", gens, n, b, ob, th, k) for th in thetas]) for k in ks]
    return float(np.mean(per))

for n in (4, 6, 8, 10, 12):
    print("interacting", n, var_grad(n), flush=True)
