# The Encoding Gauge of Fermionic Variational Quantum Algorithms

**Classical Simulability is Relative, Trainability is Invariant**

Reproducibility package — code, validation suite, figure generators, and logged
outputs supporting the manuscript.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21011413.svg)](https://doi.org/10.5281/zenodo.21011413)

> S.M. Yousuf Iqbal Tomal and Abdullah Al Shafin,
> Department of Computer Science and Engineering, BRAC University, Dhaka, Bangladesh.

**Archived release:** [https://doi.org/10.5281/zenodo.21011413](https://doi.org/10.5281/zenodo.21011413)

Standard fermion-to-qubit encodings (Jordan–Wigner, Bravyi–Kitaev, the
Bravyi–Kitaev tree, parity, ternary tree) are related by Clifford transformations,
so they form a **gauge group** acting on the classical-simulability problem. This
code demonstrates the central split: the **Pauli weight** and **causal-cone width**
are gauge-*covariant* (re-encoding can lower them exponentially), while the
**dynamical-Lie-algebra dimension** and the **non-stabilizerness (magic)** are
gauge-*invariant*. Trainability — the cost landscape, the gradient, and the
barren-plateau variance — is gauge-invariant to machine precision.

Every quantity reported in the paper is checked against an independent ground truth
(exact statevector simulation, closed-form Lie-algebra dimensions, analytic magic
values) **before** any result is produced. Run `validate_gauge.py` first.

---

## Installation

Python ≥ 3.10. Install the dependencies:

```bash
pip install -r requirements.txt
```

The molecular instances use the integrals bundled with
[OpenFermion](https://github.com/quantumlib/OpenFermion), so no external data
download is required.

## Quick start

Run from the repository root (the scripts import each other as flat modules and
expect the working directory to be this folder):

```bash
python validate_gauge.py      # 9/9 independent-ground-truth checks (run this first)
python gauge_molecules.py     # Table 1: the gauge split on real H2 (STO-3G, 6-31G)
python make_figs.py           # regenerates Figures 3, 5, 7 into figures/
```

## What reproduces what

| Paper item | Script | Notes |
|---|---|---|
| **Validation (Section 3)** | `validate_gauge.py` | isospectrality, HF energy, engine == statevector, DLA == u(L), analytic magic, weight covariance — 9 checks |
| **Fig. 1** (encoding-gauge schematic) | `make_schematic.py` | conceptual diagram |
| **Fig. 2** (origin of covariance) | `make_mechanism_fig.py` | chain vs. tree; causal cones |
| **Fig. 3(a)** (Pauli weight: JW linear, BK log) | `make_figs.py` | computed live to n = 30 |
| **Fig. 3(b)** (W\* change of verdict) | `make_figs.py` | plotted over the directly computed range **n = 6…16**; the W\* values are logged in `outputs/gauge_scaling_log.txt` |
| **Fig. 4** (the three simulability routes) | `make_routes_fig.py` | class diagram |
| **Fig. 5** (Lie-algebra route at scale) | `make_figs.py` | dim g = n²−1 vs. JW weight |
| **Fig. 6** (cost is encoding-invariant) | `make_commute_fig.py` | \*-isomorphism commutative diagram |
| **Fig. 7** (trainability is gauge-invariant) | `make_figs.py` | Var ∝ 1/dim g; data in `outputs/gauge_trainability_log.txt` |
| **Table 1** (molecular instance) | `gauge_molecules.py` | H2 STO-3G and 6-31G, JW/BK/parity |
| **Floor, both invariant routes (Section 5)** | `gauge_floor.py`, `gauge_magic_route.py` | Lie-algebra route + magic (Clifford-locus) route |
| **W\* flip window (operating point)** | `gauge_sweep.py` | shows the JW-hard / BK-simulable window on H2/6-31G |
| **Gradient-variance scaling (Section 6, App. A)** | `gauge_var_scale.py`, `gauge_var_interacting.py` | Var·dim g constant for free fermions |
| **DLA dimension at scale** | `gauge_dla_scale.py` | sparse Pauli Lie-closure, dim g = n²−1 to n = 24 |

### Core library

| Module | Contents |
|---|---|
| `gauge_micro.py` | the encodings (`encode`, `ENC`), `dla_dim`, `two_sre` (stabilizer Rényi entropy), dense operators |
| `gauge_pipeline.py` | Pauli-term bookkeeping (`qop_to_terms`) bridging OpenFermion and the engine |
| `gauge_molecules.py` | molecular Hamiltonians, per-encoding Hartree–Fock reference, circuit-level W\* |
| `gauge_trainability.py` | gradients and the barren-plateau variance diagnostic |
| `gauge_dla_scale.py` | sparse Lie-closure over Pauli strings |
| `gauge_engine/pauli.py` | truncated Pauli (Heisenberg) propagation — the weight-controlled simulator |
| `gauge_engine/statevec.py` | exact statevector reference (independent ground truth) |

### Heavy / asymptotic runs (Kaggle)

The large-n scaling that backs Fig. 3(b) and Fig. 7 was produced on Kaggle. The
notebooks and their generators are included, together with the raw logs they
emitted:

| Notebook | Generator | Logged output |
|---|---|---|
| `notebooks/gauge_scaling_kaggle.ipynb` | `make_notebook.py` | `outputs/gauge_scaling_log.txt` |
| `notebooks/gauge_trainability_kaggle.ipynb` | `make_notebook_train.py` | `outputs/gauge_trainability_log.txt` |

The notebooks are self-validating: they re-run the same independent-ground-truth
checks before producing any scaling data.

## Layout

```
.
├── validate_gauge.py            # run first: 9/9 ground-truth checks
├── gauge_*.py                   # core library + demonstration scripts
├── make_*.py                    # figure and notebook generators
├── gauge_engine/                # numpy-only engine (pauli + statevec) used by the gauge code
├── notebooks/                   # Kaggle scaling notebooks
├── figures/                     # the seven paper figures (png + pdf)
├── outputs/                     # logged numerical outputs underlying the figures/tables
├── requirements.txt
└── README.md
```

## Notes on precision and determinism

All computations use IEEE double precision (float64); the "machine-precision"
agreements quoted in the paper are at the 1e-12…1e-15 level set by accumulated
rounding. Random angles use fixed NumPy `default_rng` seeds, so the small-n scripts
are deterministic. The barren-plateau variances are Monte-Carlo estimates and carry
the corresponding sampling error.

## License

Released under the MIT License (see `LICENSE`).
