"""Designed, multi-panel data figures: a covariance figure and an invariance figure."""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from openfermion import FermionOperator
from gauge_micro import encode

mpl.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.labelsize": 11.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.fontsize": 9.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "lines.linewidth": 1.9, "lines.markersize": 6,
})
JW, BK, PAR, REF = "#D55E00", "#0072B2", "#009E73", "#666666"   # colorblind-safe; JW covariant, BK invariant
OUT = "figures/"


def mol(N, seed=1):
    rng = np.random.default_rng(seed); H = FermionOperator()
    for p in range(N):
        for q in range(p + 1, N):
            H += rng.normal() * (FermionOperator("%d^ %d" % (p, q)) + FermionOperator("%d^ %d" % (q, p)))
            H += rng.normal() * FermionOperator("%d^ %d %d^ %d" % (p, p, q, q))
    return H


def mw(q):
    return max((len(t) for t in q.terms if t), default=0)


def tag(ax, s):
    ax.text(-0.17, 1.04, s, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


# ===================== Figure: simulation cost is covariant =====================
fig, (a, b) = plt.subplots(1, 2, figsize=(8.7, 3.7))
Ns = list(range(4, 31, 2))
wt = {e: [mw(encode(mol(N), e, N)) for N in Ns] for e in ("JW", "BK", "parity")}
a.plot(Ns, wt["JW"], "o-", color=JW, label="Jordan-Wigner")
a.plot(Ns, wt["parity"], "^-", color=PAR, mfc="white", label="parity")
a.plot(Ns, wt["BK"], "s-", color=BK, label="Bravyi-Kitaev")
a.plot(Ns, [np.log2(N) for N in Ns], ":", color=REF, lw=1.5, label=r"$\log_2 n$")
a.set_xlabel(r"system size $n$"); a.set_ylabel(r"max Pauli weight of $H$")
a.text(25, 25.5, "linear", color=JW, fontsize=9.5, rotation=33, ha="center")
a.text(22, 6.4, "logarithmic", color=BK, fontsize=9.5)
a.legend(loc="upper left"); tag(a, "(a)")

# Computed W* of the sparse non-local ansatz (2 Trotter layers, angles ~U[0.4,0.85],
# observable n_{n/2}, half-filling HF reference, eps=0.05); values from the Kaggle run
# logged in "gauge scalling log.txt". Plotted only over the directly computed range n<=16.
Nw = np.array([6, 8, 10, 12, 14, 16])
jw = np.array([4, 5, 6, 8, 8, 9]); bk = np.array([3, 5, 5, 5, 5, 7]); pa = np.array([4, 5, 6, 7, 8, 9])
thr = 2 * np.log2(Nw)
b.axvspan(11, 16.6, color="#bdbdbd", alpha=0.16, lw=0)
b.plot(Nw, jw, "o-", color=JW, label="Jordan-Wigner")
b.plot(Nw, pa, "^-", color=PAR, mfc="white", label="parity")
b.plot(Nw, bk, "s-", color=BK, label="Bravyi-Kitaev")
b.plot(Nw, thr, "--", color=REF, lw=1.4, label="hardness threshold")
b.set_xlabel(r"system size $n$"); b.set_ylabel(r"truncation weight $W^{*}$")
b.text(13.8, 3.4, "verdict depends\non the encoding", color="#444", fontsize=9, ha="center")
b.set_xlim(5, 17); b.legend(loc="upper left"); tag(b, "(b)")
fig.tight_layout(); fig.savefig(OUT + "fig_covariance.pdf"); fig.savefig(OUT + "fig_covariance.png", dpi=150); plt.close(fig)

# ===================== Figure: the floor (DLA dimension vs weight) =====================
fig, ax = plt.subplots(figsize=(5.7, 4.1))
Nd = np.arange(4, 25, 2)
ax.plot(Nd, Nd ** 2 - 1, "o-", color=BK, label=r"$\dim\mathfrak{g}$ (gauge-invariant)")
ax.plot(Nd, Nd, "s-", color=JW, label="Jordan-Wigner weight (covariant)")
ax.set_xlabel(r"system size $n$"); ax.set_ylabel("resource value")
ax.text(19.6, 330, r"$\sim n^{2}$", color=BK, fontsize=12)
ax.text(21.2, 33, r"$=n$", color=JW, fontsize=12)
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig_dla_scale.pdf"); fig.savefig(OUT + "fig_dla_scale.png", dpi=150); plt.close(fig)

# ===================== Figure: variance dichotomy =====================
fig, ax = plt.subplots(figsize=(5.7, 4.1))
free_n = np.array([4, 6, 8, 10, 12, 14, 16])
free = np.array([0.83, 1.46, 1.61, 1.28, 1.46, 1.44, 1.10]) / (free_n ** 2 - 1)
int_n = np.array([4, 6, 8, 10, 12]); intv = np.array([0.054888717, 0.027732702, 0.018812638, 0.010254208, 0.008151615])
C = float(np.mean(free * (free_n ** 2 - 1)))
ax.semilogy(free_n, free, "o-", color=BK, label="free fermions (poly $\\dim\\mathfrak{g}$)")
ax.semilogy(int_n, intv, "s-", color=JW, label="interacting (exp $\\dim\\mathfrak{g}$)")
ax.semilogy(free_n, C / (free_n ** 2 - 1), "--", color=REF, lw=1.4, label=r"$\propto 1/\dim\mathfrak{g}$")
ax.set_xlabel(r"system size $n$"); ax.set_ylabel(r"$\mathrm{Var}\,[\partial_\theta \langle O\rangle]$")
ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(OUT + "fig_variance.pdf"); fig.savefig(OUT + "fig_variance.png", dpi=150); plt.close(fig)
print("wrote fig_covariance, fig_dla_scale, fig_variance (pdf + png)")
