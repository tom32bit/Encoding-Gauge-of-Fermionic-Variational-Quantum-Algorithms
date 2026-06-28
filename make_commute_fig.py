"""Commutative diagram for the encoding *-isomorphism: the cost is the same for every encoding."""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
INK, E1, E2, CL = "#16324f", "#0072B2", "#0072B2", "#D55E00"

fig, ax = plt.subplots(figsize=(8.0, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")


def box(x, y, w, h, lines, fc="#eef2f8", ec=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                 fc=fc, ec=ec, lw=1.3, zorder=2))
    for i, (txt, fs) in enumerate(lines):
        ax.text(x, y + (len(lines) - 1) * 0.22 - i * 0.44, txt, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(p0, p1, color=INK):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15, color=color, lw=1.7, zorder=1,
                 shrinkA=4, shrinkB=4))


# nodes
box(5.0, 6.0, 5.2, 1.1, [("fermionic VQA", 11.5), (r"$U(\theta)=\prod_k e^{-i\theta_k H_k},\; O,\; \rho$", 10.5)], fc="#fdf0e9", ec=CL)
box(2.3, 3.4, 3.5, 1.1, [("encoding $E$ (qubits)", 10), (r"$U_E,\,O_E,\,\rho_E$", 10.5)])
box(7.7, 3.4, 3.5, 1.1, [("encoding $E'$ (qubits)", 10), (r"$U_{E'},\,O_{E'},\,\rho_{E'}$", 10.5)])
box(5.0, 0.85, 6.0, 1.05, [(r"$L(\theta)=\langle\rho|\,U^{\dagger}(\theta)\,O\,U(\theta)\,|\rho\rangle$", 11.5)], fc="#eaf1e7", ec="#33502f")

# arrows
arrow((4.1, 5.55), (2.7, 3.95)); ax.text(3.1, 4.95, r"$E$", color=E1, fontsize=12)
arrow((5.9, 5.55), (7.3, 3.95)); ax.text(6.7, 4.95, r"$E'$", color=E2, fontsize=12)
ax.add_patch(FancyArrowPatch((4.05, 3.4), (5.95, 3.4), arrowstyle="<|-|>", mutation_scale=14, color=CL, lw=1.7, zorder=1))
ax.text(5.0, 3.62, r"$C\,(\cdot)\,C^{\dagger}$ (Clifford)", ha="center", color=CL, fontsize=10)
arrow((2.7, 2.85), (4.2, 1.35)); ax.text(3.05, 2.0, r"$\langle\,\cdot\,\rangle$", color=INK, fontsize=11)
arrow((7.3, 2.85), (5.8, 1.35)); ax.text(6.95, 2.0, r"$\langle\,\cdot\,\rangle$", color=INK, fontsize=11)

ax.text(5.0, 6.85, "the cost is the same along either path", ha="center", fontsize=10.5, style="italic", color="#444")
ax.text(5.0, 0.05, r"$L_E(\theta)=L_{E'}(\theta)$ for every encoding $\Rightarrow$ the gradient and its variance are gauge-invariant (Lemma 2)",
        ha="center", fontsize=10, color="#33502f")

fig.tight_layout()
fig.savefig("figures/fig_commute.pdf"); fig.savefig("figures/fig_commute.png", dpi=150)
print("wrote fig_commute")
