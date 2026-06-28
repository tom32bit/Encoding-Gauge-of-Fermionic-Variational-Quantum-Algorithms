"""Class/region diagram of the simulability routes, with concrete example instances and the
gauge action that moves only the (covariant) weight boundary."""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
COV, INV, HARD = "#D55E00", "#0072B2", "#8a1c1c"

fig, ax = plt.subplots(figsize=(8.2, 5.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

# universe of (problem, encoding) instances
ax.add_patch(FancyBboxPatch((0.25, 0.25), 9.5, 6.5, boxstyle="round,pad=0.02,rounding_size=0.12",
             fc="#fbece6", ec="#caa99d", lw=1.2, zorder=0))
ax.text(9.55, 0.95, "advantage", ha="right", color=HARD, fontsize=11, fontweight="bold")
ax.text(9.55, 0.55, "candidate (all routes large)", ha="right", color=HARD, fontsize=9.0)

# three route regions; weight is covariant (dashed = its boundary moves under re-encoding)
ax.add_patch(Ellipse((3.7, 3.9), 5.6, 4.4, fc=COV, alpha=0.12, ec=COV, lw=1.8, ls="--", zorder=1))
ax.add_patch(Ellipse((6.7, 4.8), 4.2, 3.0, fc=INV, alpha=0.12, ec=INV, lw=1.6, zorder=1))
ax.add_patch(Ellipse((6.7, 2.6), 4.2, 3.0, fc=INV, alpha=0.12, ec=INV, lw=1.6, zorder=1))

ax.text(2.7, 3.55, "low Pauli weight", color=COV, fontsize=10.5, fontweight="bold", ha="center")
ax.text(2.7, 3.2, r"Pauli propagation, $n^{O(W)}$", color=COV, fontsize=9, ha="center")
ax.text(2.7, 2.85, "(covariant)", color=COV, fontsize=9, style="italic", ha="center")
ax.text(8.55, 5.55, "low $\\dim\\mathfrak{g}$", color=INV, fontsize=10.5, fontweight="bold", ha="right")
ax.text(8.55, 5.2, r"$g$-sim, $\mathrm{poly}(\dim\mathfrak{g})$", color=INV, fontsize=9, ha="right")
ax.text(8.55, 1.7, "low magic", color=INV, fontsize=10.5, fontweight="bold", ha="right")
ax.text(8.55, 1.35, r"stabilizer rank, $2^{O(t)}$", color=INV, fontsize=9, ha="right")
ax.text(5.55, 3.7, "classically\nsimulable", ha="center", va="center", color="#33502f", fontsize=10, fontweight="bold")

# concrete instances
def dot(x, y, lab, dx=0.18, dy=0.18, ha="left"):
    ax.plot(x, y, "o", ms=7, color="#222", zorder=5)
    ax.text(x + dx, y + dy, lab, fontsize=8.8, ha=ha, zorder=5)

dot(6.9, 5.2, "free fermions", ha="center", dx=0, dy=0.25)
dot(6.9, 2.3, "Clifford-angle\ncircuit", ha="center", dx=0, dy=-0.55)
# a high-weight problem moved into the weight region by re-encoding
ax.plot(1.15, 6.15, "o", ms=7, color="#222", zorder=5)
ax.text(1.35, 6.3, "interacting VQA\nunder Jordan-Wigner", fontsize=8.8, zorder=5)
ax.add_patch(FancyArrowPatch((1.35, 5.95), (2.55, 4.55), arrowstyle="-|>", mutation_scale=14,
             color=COV, lw=2.0, zorder=6))
ax.text(2.95, 5.15, "re-encode\n(lowers weight)", color=COV, fontsize=8.8, ha="left", va="center")

ax.set_title("Re-encoding moves only the covariant boundary; the invariant routes are fixed", fontsize=10.5)
fig.tight_layout()
fig.savefig("figures/fig_routes.pdf"); fig.savefig("figures/fig_routes.png", dpi=150)
print("wrote fig_routes")
