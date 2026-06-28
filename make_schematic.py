"""Figure 1: the encoding gauge. Top band, covariant (weight and cost shrink under re-encoding);
bottom band, invariant (Lie-algebra dimension, magic, and trainability stay equal)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

SLATE, RED, BLUE = "#3d5a80", "#a82f2c", "#22568f"
RTINT, BTINT = "#fbefec", "#eef3fb"

fig, ax = plt.subplots(figsize=(9.0, 4.7))
ax.set_xlim(0, 17.4); ax.set_ylim(0, 7.2); ax.axis("off")


def register(x0, y, pattern, fill, s=0.50):
    for i, p in enumerate(pattern):
        ax.add_patch(Rectangle((x0 + i * s, y), s * 0.9, s * 0.9,
                     fc=(fill if p else "white"), ec="#9aa3b2", lw=0.8, zorder=3))
        if p:
            ax.text(x0 + i * s + 0.45 * s, y + 0.45 * s, p, ha="center", va="center",
                    color="white", fontsize=8.5, weight="bold", zorder=4)


def hbar(x, y, w, h, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.04",
                 fc=fc, ec="none", zorder=3))


# ===================== top band: COVARIANT =====================
ax.add_patch(FancyBboxPatch((0.15, 3.85), 16.7, 3.05, boxstyle="round,pad=0.02,rounding_size=0.12",
             fc=RTINT, ec="#e7c9c4", lw=1.0, zorder=1))
ax.text(0.5, 6.55, "What re-encoding CHANGES", fontsize=10.5, color=RED, weight="bold", zorder=5)
ax.text(0.5, 6.18, "the same operator $a_i^{\\dagger}a_j$ becomes a longer or shorter Pauli string",
        fontsize=8.7, color="#5a4340", zorder=5)

register(3.55, 5.35, ['', 'X', 'Z', 'Z', 'Z', 'Z', 'Z', 'X', '', ''], SLATE)
ax.text(3.45, 5.575, "Jordan-Wigner", ha="right", va="center", fontsize=9.5, zorder=5)
ax.text(8.7, 5.575, "weight $=n$", ha="left", va="center", fontsize=9, zorder=5)

register(3.55, 4.30, ['', 'X', '', '', 'Y', '', '', 'Z', '', ''], SLATE)
ax.text(3.45, 4.525, "tree encoding", ha="right", va="center", fontsize=9.5, zorder=5)
ax.text(8.7, 4.525, "weight $\\sim \\log n$", ha="left", va="center", fontsize=9, zorder=5)

# cost meter
ax.text(12.6, 6.18, "simulation cost", ha="center", fontsize=8.7, style="italic", color="#5a4340", zorder=5)
hbar(11.3, 5.40, 2.2, 0.34, SLATE)
hbar(11.3, 4.35, 0.95, 0.34, SLATE)
ax.add_patch(FancyArrowPatch((11.15, 5.40), (11.15, 4.69), arrowstyle="-|>", mutation_scale=12,
             color=RED, lw=1.6, zorder=5))
ax.text(14.0, 4.85, "GAUGE-\nCOVARIANT", ha="left", va="center", fontsize=9.5, color=RED, weight="bold", zorder=5)

# ===================== bottom band: INVARIANT =====================
ax.add_patch(FancyBboxPatch((0.15, 0.40), 16.7, 3.05, boxstyle="round,pad=0.02,rounding_size=0.12",
             fc=BTINT, ec="#c7d6ea", lw=1.0, zorder=1))
ax.text(0.5, 3.10, "What re-encoding CANNOT change", fontsize=10.5, color=BLUE, weight="bold", zorder=5)
ax.text(0.5, 2.73, "the dynamical Lie algebra and the magic do not change",
        fontsize=8.7, color="#34465e", zorder=5)

# equal markers for dim g and magic (JW vs tree)
for yy, name in [(1.85, "$\\dim\\mathfrak{g}$"), (1.10, "magic")]:
    ax.text(3.45, yy + 0.17, name, ha="right", va="center", fontsize=9.5, zorder=5)
    hbar(3.55, yy, 2.2, 0.34, BLUE)
    hbar(6.05, yy, 2.2, 0.34, BLUE)
    ax.text(4.65, yy + 0.17, "JW", ha="center", va="center", color="white", fontsize=8, zorder=5)
    ax.text(7.15, yy + 0.17, "tree", ha="center", va="center", color="white", fontsize=8, zorder=5)
ax.text(8.55, 1.47, "$=$", ha="center", va="center", fontsize=16, color=BLUE, zorder=5)

# trainability meter (equal bars + lock)
ax.text(12.6, 2.73, "trainability ($1/\\mathrm{variance}$)", ha="center", fontsize=8.7, style="italic", color="#34465e", zorder=5)
hbar(11.3, 1.95, 1.6, 0.34, BLUE)
hbar(11.3, 1.10, 1.6, 0.34, BLUE)
ax.text(14.0, 1.50, "GAUGE-\nINVARIANT", ha="left", va="center", fontsize=9.5, color=BLUE, weight="bold", zorder=5)

plt.savefig("figures/fig_schematic.pdf", bbox_inches="tight")
plt.savefig("figures/fig_schematic.png", dpi=170, bbox_inches="tight")
print("wrote fig_schematic.pdf and .png")
