"""The structural origin of covariance, made concrete: a specific operator a1^dag a7 and its
explicit Pauli image, numbered qubits, labelled gates, and actual cone widths."""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
HI, LO, ED, ACC, GATE, OBS = "#0072B2", "#e3e9f0", "#9aa3b2", "#D55E00", "#0072B2", "#16324f"

fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(9.2, 6.7))
for ax in (a, b, c, d):
    ax.axis("off")


def node(ax, x, y, on, lbl="", num=None):
    ax.add_patch(Circle((x, y), 0.34, fc=(HI if on else LO), ec=("#16324f" if on else ED), lw=1.3, zorder=3))
    if lbl:
        ax.text(x, y, lbl, color="white", ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=4)
    if num is not None:
        ax.text(x, y - 0.62, str(num), color="#555", ha="center", va="center", fontsize=8.5, zorder=4)


# ---------- (a) Jordan-Wigner chain: explicit Z-string ----------
n = 7
paulis = ["X", "Z", "Z", "Z", "Z", "Z", "X"]
a.plot(range(n), [0] * n, "-", color=ED, lw=1.6, zorder=1)
for i, p in enumerate(paulis):
    node(a, i, 0, True, p, num=i + 1)
a.annotate("", xy=(0, 0.66), xytext=(n - 1, 0.66), arrowprops=dict(arrowstyle="<->", color=ACC, lw=1.5))
a.text((n - 1) / 2, 1.0, r"weight $= 7$", ha="center", color=ACC, fontsize=10.5)
a.text((n - 1) / 2, -1.45, r"$a_1^{\dagger}a_7 \;\mapsto\; X_1 Z_2 Z_3 Z_4 Z_5 Z_6 X_7$", ha="center", fontsize=11)
a.set_xlim(-1.0, n - 0.0); a.set_ylim(-1.9, 1.45); a.set_aspect("equal")
a.set_title("(a) Jordan-Wigner: linear chain", loc="left", fontsize=10.5)

# ---------- (b) tree encoding: short root-to-leaf path ----------
pos = {0: (3, 2), 1: (1.4, 1), 2: (4.6, 1), 3: (0.5, 0), 4: (2.3, 0), 5: (3.7, 0), 6: (5.5, 0)}
leaf_num = {3: 1, 4: 3, 5: 5, 6: 7}
edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
path = {0: "X", 2: "Z", 6: "Y"}
for u, v in edges:
    hot = (u in path and v in path)
    b.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], "-", color=(ACC if hot else ED), lw=(2.4 if hot else 1.5), zorder=2)
for k, (x, y) in pos.items():
    node(b, x, y, k in path, path.get(k, ""), num=leaf_num.get(k))
b.text(3, 3.0, r"weight $= 3$", ha="center", color=HI, fontsize=10.5)
b.text(3, -1.45, r"$a_1^{\dagger}a_7 \;\mapsto\; X\,Z\,Y$ on one path", ha="center", fontsize=11)
b.set_xlim(-0.6, 6.6); b.set_ylim(-1.9, 3.5); b.set_aspect("equal")
b.set_title("(b) tree encoding: balanced tree", loc="left", fontsize=10.5)

# ---------- (c),(d) light cones with explicit gates and width ----------
nq, T = 6, 4.0


def gate(ax, x, q0, q1):
    ax.add_patch(FancyBboxPatch((x - 0.16, q0 - 0.16), 0.32, (q1 - q0) + 0.32,
                 boxstyle="round,pad=0.0,rounding_size=0.08", fc=GATE, ec="#0b3a5c", lw=0.8, zorder=2))


def cone(ax, conep, gates, nu, letter, title):
    ax.add_patch(Polygon(conep, closed=True, fc=ACC, alpha=0.15, ec=ACC, lw=1.1, zorder=0))
    for q in range(nq):
        ax.plot([0, T + 0.4], [q, q], "-", color=ED, lw=1.0, zorder=1)
        ax.text(-0.35, q, str(q + 1), color="#666", ha="center", va="center", fontsize=8)
    for (x, q0, q1) in gates:
        gate(ax, x, q0, q1)
    ax.add_patch(FancyBboxPatch((T + 0.18, nq - 1 - 0.18), 0.36, 0.36, boxstyle="round,pad=0,rounding_size=0.06", fc=OBS, zorder=3))
    ax.text(T + 0.78, nq - 1, r"$O$", va="center", fontsize=12)
    ax.text(T / 2, -0.75, r"cone width $\nu=%d$" % nu, ha="center", color=ACC, fontsize=10.5)
    ax.set_xlim(-0.7, T + 1.05); ax.set_ylim(-1.15, nq - 0.05); ax.set_aspect("equal")
    ax.set_title("%s %s" % (letter, title), loc="left", fontsize=10.5)


cone(c, [(T + 0.4, nq - 1), (0.6, nq - 1), (0.6, 0), (T + 0.4, 0)],
     [(1, 0, nq - 1), (2, 0, nq - 1), (3, 0, nq - 1)], 6, "(c)", "Jordan-Wigner: long-range gates")
cone(d, [(T + 0.4, nq - 1), (1.5, nq - 1), (2.3, 2.6), (T + 0.4, 2.6)],
     [(1, 4, 5), (2, 3, 4), (3, 4, 5)], 2, "(d)", "tree encoding: local gates")

fig.tight_layout()
fig.savefig("figures/fig_mechanism.pdf"); fig.savefig("figures/fig_mechanism.png", dpi=150)
print("wrote fig_mechanism")
