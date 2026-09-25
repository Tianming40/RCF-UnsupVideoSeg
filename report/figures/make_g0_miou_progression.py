#!/usr/bin/env python3
"""
Static (PDF+PNG) version of the g0 sum-mIoU progression chart, English
labels, for embedding in report.tex via \\includegraphics.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ACCENT = "#2f6f5e"
INK = "#20241f"
MUTED = "#7a7a70"
GRID = "#dedad0"

data = [
    ("v9",   136.68, "baseline", "Phase 1"),
    ("v53",  138.83, "$w_{dino}=0.1$\nDINO loss", "Phase 3"),
    ("v64",  139.33, "MultiScaleSegHead\n+ ASPP", "Phase 3"),
    ("v83",  139.89, "manually removed\nlow-quality training data", "Phase 7"),
    ("v102", 140.19, "GT flow downsample\nbilinear $\\rightarrow$ area", "Phase 11"),
]

xs = list(range(len(data)))
ys = [d[1] for d in data]

fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# gridlines
for t in [136, 137, 138, 139, 140, 141]:
    ax.axhline(t, color=GRID, linewidth=1, zorder=0)

# main line
ax.plot(xs, ys, color=ACCENT, linewidth=2.5, marker="o",
        markersize=9, markerfacecolor="white", markeredgecolor=ACCENT,
        markeredgewidth=2.5, zorder=3)
# emphasize peak (v102)
ax.plot(xs[-1], ys[-1], marker="o", markersize=11, color=ACCENT, zorder=4)

# value + mechanism labels (all mechanism text placed ABOVE its point --
# points are monotonically rising, so "above" always has clear headroom
# and never competes with the previous/next point's label the way
# alternating above/below did at the compressed low end)
for i, (ver, val, mech, phase) in enumerate(data):
    ax.annotate(f"{val:.2f}", (i, val), xytext=(0, 14),
                textcoords="offset points", ha="center",
                fontsize=11, fontweight="bold", color=INK)
    ax.annotate(mech, (i, val), xytext=(0, 40),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8.3, color="#40453e", linespacing=1.5)

ax.set_xticks(xs)
ax.set_xticklabels([d[0] for d in data], fontsize=11, fontweight="medium")
ax.set_xlim(-0.6, xs[-1] + 0.6)
ax.set_ylim(135.3, 142.6)
ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax.set_ylabel("g0 sum mIoU (instrument + tissue, %)", fontsize=10.5, color=INK)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9.5)

ax.set_title("Grasp0 Dataset mIoU: Baseline to Final Model", fontsize=14,
             fontweight="bold", color=INK, pad=18)

fig.tight_layout()
fig.savefig("g0_miou_progression.pdf", bbox_inches="tight")
fig.savefig("g0_miou_progression.png", bbox_inches="tight")
print("wrote g0_miou_progression.pdf / .png")
