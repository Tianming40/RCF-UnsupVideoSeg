#!/usr/bin/env python3
"""
Forward-backward flow (cycle) consistency vs. temporal gap: the new
adaptive-threshold measurement vs. the old fixed-magnitude "toxic" filter
it replaced (README.md:1795-1825). The point of this figure is the SHAPE
difference between the two curves, not just the endpoints -- the old
method makes gap-7 data look ~40x worse than gap-1, the new method shows
it's really only ~2.25x worse (large flow magnitude at high gap is mostly
genuine motion, not RAFT failure).
"""
import matplotlib.pyplot as plt

OLD = "#b5563c"   # terracotta -- old fixed-threshold, overestimates degradation
NEW = "#2f6f5e"   # teal -- new FB-consistency, the corrected measurement
INK = "#20241f"
MUTED = "#7a7a70"
GRID = "#dedad0"

gaps = [1, 2, 3, 4, 5, 6, 7]

# old fixed-magnitude "toxic" criterion (mean>30px OR p99>80px), README:1801-1809
old_toxic = [1.8, 13.7, 31.6, 47.4, 59.1, 65.9, 69.3]

# new forward-backward cycle-consistency, adaptive threshold, mean_incons%, README:1815-1823
new_incons = [17.96, 22.95, 27.04, 31.19, 34.60, 37.66, 40.50]

fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for t in range(0, 81, 20):
    ax.axhline(t, color=GRID, linewidth=1, zorder=0)

ax.plot(gaps, old_toxic, color=OLD, linewidth=2.2, marker="o", markersize=7,
        markerfacecolor="white", markeredgecolor=OLD, markeredgewidth=2,
        label="Old fixed-magnitude threshold\n(mean>30px OR p99>80px)", zorder=3)
ax.plot(gaps, new_incons, color=NEW, linewidth=2.2, marker="o", markersize=7,
        markerfacecolor="white", markeredgecolor=NEW, markeredgewidth=2,
        label="New FB (cycle) consistency\n(adaptive, magnitude-scaled threshold)", zorder=3)

# endpoint annotations
ax.annotate(f"{old_toxic[0]:.1f}%", (gaps[0], old_toxic[0]), xytext=(-8, -16),
            textcoords="offset points", ha="center", fontsize=9.5, color=OLD, fontweight="bold")
ax.annotate(f"{old_toxic[-1]:.1f}%", (gaps[-1], old_toxic[-1]), xytext=(0, 10),
            textcoords="offset points", ha="center", fontsize=9.5, color=OLD, fontweight="bold")
ax.annotate(f"{new_incons[0]:.2f}%", (gaps[0], new_incons[0]), xytext=(-8, 10),
            textcoords="offset points", ha="center", fontsize=9.5, color=NEW, fontweight="bold")
ax.annotate(f"{new_incons[-1]:.2f}%", (gaps[-1], new_incons[-1]), xytext=(6, -18),
            textcoords="offset points", ha="center", fontsize=9.5, color=NEW, fontweight="bold")

# ratio callouts
ax.annotate("", xy=(7.15, old_toxic[-1]), xytext=(7.15, old_toxic[0]),
            arrowprops=dict(arrowstyle="<->", color=OLD, lw=1.3), annotation_clip=False)
ax.text(7.25, (old_toxic[0] + old_toxic[-1]) / 2, "~40x", color=OLD, fontsize=10,
        fontweight="bold", va="center")

ax.annotate("", xy=(6.55, new_incons[-1]), xytext=(6.55, new_incons[0]),
            arrowprops=dict(arrowstyle="<->", color=NEW, lw=1.3), annotation_clip=False)
ax.text(6.45, (new_incons[0] + new_incons[-1]) / 2 - 1, "~2.25x", color=NEW, fontsize=10,
        fontweight="bold", va="center", ha="right")

ax.set_xlabel("temporal gap (frames)", fontsize=11, color=INK)
ax.set_ylabel("flagged-pixel fraction (%)", fontsize=11, color=INK)
ax.set_xticks(gaps)
ax.set_xlim(0.5, 7.9)
ax.set_ylim(0, 78)
ax.set_title("Flow degradation with temporal gap:\nold threshold overstates it, FB-consistency corrects it",
             fontsize=13.5, fontweight="bold", color=INK, pad=14)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9.5)

ax.legend(loc="upper left", fontsize=9, frameon=False)

fig.tight_layout()
fig.savefig("fb_consistency_vs_gap.pdf", bbox_inches="tight")
fig.savefig("fb_consistency_vs_gap.png", bbox_inches="tight")
print("wrote fb_consistency_vs_gap.pdf / .png")
