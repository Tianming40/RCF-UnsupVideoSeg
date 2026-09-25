#!/usr/bin/env python3
"""
100%-stacked bar chart: per-frame mIoU change vs. v9, for the same
versions as g0_miou_progression.py's milestones (v53 standing in for
v47, which has no per-frame log in this batch), g0 only (422 frames:
213 instrument + 209 tissue).
"""
import matplotlib.pyplot as plt

UP = "#2f6f5e"     # improved
DOWN = "#b5563c"   # declined
SAME = "#c9c5b8"   # unchanged
INK = "#20241f"
MUTED = "#7a7a70"
GRID = "#dedad0"

# version, up%, down%, same%, n, mechanism label (matches g0_miou_progression.py)
data = [
    ("v53",  50.5, 49.5, 0.0, 422, "$w_{dino}=0.1$ / DINO loss"),
    ("v64",  54.3, 45.7, 0.0, 422, "MultiScaleSegHead + ASPP"),
    ("v83",  55.9, 44.1, 0.0, 422, "manually removed\nlow-quality training data"),
    ("v102", 58.3, 41.0, 0.7, 422, "GT flow downsample\nbilinear $\\rightarrow$ area"),
]

fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ys = list(range(len(data)))[::-1]  # v53 at bottom visually reversed -> keep v53 top, v102 bottom? use natural top-to-bottom order
ys = list(range(len(data)))

bar_h = 0.6
for i, (ver, up, down, same, n, mech) in enumerate(data):
    y = len(data) - 1 - i  # v53 on top, v102 on bottom
    left = 0
    for val, color, label in [(up, UP, "improved"), (same, SAME, "unchanged"), (down, DOWN, "declined")]:
        if val <= 0:
            continue
        ax.barh(y, val, left=left, height=bar_h, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        if val >= 6:
            ax.text(left + val/2, y, f"{val:.0f}%", ha="center", va="center",
                     fontsize=10.5, fontweight="bold",
                     color="white" if color != SAME else INK, zorder=4)
        left += val
    # left-side label: version (bold) above, mechanism (small, muted) below --
    # same two-tier style as g0_miou_progression.py's point annotations
    ax.text(-2, y + 0.14, ver, ha="right", va="center", fontsize=12,
             fontweight="bold", color=INK)
    ax.text(-2, y - 0.16, mech, ha="right", va="center", fontsize=8.3,
             color=MUTED, linespacing=1.35)

ax.set_xlim(0, 100)
ax.set_ylim(-0.6, len(data)-0.4)
ax.set_yticks([])
ax.set_xticks([0,25,50,75,100])
ax.set_xticklabels(["0%","25%","50%","75%","100%"], fontsize=9.5, color=MUTED)
ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.axvline(50, color=INK, linewidth=1, linestyle=(0,(2,2)), alpha=0.35, zorder=2)

# legend
handles = [plt.Rectangle((0,0),1,1, color=UP), plt.Rectangle((0,0),1,1, color=SAME), plt.Rectangle((0,0),1,1, color=DOWN)]
ax.legend(handles, ["mIoU improved vs. baseline", "unchanged", "mIoU declined vs. baseline"],
          loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=10)

ax.text(0.5, 1.16, "Per Frame mIoU Change vs. Baseline", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=15.5, fontweight="bold", color=INK)
ax.text(0.5, 1.05, "Grasp0 dataset – 213 instrument annotations and 209 tissue annotations",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5, color=MUTED)

fig.tight_layout()
fig.savefig("g0_frame_updown.pdf", bbox_inches="tight")
fig.savefig("g0_frame_updown.png", bbox_inches="tight")
print("wrote g0_frame_updown.pdf / .png")
