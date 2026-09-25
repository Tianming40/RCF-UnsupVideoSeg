#!/usr/bin/env python3
"""
Same 100%-stacked per-frame up/down/same chart as make_g0_frame_updown.py,
but split into instrument vs. tissue sub-bars per milestone (2D: mechanism
x task), instead of one combined bar per version. Version codes (v53 etc.)
are dropped -- only the mechanism description identifies each group, since
g0_miou_progression.png already carries the version numbers.
"""
import matplotlib.pyplot as plt

UP = "#2f6f5e"
DOWN = "#b5563c"
SAME = "#c9c5b8"
INK = "#20241f"
MUTED = "#7a7a70"
GRID = "#dedad0"

# mechanism, [(task, up%, down%, same%, n), (task, up%, down%, same%, n)]
groups = [
    ("$w_{dino}=0.1$\nDINO loss", [
        ("Instrument", 30.0, 70.0, 0.0, 213),
        ("Tissue",     71.3, 28.7, 0.0, 209),
    ]),
    ("MultiScaleSegHead\n+ ASPP", [
        ("Instrument", 32.4, 67.6, 0.0, 213),
        ("Tissue",     76.6, 23.4, 0.0, 209),
    ]),
    ("manually removed\nlow-quality training data", [
        ("Instrument", 43.7, 56.3, 0.0, 213),
        ("Tissue",     68.4, 31.6, 0.0, 209),
    ]),
    ("GT flow downsample\nbilinear $\\rightarrow$ area", [
        ("Instrument", 40.4, 58.7, 0.9, 213),
        ("Tissue",     76.6, 23.0, 0.5, 209),
    ]),
]

row_h = 1.0
row_gap = 0.12      # gap between the 2 rows within a group
group_gap = 0.55     # extra gap between groups

fig, ax = plt.subplots(figsize=(11, 7.6), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bar_h = 0.68
y = 0.0
group_centers = []   # (y_center, mechanism)
sep_ys = []           # y positions for separator lines between groups

for gi, (mech, rows) in enumerate(groups):
    row_ys = []
    for task, up, down, same, n in rows:
        left = 0
        for val, color in [(up, UP), (same, SAME), (down, DOWN)]:
            if val <= 0:
                continue
            ax.barh(y, val, left=left, height=bar_h, color=color, edgecolor="white", linewidth=1.1, zorder=3)
            if val >= 7:
                ax.text(left + val/2, y, f"{val:.0f}%", ha="center", va="center",
                         fontsize=9.5, fontweight="bold",
                         color="white" if color != SAME else INK, zorder=4)
            left += val
        ax.text(-2, y, task, ha="right", va="center", fontsize=9.5, color=MUTED)
        row_ys.append(y)
        y += row_h + row_gap
    group_centers.append((sum(row_ys) / len(row_ys), mech))
    y += group_gap
    if gi < len(groups) - 1:
        sep_ys.append(y - group_gap / 2)

ax.invert_yaxis()  # first group reads at the top

ax.set_xlim(0, 100)
ax.set_ylim(y - group_gap - 0.5, -0.5)
ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9.5, color=MUTED)
ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.axvline(50, color=INK, linewidth=1, linestyle=(0, (2, 2)), alpha=0.35, zorder=2)

MECH_X = -22  # further left than the task labels (-2), its own column
for yc, mech in group_centers:
    ax.text(MECH_X, yc, mech, ha="right", va="center", fontsize=10.5,
             fontweight="bold", color=INK, linespacing=1.45)

for sy in sep_ys:
    ax.axhline(sy, color=GRID, linewidth=1, xmin=-0.62, xmax=1.0, clip_on=False, zorder=1)

handles = [plt.Rectangle((0, 0), 1, 1, color=UP), plt.Rectangle((0, 0), 1, 1, color=SAME), plt.Rectangle((0, 0), 1, 1, color=DOWN)]
ax.legend(handles, ["mIoU improved vs. baseline", "unchanged", "mIoU declined vs. baseline"],
          loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=3, frameon=False, fontsize=10)

ax.text(0.5, 1.065, "Per Frame mIoU Change vs. Baseline, by Task", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=15.5, fontweight="bold", color=INK)
ax.text(0.5, 1.02, "Grasp0 dataset – 213 instrument annotations and 209 tissue annotations",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5, color=MUTED)

fig.subplots_adjust(left=0.5)
fig.savefig("g0_frame_updown_task.pdf", bbox_inches="tight")
fig.savefig("g0_frame_updown_task.png", bbox_inches="tight")
print("wrote g0_frame_updown_task.pdf / .png")
