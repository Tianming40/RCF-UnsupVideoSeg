#!/usr/bin/env python3
"""
Three-bar comparison of the comb-artifact row-difference metric before/after
the bwdif deinterlace fix, against this dataset's own native-progressive
frames as the realistic reference ceiling (README.md:1644-1656).

  Old (naive row-duplication):        0.499   gap to reference = 0.086
  New (bwdif-processed TFF frames):   0.553   gap to reference = 0.032
  Reference (native progressive):     0.585   —

New method closes ~63% of the gap ((0.086-0.032)/0.086 = 62.8%).
"""
import matplotlib.pyplot as plt

OLD = "#b5563c"    # terracotta -- old duplication method
NEW = "#2f6f5e"     # teal -- new bwdif method
REF = "#7a7a70"     # muted grey -- reference ceiling (not a "result", a target)
INK = "#20241f"
MUTED = "#7a7a70"
GRID = "#dedad0"

labels = ["Old\n(row-duplication)", "New\n(bwdif-corrected)", "Reference\n(native progressive)"]
values = [0.499, 0.553, 0.585]
colors = [OLD, NEW, REF]

fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for t in [0.40, 0.45, 0.50, 0.55, 0.60]:
    ax.axhline(t, color=GRID, linewidth=1, zorder=0)

xs = [0, 1, 2]
bars = ax.bar(xs, values, width=0.55, color=colors, zorder=3, edgecolor="white", linewidth=1.2)

for x, v, c in zip(xs, values, colors):
    ax.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=12.5,
            fontweight="bold", color=c)

# reference line across the whole plot
ax.axhline(values[2], color=REF, linewidth=1.3, linestyle=(0, (4, 3)), zorder=2, alpha=0.7)

# gap annotations: old -> reference, new -> reference
gap_x_old = 0
gap_x_new = 1
ax.annotate("", xy=(gap_x_old, values[2]), xytext=(gap_x_old, values[0]),
            arrowprops=dict(arrowstyle="<->", color=OLD, lw=1.4), zorder=4)
ax.text(gap_x_old - 0.32, (values[0] + values[2]) / 2, "gap\n0.086", color=OLD,
        fontsize=9.5, fontweight="bold", ha="center", va="center")

ax.annotate("", xy=(gap_x_new, values[2]), xytext=(gap_x_new, values[1]),
            arrowprops=dict(arrowstyle="<->", color=NEW, lw=1.4), zorder=4)
ax.text(gap_x_new + 0.32, (values[1] + values[2]) / 2, "gap\n0.032", color=NEW,
        fontsize=9.5, fontweight="bold", ha="center", va="center")

ax.text(0.5, 0.615, "closes ~63% of the gap", ha="center", va="bottom",
        fontsize=11, fontweight="bold", color=INK,
        transform=ax.transData)

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=10.5, color=INK)
ax.set_ylabel("comb metric (row-difference ratio)", fontsize=11, color=INK)
ax.set_ylim(0.40, 0.63)
ax.set_title("Deinterlacing fix: closing the gap to native-progressive quality",
             fontsize=12.5, fontweight="bold", color=INK, pad=14)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9.5)

fig.tight_layout()
fig.savefig("deinterlace_comb_metric.pdf", bbox_inches="tight")
fig.savefig("deinterlace_comb_metric.png", bbox_inches="tight")
print("wrote deinterlace_comb_metric.pdf / .png")
