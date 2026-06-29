"""
Architecture comparison diagram: v52–v62 seg head evolution.
Generates: saved/arch_comparison.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
C_FEAT   = '#4A90D9'   # backbone feature boxes
C_OP     = '#E8A838'   # projection / conv ops
C_FUSE   = '#5BAD6F'   # fusion / add ops
C_OUT    = '#C0392B'   # output / classifier
C_AUX    = '#9B59B6'   # aux heads
C_BG     = '#F7F9FC'
C_LINE   = '#555555'
C_ARROW  = '#333333'

def box(ax, x, y, w, h, label, sub='', color=C_FEAT, fontsize=8, alpha=0.92):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02",
                          linewidth=1.2, edgecolor=C_LINE,
                          facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.07 if sub else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', zorder=4)
    if sub:
        ax.text(x, y - 0.13, sub,
                ha='center', va='center', fontsize=6.5,
                color='white', alpha=0.9, zorder=4)

def arr(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.4, style='->'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0'))

def plus(ax, x, y, r=0.08):
    circ = plt.Circle((x, y), r, color=C_FUSE, zorder=5, linewidth=1.2,
                       edgecolor=C_LINE)
    ax.add_patch(circ)
    ax.text(x, y, '+', ha='center', va='center', fontsize=11,
            fontweight='bold', color='white', zorder=6)

fig, axes = plt.subplots(1, 3, figsize=(15, 9))
fig.patch.set_facecolor(C_BG)
titles = [
    'MultiScaleSegHead\n(v52–v60, dilation backbone)',
    'UNetSegHead\n(v61, dilation backbone)',
    'UNetSegHeadV2\n(v62, standard backbone)',
]
subtitles = [
    'feat1+feat2+feat3 → sum → upsample → concat feat0',
    'feat3 → +feat2 → +feat1 (all H/8) → upsample → concat feat0',
    'feat3 H/32 → up+feat2 H/16 → up+feat1 H/8 → up → concat feat0 H/4',
]

# ─── Panel 0: MultiScaleSegHead ──────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(C_BG)
ax.set_xlim(0, 2); ax.set_ylim(0, 9.5)
ax.axis('off')
ax.set_title(titles[0], fontsize=10, fontweight='bold', pad=8)
ax.text(1.0, 9.1, subtitles[0], ha='center', fontsize=7, color='#555', style='italic')

# Backbone features (left column)
feats = [('feat0', 'H/4  256ch',  1.4), ('feat1', 'H/8  512ch', 2.6),
         ('feat2', 'H/8 1024ch',  3.8), ('feat3', 'H/8 2048ch', 4.9)]
for name, size, y in feats:
    box(ax, 0.45, y, 0.7, 0.45, name, size, C_FEAT)

# proj1 proj2 proj3
projs = [(1.25, 2.6), (1.25, 3.8), (1.25, 4.9)]
for px, py in projs:
    box(ax, px, py, 0.5, 0.35, 'proj\n256ch', color=C_OP, fontsize=7)
    arr(ax, 0.8, py, 1.0, py)

# SUM circle
plus(ax, 1.55, 3.8)
arr(ax, 1.5, 2.6, 1.55, 3.65)   # from proj1
arr(ax, 1.5, 3.8, 1.65, 3.8)    # from proj2 (goes right into plus, handled by sum below)
arr(ax, 1.5, 4.9, 1.55, 3.95)   # from proj3
ax.text(1.78, 3.8, '(elem-wise\nsum)', fontsize=6, color='#444', va='center')

# fuse_conv
box(ax, 1.55, 2.85, 0.56, 0.38, 'fuse_conv', 'dil=6 H/8', C_OP)
arr(ax, 1.55, 3.62, 1.55, 3.04)

# upsample
box(ax, 1.55, 2.2, 0.52, 0.38, 'upsample', '→ H/4', C_OP)
arr(ax, 1.55, 2.66, 1.55, 2.39)

# concat feat0
plus(ax, 1.55, 1.65)
ax.text(1.78, 1.65, 'concat\nfeat0', fontsize=6.5, color='#444', va='center')
arr(ax, 0.8, 1.4, 1.4, 1.57)     # feat0 diagonal
arr(ax, 1.55, 2.01, 1.55, 1.73)   # from upsample

# decode convs
box(ax, 1.55, 1.1, 0.56, 0.38, 'decode_conv×2', 'H/4 256ch', C_OP)
arr(ax, 1.55, 1.47, 1.55, 1.29)

# output
box(ax, 1.55, 0.55, 0.56, 0.38, 'conv_seg', '5 classes H/4', C_OUT)
arr(ax, 1.55, 0.91, 1.55, 0.74)

# ─── Panel 1: UNetSegHead ─────────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor(C_BG)
ax.set_xlim(0, 2); ax.set_ylim(0, 9.5)
ax.axis('off')
ax.set_title(titles[1], fontsize=10, fontweight='bold', pad=8)
ax.text(1.0, 9.1, subtitles[1], ha='center', fontsize=7, color='#555', style='italic')

feats1 = [('feat0', 'H/4  256ch', 1.4), ('feat1', 'H/8  512ch', 2.8),
          ('feat2', 'H/8 1024ch', 4.1), ('feat3', 'H/8 2048ch', 5.6)]
for name, size, y in feats1:
    box(ax, 0.45, y, 0.7, 0.45, name, size, C_FEAT)

# Stage 1: proj3
box(ax, 1.3, 5.6, 0.5, 0.38, 'proj3', '256ch H/8', C_OP)
arr(ax, 0.8, 5.6, 1.05, 5.6)
box(ax, 1.3, 5.0, 0.5, 0.38, 'stage1_conv', 'H/8', C_OP)
arr(ax, 1.3, 5.41, 1.3, 5.19)
box(ax, 1.72, 5.0, 0.38, 0.3, 'aux1', 'H/8', C_AUX, fontsize=7)
arr(ax, 1.55, 5.0, 1.53, 5.0)

# Stage 2: += feat2
plus(ax, 1.3, 4.5)
arr(ax, 1.3, 4.81, 1.3, 4.58)   # from stage1
box(ax, 0.82, 4.1, 0.42, 0.32, 'proj2', '256ch', C_OP, fontsize=7)
arr(ax, 0.8, 4.1, 0.6, 4.1)     # feat2→proj2
arr(ax, 1.03, 4.1, 1.22, 4.42)  # proj2→plus
box(ax, 1.3, 3.9, 0.5, 0.35, 'stage2_conv', 'H/8', C_OP)
arr(ax, 1.3, 4.32, 1.3, 4.07)
box(ax, 1.72, 3.9, 0.38, 0.3, 'aux2', 'H/8', C_AUX, fontsize=7)
arr(ax, 1.55, 3.9, 1.53, 3.9)

# Stage 3: += feat1
plus(ax, 1.3, 3.35)
arr(ax, 1.3, 3.72, 1.3, 3.43)
box(ax, 0.82, 2.8, 0.42, 0.32, 'proj1', '256ch', C_OP, fontsize=7)
arr(ax, 0.8, 2.8, 0.6, 2.8)
arr(ax, 1.03, 2.8, 1.22, 3.27)
box(ax, 1.3, 2.7, 0.56, 0.35, 'stage3_conv', 'dil=6 H/8', C_OP)
arr(ax, 1.3, 3.17, 1.3, 2.87)
box(ax, 1.72, 2.7, 0.38, 0.3, 'aux3', 'H/8', C_AUX, fontsize=7)
arr(ax, 1.58, 2.7, 1.53, 2.7)

# upsample
box(ax, 1.3, 2.15, 0.5, 0.35, 'upsample', '→ H/4', C_OP)
arr(ax, 1.3, 2.52, 1.3, 2.32)

# concat feat0
plus(ax, 1.3, 1.65)
arr(ax, 1.3, 1.97, 1.3, 1.73)
arr(ax, 0.8, 1.4, 1.14, 1.57)
ax.text(1.53, 1.65, 'concat\nfeat0', fontsize=6.5, color='#444', va='center')

box(ax, 1.3, 1.1, 0.56, 0.35, 'decode_conv×2', 'H/4', C_OP)
arr(ax, 1.3, 1.47, 1.3, 1.27)
box(ax, 1.3, 0.55, 0.52, 0.35, 'conv_seg', '5cls H/4', C_OUT)
arr(ax, 1.3, 0.92, 1.3, 0.72)

# ─── Panel 2: UNetSegHeadV2 ───────────────────────────────────────────────────
ax = axes[2]
ax.set_facecolor(C_BG)
ax.set_xlim(0, 2.1); ax.set_ylim(0, 9.5)
ax.axis('off')
ax.set_title(titles[2], fontsize=10, fontweight='bold', pad=8)
ax.text(1.05, 9.1, subtitles[2], ha='center', fontsize=6.5, color='#555', style='italic')

# Backbone features — different resolutions (standard strides)
feats2 = [('feat0', 'H/4  256ch',  1.4), ('feat1', 'H/8  512ch',  2.8),
          ('feat2', 'H/16 1024ch', 4.1),  ('feat3', 'H/32 2048ch', 5.6)]
for name, size, y in feats2:
    box(ax, 0.45, y, 0.72, 0.45, name, size, C_FEAT)

# Stage 1: proj3 → stage1_conv at H/32
box(ax, 1.3, 5.6, 0.5, 0.38, 'proj3', '256ch H/32', C_OP, fontsize=7)
arr(ax, 0.81, 5.6, 1.05, 5.6)
box(ax, 1.3, 5.0, 0.5, 0.38, 'stage1_conv', 'H/32', C_OP)
arr(ax, 1.3, 5.41, 1.3, 5.19)
box(ax, 1.78, 5.0, 0.38, 0.3, 'aux1', 'H/32', C_AUX, fontsize=7)
arr(ax, 1.55, 5.0, 1.59, 5.0)

# Stage 2: upsample_add feat2 at H/16
ax.text(1.3, 4.68, '↑ upsample to H/16', ha='center', fontsize=6.5, color='#666')
arr(ax, 1.3, 4.81, 1.3, 4.58, style='->')
plus(ax, 1.3, 4.48)
box(ax, 0.82, 4.1, 0.44, 0.32, 'lateral2', '256ch', C_OP, fontsize=7)
arr(ax, 0.8, 4.1, 0.6, 4.1)
arr(ax, 1.04, 4.1, 1.22, 4.40)
box(ax, 1.3, 3.9, 0.5, 0.35, 'stage2_conv', 'H/16', C_OP)
arr(ax, 1.3, 4.30, 1.3, 4.07)
box(ax, 1.78, 3.9, 0.38, 0.3, 'aux2', 'H/16', C_AUX, fontsize=7)
arr(ax, 1.55, 3.9, 1.59, 3.9)

# Stage 3: upsample_add feat1 at H/8
ax.text(1.3, 3.57, '↑ upsample to H/8', ha='center', fontsize=6.5, color='#666')
arr(ax, 1.3, 3.72, 1.3, 3.43, style='->')
plus(ax, 1.3, 3.33)
box(ax, 0.82, 2.8, 0.44, 0.32, 'lateral1', '256ch', C_OP, fontsize=7)
arr(ax, 0.8, 2.8, 0.6, 2.8)
arr(ax, 1.04, 2.8, 1.22, 3.25)
box(ax, 1.3, 2.7, 0.5, 0.35, 'stage3_conv', 'H/8', C_OP)
arr(ax, 1.3, 3.15, 1.3, 2.87)
box(ax, 1.78, 2.7, 0.38, 0.3, 'aux3', 'H/8', C_AUX, fontsize=7)
arr(ax, 1.55, 2.7, 1.59, 2.7)

# Stage 4: upsample to H/4 and concat feat0
ax.text(1.3, 2.38, '↑ upsample to H/4', ha='center', fontsize=6.5, color='#666')
arr(ax, 1.3, 2.52, 1.3, 2.25, style='->')
plus(ax, 1.3, 1.65)
arr(ax, 1.3, 1.97, 1.3, 1.73)
arr(ax, 0.81, 1.4, 1.14, 1.57)
ax.text(1.53, 1.65, 'concat\nfeat0', fontsize=6.5, color='#444', va='center')

box(ax, 1.3, 1.1, 0.56, 0.35, 'decode_conv×2', 'H/4', C_OP)
arr(ax, 1.3, 1.47, 1.3, 1.27)
box(ax, 1.3, 0.55, 0.52, 0.35, 'conv_seg', '5cls H/4', C_OUT)
arr(ax, 1.3, 0.92, 1.3, 0.72)

# ─── Legend ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_FEAT,  label='Backbone feature'),
    mpatches.Patch(color=C_OP,    label='Conv / projection'),
    mpatches.Patch(color=C_FUSE,  label='Add / concat'),
    mpatches.Patch(color=C_AUX,   label='Aux seg head'),
    mpatches.Patch(color=C_OUT,   label='Output classifier'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=5,
           fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

# ─── Backbone note ────────────────────────────────────────────────────────────
fig.text(0.34, 0.955,
         'Backbone (dilation): strides=[1,2,1,1], dilations=[1,1,2,4] → feat1/2/3 all at H/8',
         ha='center', fontsize=8, color='#555', style='italic')
fig.text(0.83, 0.955,
         'Backbone (standard): strides=[1,2,2,2], dilations=[1,1,1,1] → true multi-resolution',
         ha='center', fontsize=8, color='#555', style='italic')

plt.suptitle('Seg Head Architecture Comparison  (v52–v62)', fontsize=13,
             fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0.06, 1, 0.97])

out = 'saved/arch_comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=C_BG)
print(f'Saved → {out}')
