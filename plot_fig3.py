"""
Figure 3: complementarity of ErG v2 and ECFP4 (decoy_fusion_targets.csv).
 a) per-target MCC, ErG v2 against ECFP4, coloured by whether the rank fusion beats both single scorers
 b) which actives are found: by both, ErG only, ECFP4 only, neither (top-k rule, summed over targets)
 c) pooled MCC of the scorers
usage: python plot_fig3.py
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
t = pd.read_csv(os.path.join(here, 'decoy_fusion_targets.csv'))

C_ERG, C_ECF, C_RDK, C_FUS = '#2a6fb0', '#7a5aa6', '#d9822b', '#2f8f5b'
INK, MUTED, GRID, NEUT = '#1f2328', '#6b7280', '#e5e7eb', '#b8bec7'


def mcc(tp, fp, fn, tn):
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


fig = plt.figure(figsize=(11.2, 4.4), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 0.85], wspace=0.42)

# ---- a) scatter
ax = fig.add_subplot(gs[0])
best = np.maximum(t.mcc_ErG, t.mcc_ECFP4)
win = t.mcc_FUSION > best
ax.plot([-0.1, 1.05], [-0.1, 1.05], color=MUTED, lw=0.8, zorder=1)
ax.scatter(t.mcc_ECFP4[~win], t.mcc_ErG[~win], s=34, color=NEUT, edgecolor='white', linewidth=0.7, zorder=3,
           label=f'fusion not above both ({int((~win).sum())})')
ax.scatter(t.mcc_ECFP4[win], t.mcc_ErG[win], s=34, color=C_FUS, edgecolor='white', linewidth=0.7, zorder=4,
           label=f'fusion above both ({int(win.sum())})')
ax.set_xlim(-0.1, 1.05); ax.set_ylim(-0.1, 1.05); ax.set_box_aspect(1)
ax.set_xlabel('MCC, ECFP4', fontsize=9, color=INK); ax.set_ylabel('MCC, ErG v2', fontsize=9, color=INK)
ax.legend(loc='upper left', fontsize=7.5, frameon=False, handletextpad=0.3, borderaxespad=0.2)
ax.text(-0.14, 1.04, 'a', transform=ax.transAxes, fontsize=11, fontweight='bold', color=INK, va='bottom')
ax.grid(color=GRID, lw=0.5, zorder=0)

# ---- b) stacked bar of the actives (n = total actives)
ax2 = fig.add_subplot(gs[1])
tot = int(t.n_act.sum())
both, eo, co = int(t.tp_both.sum()), int(t.tp_erg_only.sum()), int(t.tp_ecfp_only.sum())
neither = tot - both - eo - co
parts = [('found by both', both, C_FUS), ('ErG v2 only', eo, C_ERG), ('ECFP4 only', co, C_ECF), ('neither', neither, NEUT)]
left = 0
for k, (lab, v, col) in enumerate(parts):
    ax2.barh(0, v / tot * 100, left=left, height=0.9, color=col, edgecolor='white', linewidth=2)
    left += v / tot * 100
ax2.set_xlim(0, 100); ax2.set_ylim(-1.7, 1.7); ax2.set_yticks([])
ax2.set_xlabel('% of the 8,480 actives (top-k rule)', fontsize=9, color=INK)
cum = np.cumsum([0] + [v / tot * 100 for _, v, _ in parts])
mid = [(cum[k] + cum[k + 1]) / 2 for k in range(4)]
ax2.text(mid[0], 0, f'found by both\n{parts[0][1] / tot * 100:.1f} %', ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
for k, ypos, va in ((1, -0.95, 'top'), (2, 0.95, 'bottom'), (3, -0.95, 'top')):
    lab, v, col = parts[k]
    ax2.plot([mid[k], mid[k]], [-0.45 if ypos < 0 else 0.45, ypos * 0.92], color=MUTED, lw=0.6)
    ax2.text(mid[k], ypos, f'{lab}\n{v / tot * 100:.1f} %', ha='center', va=va, fontsize=8, color=INK)
for s in ('top', 'right', 'left'):
    ax2.spines[s].set_visible(False)
ax2.spines['bottom'].set_color(MUTED)
ax2.text(-0.14, 1.04, 'b', transform=ax2.transAxes, fontsize=11, fontweight='bold', color=INK, va='bottom')

# ---- c) pooled MCC
ax3 = fig.add_subplot(gs[2])
rows = []
for nm, lab, col in (('ECFP4', 'ECFP4', C_ECF), ('ErG', 'ErG v2', C_ERG), ('ErG_RDKit', 'RDKit ErG', C_RDK),
                     ('FUSION', 'Fusion v2 + ECFP4', C_FUS)):
    v = mcc(*(int(t[f'{k}_{nm}'].sum()) for k in ('tp', 'fp', 'fn', 'tn')))
    rows.append((lab, v, col))
for i, (lab, v, col) in enumerate(rows):
    ax3.bar(i, v, width=0.55, color=col, edgecolor='white', linewidth=2)
    ax3.text(i, v + 0.012, f'{v:.3f}', ha='center', va='bottom', fontsize=8, color=INK)
ax3.set_xticks(range(len(rows))); ax3.set_xticklabels([r[0] for r in rows], fontsize=7.5, color=INK, rotation=25, ha='right')
ax3.set_ylim(0, 0.7); ax3.set_ylabel('pooled MCC', fontsize=9, color=INK)
for s in ('top', 'right'):
    ax3.spines[s].set_visible(False)
ax3.spines['left'].set_color(MUTED); ax3.spines['bottom'].set_color(MUTED)
ax3.yaxis.grid(color=GRID, lw=0.5); ax3.set_axisbelow(True)
ax3.text(-0.14, 1.04, 'c', transform=ax3.transAxes, fontsize=11, fontweight='bold', color=INK, va='bottom')

for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(here, f'fig3_complementarity.{ext}'), bbox_inches='tight')
print('saved', [(l, round(v, 3)) for l, v, _ in rows], 'fusion above both:', int(win.sum()), '/', len(t))
