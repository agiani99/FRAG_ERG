"""Figure 4 from mp_quartiles.csv: activity kept (same-assay matched pairs) by global-ECFP4 quartile, ErG >= 0.9 vs below, with
cluster-bootstrap 95 % CIs over targets; right panel: the difference. usage: python plot_fig4.py"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
r = pd.read_csv(os.path.join(here, 'mp_quartiles.csv'))
C_HI, C_LO, INK, MUTED, GRID = '#2a6fb0', '#9aa1ab', '#1f2328', '#6b7280', '#e5e7eb'
labs = ['lowest\n(≤ 0.625)\nlarge change', '0.625–0.692', '0.692–0.750', 'highest\n(> 0.750)\nsmall change']
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=300, gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.28))
x = np.arange(4)
for hi, col, off, mk, lab in ((True, C_HI, -0.08, 'o', 'ErG ≥ 0.9'), (False, C_LO, 0.08, 's', 'ErG < 0.9')):
    s = r[r.hi == hi].sort_values('q')
    ax.vlines(x + off, s.lo, s.hi_ci, color=col, lw=2.2, zorder=2)
    ax.scatter(x + off, s.p_kept, s=46, marker=mk, color=col, edgecolor='white', linewidth=0.8, zorder=3, label=lab)
nn = {True: r[r.hi].sort_values('q').n.values, False: r[~r.hi].sort_values('q').n.values}
for xi in range(4):
    ax.text(xi, 0.568, f'n = {nn[True][xi] / 1000:.0f}k vs {nn[False][xi] / 1000:.0f}k', ha='center', fontsize=6.8, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7.5, color=INK)
ax.set_ylim(0.56, 1.0); ax.set_ylabel('fraction of pairs that keep the activity', fontsize=9, color=INK)
ax.set_xlabel('quartile of global ECFP4 similarity within the pair', fontsize=9, color=INK)
ax.yaxis.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='lower right', fontsize=8, frameon=False, bbox_to_anchor=(1.0, 0.1))
ax.text(-0.14, 1.04, 'a', transform=ax.transAxes, fontsize=11, fontweight='bold', color=INK)

# difference with its own bootstrap CI (re-read: stored per cell only, so recompute from cell CIs is not valid) -> use mp_quartiles_diff.csv
dfile = os.path.join(here, 'mp_quartiles_diff.csv')
dd = pd.read_csv(dfile)
ax2.axvline(0, color=INK, lw=0.9)
y = np.arange(4)[::-1]
ax2.hlines(y, dd.lo * 100, dd.hi * 100, color=C_HI, lw=2.2, zorder=2)
ax2.scatter(dd['diff'] * 100, y, s=46, color=C_HI, edgecolor='white', linewidth=0.8, zorder=3)
for yi, v in zip(y, dd['diff']):
    ax2.text(dd.hi.max() * 100 + 1.5, yi, f'{v * 100:+.1f}', va='center', fontsize=8, color=INK)
ax2.set_yticks(y); ax2.set_yticklabels(['lowest', '2nd', '3rd', 'highest'], fontsize=8, color=INK)
ax2.set_xlim(-2, 32); ax2.set_ylim(-0.6, 3.6)
ax2.set_xlabel('gain in kept fraction, ErG ≥ 0.9 minus below\n(percentage points; 95 % CI)', fontsize=9, color=INK)
ax2.set_ylabel('ECFP4 quartile', fontsize=9, color=INK)
ax2.xaxis.grid(color=GRID, lw=0.5); ax2.set_axisbelow(True)
for a_ in (ax, ax2):
    for s in ('top', 'right'):
        a_.spines[s].set_visible(False)
    a_.spines['left'].set_color(MUTED); a_.spines['bottom'].set_color(MUTED)
ax2.text(-0.30, 1.04, 'b', transform=ax2.transAxes, fontsize=11, fontweight='bold', color=INK)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(here, f'fig4_matched_pairs.{ext}'), bbox_inches='tight')
print('saved')
