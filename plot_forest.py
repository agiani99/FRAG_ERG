"""
Figure 2: per-target paired difference in MCC against ECFP4 (decoy_fusion_targets.csv) for ErG v2 and RDKit ErG,
with the pooled mean and bootstrap 90 % CI (one-sided 5 % non-inferiority) against the margin -delta.
usage: python plot_forest.py [--delta 0.10]
"""
import os, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--delta', type=float, default=0.10)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))

t = pd.read_csv(os.path.join(here, 'decoy_fusion_targets.csv'))
names = pd.read_csv(os.path.join(here, 'retro2_results.csv'), usecols=['target', 'name']).drop_duplicates('target')
t = t.merge(names, on='target', how='left')
t['d_erg'] = t.mcc_ErG - t.mcc_ECFP4
t['d_rdk'] = t.mcc_ErG_RDKit - t.mcc_ECFP4
t = t.sort_values('d_erg').reset_index(drop=True)


def ci90(x, n=20000, seed=1):
    g = np.random.default_rng(seed)
    b = np.array([g.choice(x, len(x)).mean() for _ in range(n)])
    return np.percentile(b, [5, 95])


C_ERG, C_RDK = '#2a6fb0', '#d9822b'      # two hues, distinct in lightness; marker shapes also differ
INK, MUTED, GRID = '#1f2328', '#6b7280', '#e5e7eb'

n = len(t)
fig, ax = plt.subplots(figsize=(7.2, 9.2), dpi=300)
y = np.arange(n) + 2.5                   # summary rows sit below at y = 0 and 1
ax.axvspan(-a.delta, 0, color='#f3f4f6', zorder=0)
for yi in y:
    ax.axhline(yi, color=GRID, lw=0.5, zorder=1)
ax.scatter(t.d_erg, y, s=26, marker='o', color=C_ERG, edgecolor='white', linewidth=0.6, zorder=3, label='ErG v2 (this work)')
ax.scatter(t.d_rdk, y, s=26, marker='D', color=C_RDK, edgecolor='white', linewidth=0.6, zorder=3, label='ErG, RDKit reduced graph')
ax.axvline(0, color=INK, lw=0.9, zorder=2)
ax.axvline(-a.delta, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)

for yy, col, mk, key, lab in ((1, C_ERG, 'o', 'd_erg', 'ErG v2'), (0, C_RDK, 'D', 'd_rdk', 'RDKit ErG')):
    m = t[key].mean(); lo, hi = ci90(t[key].values)
    ax.plot([lo, hi], [yy, yy], color=col, lw=2.2, solid_capstyle='butt', zorder=3)
    ax.scatter([m], [yy], s=60, marker=mk, color=col, edgecolor='white', linewidth=0.8, zorder=4)
    ax.text(0.30, yy, f'{m:+.3f}  [{lo:+.3f}, {hi:+.3f}]', va='center', ha='left', fontsize=8, color=INK)
    ax.text(-0.485, yy, f'mean, {lab}', va='center', ha='left', fontsize=8, color=INK, fontweight='bold')
ax.axhline(2.0, color=INK, lw=0.6)

lbl = [f'{(nm if isinstance(nm, str) else tg)[:44]}' for nm, tg in zip(t.name, t.target)]
ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=6.4, color=INK)
ax.set_ylim(-0.8, n + 3.4)
ax.set_xlim(-0.50, 0.85)
ax.set_xlabel('Paired difference in MCC against ECFP4  (positive: the ErG scorer is better)', fontsize=8.5, color=INK)
ax.tick_params(axis='x', labelsize=8, colors=INK)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(MUTED)
ax.tick_params(axis='y', length=0)

ax.text(-a.delta - 0.01, n + 3.0, f'margin −{a.delta:.2f}', fontsize=7.5, ha='right', va='bottom', color=INK)
ax.text(0.01, n + 3.0, 'no difference', fontsize=7.5, ha='left', va='bottom', color=INK)
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.075), fontsize=7.5, frameon=False, handletextpad=0.4)
fig.text(0.01, 0.005, f'{n} targets, balanced sets with size-matched decoys, top-k rule. Summary rows: mean over targets and '
         'bootstrap 90 % CI (one-sided 5 %).\nNon-inferior if the lower bound is above the margin.', fontsize=6.5, color=MUTED, ha='left', va='bottom')
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(here, f'fig2_forest.{ext}'))
print('saved fig2_forest.png / .pdf')
