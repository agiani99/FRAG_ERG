"""Figure 6: rebuild-known-molecules benchmark (100 drugs; README 2.28-2.29, 2.33).
 a) recovery of the original molecule by start state and block library (Wilson 95 % CI), numbers from rebuild_summary.txt /
    rebuild_std_summary.txt
 b) recovery from one / two blocks against the number of distractor blocks (restricted library, capping rule off) and at the full library (standard rules), guided vs
    random choice, all 100 molecules (per-molecule mean over the 3 replicate draws, bootstrap 95 % CI over molecules)
 c) the same for molecules with <= 3 blocks to grow
usage: python plot_fig6.py
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = '#1f2328', '#6b7280', '#e5e7eb'
C_STD, C_ORA, C_G, C_R = '#2a6fb0', '#a9c4e4', '#2a6fb0', '#8b9099'
n = 100


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot(per_mol, B=2000, seed=0):
    g = np.random.default_rng(seed); v = per_mol.values
    b = np.array([g.choice(v, len(v)).mean() for _ in range(B)])
    return np.percentile(b, [2.5, 97.5])


# ---- data for b / c
# restricted sweep with the grower's capping rule switched off (--no-closable-check): isolates the competition (README 2.33)
rs = pd.concat([pd.read_csv(os.path.join(here, 'restricted_nocheck_results.csv')),
                pd.read_csv(os.path.join(here, 'restricted_nocheck_fine_results.csv'))], ignore_index=True)  # N = 0,100,1000,10000 + 10,30,300
rs = rs[rs.error.isna()] if 'error' in rs else rs
old = pd.read_csv(os.path.join(here, 'rebuild_results.csv'))
orc = pd.read_csv(os.path.join(here, 'rebuild_rand_results.csv'))
NS = [0, 10, 30, 100, 300, 1000, 10000]
XL = ['0', '10', '30', '100', '300', '1,000', '10,000', 'full\n(48,489)']


def series(cond, rand, stratum):
    """[(mean, lo, hi)] per x position; stratum 'all' or 'le3' (<= 3 blocks to grow)."""
    out = []
    for nd in NS + ['full']:
        if nd == 'full':
            d = (orc[orc.cond == cond + '_rand'] if rand else old[(old.cond == cond) & (old.lib == 'oracle')]).copy()
            d['ref_found'] = d.ref_found.astype(float)
        else:
            d = rs[(rs.cond == cond + ('_rand' if rand else '')) & (rs.n_dist == nd)].copy()
            d['ref_found'] = d.ref_found.astype(float)
        if stratum == 'le3':
            d = d[d.grown_blocks <= 3]
        pm = d.groupby('id').ref_found.mean()
        lo, hi = boot(pm)
        out.append((pm.mean(), lo, hi))
    return np.array(out)


fig = plt.figure(figsize=(16.0, 4.6), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.15, 1.15], wspace=0.30)

# ---- a) bars
ax = fig.add_subplot(gs[0])
data = {'only the most\ninfluential block\ngiven (n1)': (1, 1), 'two connected\nblocks given\n(n2)': (2, 2),
        'largest terminal\nblock missing,\nrandom choice\n(control)': (1, 2),
        'largest terminal\nblock missing,\nErG-guided\n(leaf)': (63, 77)}
x = np.arange(len(data)); w = 0.34
for off, idx, col, lab in ((-w / 2 - 0.01, 0, C_STD, 'standard library'), (w / 2 + 0.01, 1, C_ORA, 'oracle library (+ own blocks)')):
    ks = [v[idx] for v in data.values()]
    ax.bar(x + off, [k / n * 100 for k in ks], width=w, color=col, edgecolor='white', linewidth=2, label=lab, zorder=2)
    for xi, k in zip(x, ks):
        lo, hi = wilson(k, n)
        ax.vlines(xi + off, lo * 100, hi * 100, color=INK, lw=1.2, zorder=3)
        ax.text(xi + off, hi * 100 + 1.5, f'{k}', ha='center', va='bottom', fontsize=8.5, color=INK, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(list(data), fontsize=7.2, color=INK)
ax.set_ylim(0, 100); ax.set_ylabel('original molecule recovered (% of 100 drugs)', fontsize=9, color=INK)
ax.yaxis.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='upper left', fontsize=8, frameon=False)
ax.text(-0.12, 1.03, 'a', transform=ax.transAxes, fontsize=11, fontweight='bold', color=INK)

# ---- b / c) recovery vs library size
def panel(ax, stratum, letter, title, ymax):
    xs = np.arange(len(XL))
    for cond, mk in (('n1', 'o'), ('n2', 's')):
        for rand in (False, True):
            s = series(cond, rand, stratum)
            col = C_R if rand else C_G
            ax.errorbar(xs, s[:, 0] * 100, yerr=[(s[:, 0] - s[:, 1]) * 100, (s[:, 2] - s[:, 0]) * 100], color=col, lw=1.6,
                        ls='--' if rand else '-', marker=mk, ms=6, mfc='white' if rand else col, mec=col, mew=1.4,
                        capsize=2.5, elinewidth=0.9, zorder=3 if not rand else 2)
    ax.axvline(len(NS) - 0.5, color=GRID, lw=1.2, zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels(XL, fontsize=8, color=INK)
    ax.set_xlabel('distractor blocks added to the\nmolecule\'s own blocks', fontsize=8.5, color=INK)
    ax.set_ylim(-3, ymax); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel('recovered (%)', fontsize=9, color=INK)
    ax.yaxis.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=9, color=INK, loc='left', pad=6)
    ax.text(-0.17, 1.03, letter, transform=ax.transAxes, fontsize=11, fontweight='bold', color=INK)

axb = fig.add_subplot(gs[1]); panel(axb, 'all', 'b', 'all 100 molecules', 106)
axc = fig.add_subplot(gs[2]); panel(axc, 'le3', 'c', '≤ 3 blocks to grow', 106)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=C_G, marker='o', mfc=C_G, ls='-', label='guided, one block'),
     Line2D([], [], color=C_G, marker='s', mfc=C_G, ls='-', label='guided, two blocks'),
     Line2D([], [], color=C_R, marker='o', mfc='white', ls='--', label='random, one block'),
     Line2D([], [], color=C_R, marker='s', mfc='white', ls='--', label='random, two blocks')]
axb.legend(handles=h, loc='upper right', fontsize=7.4, frameon=False)
for a_ in (ax, axb, axc):
    for s in ('top', 'right'):
        a_.spines[s].set_visible(False)
    a_.spines['left'].set_color(MUTED); a_.spines['bottom'].set_color(MUTED)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(here, f'fig6_rebuild.{ext}'), bbox_inches='tight')
print('saved')
for stratum in ('all', 'le3'):
    for cond in ('n1', 'n2'):
        for rand in (False, True):
            print(stratum, cond, 'random' if rand else 'guided', [f'{v[0]*100:.1f}' for v in series(cond, rand, stratum)])
