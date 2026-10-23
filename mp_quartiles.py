"""Matched-pair table for Figure 4: P(kept) by global-ECFP4 quartile, ErG >= tau vs below (same-assay pairs, zone block ECFP4 <= 0.35,
definition 1: kept = other pChEMBL >= 7, lost = < 6, grey ignored). Cluster bootstrap over targets. Writes mp_quartiles.csv."""
import os, numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(here, '..', 'BIOMIM', 'mp_assay_pairs.csv'),
                usecols=['target', 'p_other', 'erg', 'block_ecfp4', 'ecfp4'])
d = d[(d.block_ecfp4 <= 0.35) & ((d.p_other >= 7) | (d.p_other < 6))].copy()
d['kept'] = (d.p_other >= 7).astype(np.int8)
d['hi'] = d.erg >= 0.9
q = d.ecfp4.quantile([0.25, 0.5, 0.75]).values
d['q'] = np.searchsorted(q, d.ecfp4.values, side='right')   # 0 = lowest global ECFP4
print('zone pairs', len(d), 'quartile edges', np.round(q, 3))
g = d.groupby(['target', 'q', 'hi']).kept.agg(['sum', 'count']).reset_index()
tg = g.target.unique(); rng = np.random.default_rng(0)
piv = {k: v for k, v in g.groupby('target')}
rows = []
def stat(sel):
    x = pd.concat([piv[t] for t in sel]); a = x.groupby(['q', 'hi'])[['sum', 'count']].sum()
    return (a['sum'] / a['count']), a['count']
p0, n0 = stat(tg)
B = []
for _ in range(500):
    p, _n = stat(rng.choice(tg, len(tg)))
    B.append(p)
B = pd.DataFrame(B)
for (qq, hi), v in p0.items():
    lo, up = np.percentile(B[(qq, hi)].dropna(), [2.5, 97.5])
    rows.append(dict(q=qq, hi=hi, p_kept=v, lo=lo, hi_ci=up, n=int(n0[(qq, hi)])))
r = pd.DataFrame(rows); r.to_csv(os.path.join(here, 'mp_quartiles.csv'), index=False)
dd = pd.DataFrame({qq: B[(qq, True)] - B[(qq, False)] for qq in range(4)})
out=[]
for qq in range(4):
    lo_,hi_=np.percentile(dd[qq],[2.5,97.5]); out.append(dict(q=qq,diff=p0[(qq,True)]-p0[(qq,False)],lo=lo_,hi=hi_))
    print(qq, 'diff', round(p0[(qq, True)] - p0[(qq, False)], 3), np.round(np.percentile(dd[qq], [2.5, 97.5]), 3))
print(r)

pd.DataFrame(out).to_csv(os.path.join(here, 'mp_quartiles_diff.csv'), index=False)
