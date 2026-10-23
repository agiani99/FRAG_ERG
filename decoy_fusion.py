"""
Fusion and complementarity of ErG v2 and ECFP4 on the balanced decoy test (same sets as decoy_eval.py, same seed).

Scorers (similarity to the reference): ErG, ECFP4, FUSION = mean of the within-set percentile ranks of the two
(parameter-free, nothing tuned), size_only. Rule as in decoy_eval.py: top k = n_actives predicted active.
Reports per-target confusion + MCC, the paired non-inferiority test ErG vs ECFP4 (margin --delta, bootstrap over
targets), fusion vs the better single scorer, and the overlap of the true positives of ErG and ECFP4
(share of actives found by only one of them).

Writes decoy_fusion_targets.csv and decoy_fusion_summary.txt. decoy_eval outputs are not touched.
usage: python decoy_fusion.py [--per-target 300] [--seed 0] [--delta 0.10]
"""
import paths
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from scipy.stats import rankdata, wilcoxon, spearmanr
from rdkit.Chem import rdReducedGraphs as RG

import retro as R
import erg_v2 as E
import grow as G
import retro2 as R2

RDLogger.DisableLog('rdApp.*')
ap = argparse.ArgumentParser()
ap.add_argument('--per-target', type=int, default=300)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--delta', type=float, default=0.10)
ap.add_argument('--decoys', default=paths.data('CHEMBL37_notarget.parquet'))
a = ap.parse_args()
rng = np.random.default_rng(a.seed)


def topk_pred(sc, k, rng):
    """boolean predicted-active vector: top k by score, ties broken at random"""
    order = np.lexsort((rng.random(len(sc)), -sc))
    p = np.zeros(len(sc), bool)
    p[order[:k]] = True
    return p


def conf(p, y):
    return (int((p & (y == 1)).sum()), int((p & (y == 0)).sum()),
            int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum()))


res = pd.read_csv(os.path.join(R.HERE, 'retro2_results.csv'))
dec = pd.read_parquet(a.decoys, columns=['SMILES']).SMILES.drop_duplicates().tolist()
dm = [R.neutral(s) for s in dec]
dec = [(Chem.MolToSmiles(m), m.GetNumHeavyAtoms()) for m in dm if m is not None and 12 <= m.GetNumHeavyAtoms() <= 60]
dec_ha = np.array([h for _, h in dec])
print(f'{len(dec)} decoys', flush=True)

d = pd.read_parquet(R.CHEMBL, columns=['chembl_target_id', 'molecule_chembl_id', 'SMILES',
                                       'pchembl_value']).dropna(subset=['pchembl_value'])
rows = []
for _, r in res.iterrows():
    ref = Chem.MolFromSmiles(r.ref)
    sub = d[(d.chembl_target_id == r.target) & (d.pchembl_value >= 7)].drop_duplicates('molecule_chembl_id')
    sub = sub.sample(min(a.per_target, len(sub)), random_state=a.seed)
    pos = [m for m in (R.neutral(s) for s in sub.SMILES) if m is not None and Chem.MolToSmiles(m) != r.ref
           and 12 <= m.GetNumHeavyAtoms() <= 60]
    if len(pos) < 20:
        continue
    used, neg_idx = set(), []
    for m in pos:
        cand = np.nonzero(dec_ha == m.GetNumHeavyAtoms())[0]
        cand = [c for c in cand if c not in used]
        if not cand:
            cand = [c for c in np.argsort(np.abs(dec_ha - m.GetNumHeavyAtoms())) if c not in used][:20]
        c = int(rng.choice(cand))
        used.add(c); neg_idx.append(c)
    neg = [Chem.MolFromSmiles(dec[c][0]) for c in neg_idx]
    ms = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    tg = G.Target(E.erg_raw(ref).astype(np.int64))
    V = np.array([E.erg_raw(m) for m in ms]).astype(np.int64)
    erg = tg.tanimoto(V)
    ecf = np.array(DataStructs.BulkTanimotoSimilarity(R.fp(ref), [R.fp(m) for m in ms]))
    fus = (rankdata(erg) + rankdata(ecf)) / (2 * len(ms))
    # RDKit reduced-graph ErG (Stiefl): 315 values, rings collapsed to one node; same continuous (dot) Tanimoto
    rv = np.array([RG.GetErGFingerprint(m) for m in ms], dtype=np.float64)
    rt = RG.GetErGFingerprint(ref).astype(np.float64)
    dot = rv @ rt
    rdk = dot / np.maximum((rv * rv).sum(1) + (rt * rt).sum() - dot, 1e-12)
    fus2 = (rankdata(rdk) + rankdata(ecf)) / (2 * len(ms))
    k = int(y.sum())
    pr = {'ErG': topk_pred(erg, k, rng), 'ECFP4': topk_pred(ecf, k, rng), 'FUSION': topk_pred(fus, k, rng),
          'ErG_RDKit': topk_pred(rdk, k, rng), 'FUSION_RDKit': topk_pred(fus2, k, rng)}
    row = dict(target=r.target, n_act=k)
    for nm, p in pr.items():
        tp, fp_, fn, tn = conf(p, y)
        row[f'mcc_{nm}'] = R2.mcc_from(tp, fp_, fn, tn)
        row[f'tp_{nm}'], row[f'fp_{nm}'], row[f'fn_{nm}'], row[f'tn_{nm}'] = tp, fp_, fn, tn
    a_ = pr['ErG'] & (y == 1); b_ = pr['ECFP4'] & (y == 1)
    row.update(tp_both=int((a_ & b_).sum()), tp_erg_only=int((a_ & ~b_).sum()), tp_ecfp_only=int((~a_ & b_).sum()))
    row['rho_erg_rdkit'] = float(spearmanr(erg, rdk)[0])
    row['jaccard_pred'] = float((pr['ErG'] & pr['ECFP4']).sum() / (pr['ErG'] | pr['ECFP4']).sum())
    rows.append(row)
    print(f'{r.target} n={k}  ErG {row["mcc_ErG"]:+.2f}  ECFP4 {row["mcc_ECFP4"]:+.2f}  FUSION {row["mcc_FUSION"]:+.2f}',
          flush=True)

t = pd.DataFrame(rows)
t.to_csv(os.path.join(R.HERE, 'decoy_fusion_targets.csv'), index=False)


def boot(x, n=20000, seed=1):
    g = np.random.default_rng(seed)
    b = np.array([g.choice(x, len(x)).mean() for _ in range(n)])
    return np.percentile(b, [2.5, 5, 95, 97.5])


out = [f'{len(t)} targets, balanced (50 % actives vs size-matched no-target decoys), top k = n_actives', '']
out.append(f'{"scorer":13s}{"TP":>8s}{"FP":>8s}{"FN":>8s}{"TN":>8s}{"pooled MCC":>12s}{"mean MCC":>10s}{"median MCC":>12s}')
for nm in ('ErG', 'ErG_RDKit', 'ECFP4', 'FUSION', 'FUSION_RDKit'):
    tp, fp_, fn, tn = (int(t[f'{k}_{nm}'].sum()) for k in ('tp', 'fp', 'fn', 'tn'))
    out.append(f'{nm:13s}{tp:8d}{fp_:8d}{fn:8d}{tn:8d}{R2.mcc_from(tp, fp_, fn, tn):12.3f}'
               f'{t[f"mcc_{nm}"].mean():10.3f}{t[f"mcc_{nm}"].median():12.3f}')


def paired(x, y, label):
    dd = (t[f'mcc_{x}'] - t[f'mcc_{y}']).values
    lo95, lo90, hi90, hi95 = boot(dd)
    out.append(f'\n{label}: mean diff {dd.mean():+.3f}, median {np.median(dd):+.3f}, {x} higher in {(dd > 0).sum()}/{len(dd)}, '
               f'Wilcoxon p = {wilcoxon(dd).pvalue:.3g}')
    out.append(f'   bootstrap over targets: 95 % CI [{lo95:+.3f}, {hi95:+.3f}], 90 % CI [{lo90:+.3f}, {hi90:+.3f}]')
    return dd, lo90, hi90


dd, lo90, _ = paired('ErG', 'ECFP4', 'ErG - ECFP4')
out.append(f'   non-inferiority (one-sided 5 %, margin delta = {a.delta:.2f} MCC): lower 90 % bound {lo90:+.3f} '
           f'> {-a.delta:+.2f}?  {"YES (non-inferior)" if lo90 > -a.delta else "NO"}; '
           f'smallest margin supported by the 90 % bound: {-lo90:.3f}')
best = np.where(t.mcc_ErG.values >= t.mcc_ECFP4.values, t.mcc_ErG, t.mcc_ECFP4)
t['_best'] = best
dd2 = (t.mcc_FUSION - t._best).values
out.append(f'\nFUSION - best single scorer per target (optimistic oracle): mean {dd2.mean():+.3f}, FUSION higher in {(dd2 > 0).sum()}/{len(dd2)}')
paired('FUSION', 'ECFP4', 'FUSION - ECFP4')
paired('FUSION', 'ErG', 'FUSION - ErG')
dd3, lo90b, _ = paired('ErG_RDKit', 'ECFP4', 'ErG_RDKit - ECFP4')
out.append(f'   non-inferiority (margin {a.delta:.2f}): lower 90 % bound {lo90b:+.3f}  {"YES" if lo90b > -a.delta else "NO"}')
paired('ErG', 'ErG_RDKit', 'ErG v2 - ErG_RDKit')
paired('FUSION_RDKit', 'ECFP4', 'FUSION_RDKit - ECFP4')
out.append(f'\nSpearman rho between ErG v2 and RDKit ErG similarity within a target: median {t.rho_erg_rdkit.median():.3f}, '
           f'range [{t.rho_erg_rdkit.min():.3f}, {t.rho_erg_rdkit.max():.3f}]')

b_, e_, c_ = int(t.tp_both.sum()), int(t.tp_erg_only.sum()), int(t.tp_ecfp_only.sum())
tot = int(t.n_act.sum())
out.append(f'\ncomplementarity (true positives among {tot} actives): found by both {b_} ({b_ / tot:.1%}), '
           f'ErG only {e_} ({e_ / tot:.1%}), ECFP4 only {c_} ({c_ / tot:.1%}); '
           f'mean Jaccard of the two predicted-active sets {t.jaccard_pred.mean():.3f}')
text = '\n'.join(out)
open(os.path.join(R.HERE, 'decoy_fusion_summary.txt'), 'w').write(text)
print(text)
