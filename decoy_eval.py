"""
Balanced classification test with size-matched decoys (fixes the 84 % actives problem of retro2).

For each target of retro2_results.csv: positives = up to 300 other actives (pChEMBL >= 7, reference excluded),
negatives = the same number of ChEMBL compounds that have no target annotation at all
(CHEMBL37_notarget.parquet), each matched to one active by heavy-atom count, so that a size-only score is at
chance by construction. Scorers (similarity to the reference): ErG v2, ECFP4, heavy atoms only.
Decision rule fixed in advance: the top k = n_actives of each set are predicted active (prevalence 50 %).
Reports pooled confusion matrix + MCC, per-target median MCC, and AUC only as a side column.

usage: python decoy_eval.py [--per-target 300] [--seed 0]
"""
import paths
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

import retro as R
import erg_v2 as E
import grow as G
import retro2 as R2

RDLogger.DisableLog('rdApp.*')
DECOY = paths.data('CHEMBL37_notarget.parquet')

ap = argparse.ArgumentParser()
ap.add_argument('--per-target', type=int, default=300)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--decoys', default=DECOY)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

res = pd.read_csv(os.path.join(R.HERE, 'retro2_results.csv'))
dec = pd.read_parquet(a.decoys, columns=['SMILES']).SMILES.drop_duplicates().tolist()
dm = [R.neutral(s) for s in dec]
dec = [(Chem.MolToSmiles(m), m.GetNumHeavyAtoms()) for m in dm if m is not None and 12 <= m.GetNumHeavyAtoms() <= 60]
dec_ha = np.array([h for _, h in dec])
print(f'{len(dec)} decoys (neutralised, 12-60 heavy atoms)', flush=True)

d = pd.read_parquet(R.CHEMBL, columns=['chembl_target_id', 'molecule_chembl_id', 'SMILES',
                                       'pchembl_value']).dropna(subset=['pchembl_value'])
rows, cls = [], []
for _, r in res.iterrows():
    ref = Chem.MolFromSmiles(r.ref)
    sub = d[(d.chembl_target_id == r.target) & (d.pchembl_value >= 7)].drop_duplicates('molecule_chembl_id')
    sub = sub.sample(min(a.per_target, len(sub)), random_state=a.seed)
    pos = [m for m in (R.neutral(s) for s in sub.SMILES) if m is not None and Chem.MolToSmiles(m) != r.ref
           and 12 <= m.GetNumHeavyAtoms() <= 60]
    if len(pos) < 20:
        continue
    # one size-matched decoy per active (without replacement)
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
    ha = np.array([m.GetNumHeavyAtoms() for m in ms], dtype=float)
    scorers = {'ErG_sim': tg.tanimoto(V),
               'ECFP4_sim': np.array(DataStructs.BulkTanimotoSimilarity(R.fp(ref), [R.fp(m) for m in ms])),
               'size_only': ha}
    for nm, sc in scorers.items():
        tp, fp_, fn, tn = R2.topk_confusion(sc, y, int(y.sum()), rng)
        cls.append(dict(target=r.target, scorer=nm, tp=tp, fp=fp_, fn=fn, tn=tn,
                        mcc=R2.mcc_from(tp, fp_, fn, tn), auc=float(roc_auc_score(y, sc)), n_act=int(y.sum())))
    print(f'{r.target} {r["name"][:36]:36s} n={len(pos)}+{len(neg)}  '
          + '  '.join(f'{c["scorer"]} MCC {c["mcc"]:+.2f}' for c in cls[-3:]), flush=True)

cls = pd.DataFrame(cls)
cls.to_csv(os.path.join(R.HERE, 'decoy_eval_classification.csv'), index=False)
out = [f'\nbalanced sets (50 % actives vs size-matched no-target decoys), {cls.target.nunique()} targets, '
       f'rule: top k = n_actives predicted active', f'{"scorer":12s}{"TP":>8s}{"FP":>8s}{"FN":>8s}{"TN":>8s}'
       f'{"pooled MCC":>12s}{"median MCC":>12s}{"MCC>0":>9s}{"median AUC":>12s}']
for nm in ('ErG_sim', 'ECFP4_sim', 'size_only'):
    c = cls[cls.scorer == nm]
    tp, fp_, fn, tn = (int(c[k].sum()) for k in ('tp', 'fp', 'fn', 'tn'))
    out.append(f'{nm:12s}{tp:8d}{fp_:8d}{fn:8d}{tn:8d}{R2.mcc_from(tp, fp_, fn, tn):12.3f}{c.mcc.median():12.3f}'
               f'{int((c.mcc > 0).sum()):5d}/{len(c):<3d}{c.auc.median():12.3f}')
e = cls[cls.scorer == 'ErG_sim'].set_index('target')
for o in ('ECFP4_sim', 'size_only'):
    c = cls[cls.scorer == o].set_index('target')
    dd = (e.mcc - c.mcc).dropna()
    out.append(f'   MCC ErG_sim - {o:10s}: median diff {dd.median():+.3f}, ErG higher in {(dd > 0).sum()}/{len(dd)}, '
               f'Wilcoxon p = {wilcoxon(dd).pvalue:.3g}')
text = '\n'.join(out)
open(os.path.join(R.HERE, 'decoy_eval_summary.txt'), 'w').write(text)
print(text)
