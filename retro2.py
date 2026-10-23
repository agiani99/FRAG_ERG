"""
Retrospective test 2: scaffold-hop split, ErG nearest-neighbour baseline, MCC / confusion matrices.

Same targets / references as retro.py (seed 0, ChEMBL, most potent compound = reference), but

 held-out sets   A_all : all other actives (pChEMBL >= 7)
                 A_hop : actives with ECFP4 < 0.3 to the reference (scaffold hops; needs >= 20)
 conditions      G guided        reference ErG as target
                 X wrong target  ErG of a random ChEMBL molecule (same anchor / constraints / filters)
                 R random        random choices (same anchor / constraints / filters)
                 N ErG-NN        top 10 by ErG Tanimoto to the reference from a pool of ~100k ChEMBL molecules
                                 (22-40 heavy atoms), excluding every compound measured on the target
 generated-molecule metrics (top 10 per condition)
                 maxsim  mean over molecules of the max ECFP4 Tanimoto to the held-out set
                 hit     fraction of molecules above a label-free cutoff: the 90th percentile of the same
                         quantity for 200 random size-matched ChEMBL molecules (expected 10 % for a random molecule)
                 EF      hit / 0.10
 classification diagnostic: the target's measured compounds (actives vs pChEMBL <= 5) ranked by
                 ErG-sim-to-reference, ECFP4-sim-to-reference, heavy atoms (size-only), -|dHeavyAtoms| (size-match).
                 Decision rules fixed in advance: predict the top k = n_actives as active, and the top 25 %.
                 Confusion matrices and MCC (per target and pooled); AUC only as a side column.

usage: python retro2.py [n_targets=40] [workers=14] [--seed 0] [--pool 100000]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, argparse, time, multiprocessing as mp
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

import retro as R
import frag_erg as F
import erg_v2 as E
import grow as G

RDLogger.DisableLog('rdApp.*')
HERE = R.HERE
HOP = 0.3     # 0.4 left 89-97 % of actives in the 'hop' set (smoke test), i.e. barely restrictive
_W = {}


def mcc_from(tp, fp_, fn, tn):
    den = np.sqrt(float(tp + fp_) * (tp + fn) * (tn + fp_) * (tn + fn))
    return 0.0 if den == 0 else (tp * tn - fp_ * fn) / den


def topk_confusion(score, y, k, rng):
    order = np.argsort(-(score + rng.random(len(score)) * 1e-9))
    pred = np.zeros(len(y), dtype=bool)
    pred[order[:k]] = True
    tp = int((pred & (y == 1)).sum()); fp_ = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return tp, fp_, fn, tn


def _neut_row(t):
    m = R.neutral(t[1])
    if m is None or not (22 <= m.GetNumHeavyAtoms() <= 40):
        return None
    return t[0], Chem.MolToSmiles(m)


def build_pool(d, n, seed, workers):
    u = d.drop_duplicates('molecule_chembl_id').sample(frac=1.0, random_state=seed).head(int(n * 1.6))
    with mp.Pool(workers) as p:
        rows = [r for r in p.map(_neut_row, list(zip(u.molecule_chembl_id, u.SMILES)), chunksize=2000) if r]
    rows = rows[:n]
    ids = np.array([r[0] for r in rows])
    smis = [r[1] for r in rows]
    raw, ok = E.raw_many(smis, jobs=workers, progress=False)
    return ids[ok], [s for s, o in zip(smis, ok) if o], raw[ok]


def init2(lib_args, args, pool):
    R.init(lib_args, args)
    _W['pool'] = pool


def one_target2(job):
    tid, name, ref_smi, act_smis, weak_smis, pool_smis, seed, tmols, meas = job
    t0 = time.time()
    a, lib = R._W['a'], R._W['lib']
    p_ids, p_smis, p_raw = _W['pool']
    rng = np.random.default_rng(seed)
    ref = Chem.MolFromSmiles(ref_smi)
    f_ref = R.fp(ref)
    tg = G.Target(E.erg_raw(ref).astype(np.int64))
    A = [m for m in (R.neutral(s) for s in act_smis) if m is not None]
    W = [m for m in (R.neutral(s) for s in weak_smis) if m is not None]
    fA, fW = [R.fp(m) for m in A], [R.fp(m) for m in W]
    simref = np.array(DataStructs.BulkTanimotoSimilarity(f_ref, fA))
    fH = [f for f, s in zip(fA, simref) if s < HOP]
    row = dict(target=tid, name=name, ref=ref_smi, n_act=len(A), n_hop=len(fH), n_weak=len(W))
    bg = [R.fp(m) for m in (Chem.MolFromSmiles(s) for s in rng.choice(pool_smis, 200, replace=False))]
    cut_all = float(np.percentile(R.max_sim(bg, fA), 90))
    cut_hop = float(np.percentile(R.max_sim(bg, fH), 90)) if len(fH) >= 20 else np.nan
    row.update(cut_all=cut_all, cut_hop=cut_hop)
    mols = []

    def score(cond, smiles):
        if len(smiles) < 3:
            row[f'{cond}_n'] = len(smiles)
            return
        fps = [R.fp(Chem.MolFromSmiles(s)) for s in smiles]
        ma = R.max_sim(fps, fA)
        row[f'{cond}_n'] = len(smiles)
        row[f'{cond}_all_maxsim'] = float(ma.mean())
        row[f'{cond}_all_hit'] = float((ma >= cut_all).mean())
        if len(fH) >= 20:
            mh = R.max_sim(fps, fH)
            row[f'{cond}_hop_maxsim'] = float(mh.mean())
            row[f'{cond}_hop_hit'] = float((mh >= cut_hop).mean())
        row[f'{cond}_weak_maxsim'] = float(R.max_sim(fps, fW).mean()) if fW else np.nan
        row[f'{cond}_sim_ref'] = float(np.mean(DataStructs.BulkTanimotoSimilarity(f_ref, fps)))

    dfG, infoG = G.run_grow(a, lib, ref_smi, verbose=False)
    row['G_ref_found'] = infoG['stat']['ref_found']
    row['anchor'] = ' '.join(infoG['anchor_keys'])
    dfX, _ = G.run_grow(a, lib, ref_smi, target_smiles=pool_smis[int(rng.integers(len(pool_smis)))], verbose=False)
    dfR, _ = G.run_grow(a, lib, ref_smi, rng=np.random.default_rng(seed + 1), verbose=False)
    for cond, df, pick in (('G', dfG, 'top'), ('X', dfX, 'top'), ('R', dfR, 'rnd')):
        if df is None or len(df) < 3:
            row[f'{cond}_n'] = 0 if df is None else len(df)
            continue
        top = df.head(10) if pick == 'top' else df.sample(min(10, len(df)), random_state=int(seed))
        score(cond, top.smiles.tolist())
        for r_, s in enumerate(top.smiles):
            mols.append(dict(target=tid, cond=cond, rank=r_, smiles=s))
    # N: ErG nearest neighbours from the pool, target's own compounds excluded
    sc = tg.tanimoto(p_raw.astype(np.int64))
    sc[np.isin(p_ids, list(tmols))] = -1
    nn = np.argsort(-sc)[:10]
    score('N', [p_smis[i] for i in nn])
    for r_, i in enumerate(nn):
        mols.append(dict(target=tid, cond='N', rank=r_, smiles=p_smis[i]))

    # classification diagnostic on the measured actives / weak compounds
    ms, y = [], []
    for s, pv, mid in meas:
        m = R.neutral(s)
        if m is not None and m.GetNumHeavyAtoms() >= 10 and Chem.MolToSmiles(m) != ref_smi:
            ms.append(m); y.append(int(pv >= 7))
    y = np.array(y)
    cls = []
    if y.sum() >= 10 and (1 - y).sum() >= 10:
        V = np.array([E.erg_raw(m) for m in ms]).astype(np.int64)
        ha = np.array([m.GetNumHeavyAtoms() for m in ms], dtype=float)
        scorers = {'ErG_sim': tg.tanimoto(V),
                   'ECFP4_sim': np.array(DataStructs.BulkTanimotoSimilarity(f_ref, [R.fp(m) for m in ms])),
                   'size_only': ha, 'size_match': -np.abs(ha - ref.GetNumHeavyAtoms())}
        from sklearn.metrics import roc_auc_score
        for nm, scv in scorers.items():
            for rule, k in (('k=n_act', int(y.sum())), ('top25%', int(np.ceil(0.25 * len(y))))):
                tp, fp_, fn, tn = topk_confusion(scv, y, k, rng)
                cls.append(dict(target=tid, scorer=nm, rule=rule, tp=tp, fp=fp_, fn=fn, tn=tn,
                                mcc=mcc_from(tp, fp_, fn, tn), auc=float(roc_auc_score(y, scv)),
                                n=len(y), n_act=int(y.sum())))
    row['seconds'] = time.time() - t0
    return row, mols, cls


def summarize(res, cls, wilcoxon):
    out = []
    for c in 'GXRN':
        for suf in ('n', 'all_maxsim', 'all_hit', 'hop_maxsim', 'hop_hit'):
            if f'{c}_{suf}' not in res:
                res[f'{c}_{suf}'] = np.nan
    out.append(f'targets: {len(res)}; with G, X, R and N all producing >= 3 molecules: '
               f'{int(res[["G_n", "X_n", "R_n", "N_n"]].ge(3).all(1).sum())}')
    for tag, hopname in (('all', 'all held-out actives'), ('hop', f'scaffold-hop actives only (ECFP4 < {HOP} to reference)')):
        cols = [f'{c}_{tag}_maxsim' for c in 'GXRN'] + [f'{c}_{tag}_hit' for c in 'GXRN']
        o = res.dropna(subset=cols)
        out.append(f'\n=== {hopname}: n = {len(o)} targets ===')
        out.append(f'{"":12s}{"G guided":>10s}{"X wrong":>10s}{"R random":>10s}{"N ErG-NN":>10s}')
        out.append(f'{"maxsim":12s}' + ''.join(f'{o[f"{c}_{tag}_maxsim"].median():10.3f}' for c in 'GXRN') + '   (median)')
        out.append(f'{"hit rate":12s}' + ''.join(f'{o[f"{c}_{tag}_hit"].mean():10.3f}' for c in 'GXRN') +
                   '   (mean; 0.10 = random molecule)')
        out.append(f'{"EF":12s}' + ''.join(f'{o[f"{c}_{tag}_hit"].mean() / 0.10:10.2f}' for c in 'GXRN'))
        for c, nm in (('X', 'wrong-target'), ('R', 'random'), ('N', 'ErG-NN')):
            for met in ('maxsim', 'hit'):
                d = o[f'G_{tag}_{met}'] - o[f'{c}_{tag}_{met}']
                p = wilcoxon(d).pvalue if (d != 0).any() else float('nan')
                out.append(f'   {met:7s} guided - {nm:12s}: mean diff {d.mean():+.3f}, median {d.median():+.3f}, '
                           f'guided higher in {(d > 0).sum()}/{len(d)}, Wilcoxon p = {p:.3g}')
    out.append('\n=== classification of measured compounds (active = pChEMBL >= 7, weak = <= 5) ===')
    for rule in ('k=n_act', 'top25%'):
        out.append(f'\nrule {rule}: predict the top-scored compounds as active')
        out.append(f'{"scorer":12s}{"TP":>8s}{"FP":>8s}{"FN":>8s}{"TN":>8s}{"pooled MCC":>12s}{"median MCC":>12s}'
                   f'{"MCC>0":>8s}{"median AUC":>12s}')
        for nm in ('ErG_sim', 'ECFP4_sim', 'size_only', 'size_match'):
            c = cls[(cls.scorer == nm) & (cls.rule == rule)]
            tp, fp_, fn, tn = (int(c[k].sum()) for k in ('tp', 'fp', 'fn', 'tn'))
            out.append(f'{nm:12s}{tp:8d}{fp_:8d}{fn:8d}{tn:8d}{mcc_from(tp, fp_, fn, tn):12.3f}'
                       f'{c.mcc.median():12.3f}{int((c.mcc > 0).sum()):5d}/{len(c):<3d}{c.auc.median():12.3f}')
        e = cls[(cls.scorer == 'ErG_sim') & (cls.rule == rule)].set_index('target')
        for other in ('ECFP4_sim', 'size_only', 'size_match'):
            c = cls[(cls.scorer == other) & (cls.rule == rule)].set_index('target')
            d = (e.mcc - c.mcc).dropna()
            p = wilcoxon(d).pvalue if (d != 0).any() else float('nan')
            out.append(f'   MCC ErG_sim - {other:10s}: median diff {d.median():+.3f}, ErG higher in '
                       f'{(d > 0).sum()}/{len(d)}, Wilcoxon p = {p:.3g}')
    out.append(f'\nreference rediscovered (guided): {int(res.G_ref_found.fillna(False).sum())}/{len(res)}')
    return '\n'.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('n_targets', nargs='?', type=int, default=40)
    ap.add_argument('workers', nargs='?', type=int, default=14)
    ap.add_argument('--min-act', type=int, default=80)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pool', type=int, default=100000)
    ap.add_argument('--beam', type=int, default=40)
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--anchor-n', type=int, default=2)
    ap.add_argument('--close-top', type=int, default=60)
    ap.add_argument('--tag', default='retro2')
    r = ap.parse_args()
    from scipy.stats import wilcoxon

    a = G.build_parser().parse_args(['--smiles', 'x', '--beam', str(r.beam), '--steps', str(r.steps),
                                     '--anchor-n', str(r.anchor_n), '--close-top', str(r.close_top)])
    lib_args = (a.library, a.min_count, a.max_block_heavy, a.max_block_attach, 1, a.allow_enamine, a.allow_motifs)
    G.load_library(a.library, a.min_count, a.max_block_heavy, a.max_block_attach, r.workers,
                   a.allow_enamine, a.allow_motifs)
    d = pd.read_parquet(R.CHEMBL, columns=['chembl_target_id', 'chembl_target_name', 'molecule_chembl_id',
                                           'SMILES', 'pchembl_value']).dropna(subset=['pchembl_value'])
    jobs0 = R.pick_targets(d, r.n_targets, r.min_act, r.seed)
    jobs = []
    for (tid, name, ref_smi, acts, weak, pool_smis, seed) in jobs0:
        sub = d[d.chembl_target_id == tid].drop_duplicates('molecule_chembl_id')
        meas = sub[(sub.pchembl_value >= 7) | (sub.pchembl_value <= 5)]
        if len(meas) > 1500:
            meas = meas.sample(1500, random_state=0)
        jobs.append((tid, name, ref_smi, acts, weak, pool_smis, seed, set(sub.molecule_chembl_id),
                     list(zip(meas.SMILES, meas.pchembl_value, meas.molecule_chembl_id))))
    print(f'{len(jobs)} targets; building the ErG-NN pool ({r.pool} ChEMBL molecules) ...', flush=True)
    pool = build_pool(d, r.pool, r.seed, r.workers)
    print(f'pool: {len(pool[0])} molecules; beam {r.beam}, steps {r.steps}, anchor-n {r.anchor_n}', flush=True)
    rows, mols, cls, t0 = [], [], [], time.time()
    with mp.Pool(r.workers, initializer=init2, initargs=(lib_args, a, pool)) as p:
        for k, (row, m, c) in enumerate(p.imap_unordered(one_target2, jobs), 1):
            rows.append(row); mols += m; cls += c
            pd.DataFrame(rows).to_csv(os.path.join(HERE, f'{r.tag}_results.csv'), index=False)
            print(f'[{k}/{len(jobs)}] {row["target"]} {row["name"][:38]}: hop {row["n_hop"]}/{row["n_act"]} actives'
                  f' ({row["seconds"]:.0f}s, {time.time() - t0:.0f}s total)', flush=True)
    res, cls = pd.DataFrame(rows), pd.DataFrame(cls)
    pd.DataFrame(mols).to_csv(os.path.join(HERE, f'{r.tag}_molecules.csv'), index=False)
    cls.to_csv(os.path.join(HERE, f'{r.tag}_classification.csv'), index=False)
    text = summarize(res, cls, wilcoxon)
    open(os.path.join(HERE, f'{r.tag}_summary.txt'), 'w').write(text)
    print('\n' + text)
