"""
Anchor-based de novo grow/score loop in ErG space.

  target   = ErG v2 (fuzzy) of a reference ligand
  anchor   = the reference's most pharmacophore-influential BRICS block(s), kept verbatim
  library  = unique BRICS blocks (fragment_counts_pubchem_sample.parquet from compress.py)
  grow     = beam search; a block is bonded to an open attachment (BRICS label rules)
  score    = continuous Tanimoto between fuzzy(ErG of the partial molecule) and the target.
             The ErG of a candidate is the exact incremental update  v + counts + internal + cross,
             the cross term being one small convolution of the open attachment's reach profile with
             the block's attachment profile. No molecule is ever built during scoring.
  close    = every open attachment is finally capped with the best mono-valent block, the molecule is
             built, sanitized and its ErG recomputed from scratch (erg_v2.erg_raw) to verify the loop.

usage: python grow.py --smiles "<reference ligand>" [--min-count 3] [--beam 40] [--steps 6] ...
"""
import paths
import argparse, os, sys, time, pickle, heapq
import multiprocessing as mp
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, BRICS, rdmolops

import frag_erg as F
import erg_v2 as E
import filters as FL

RDLogger.DisableLog('rdApp.*')
NB, PW = E.NBINS, 16                      # distance bins, profile width (index 15 = ">=15")
HERE = os.path.dirname(os.path.abspath(__file__))


def _compat():
    c = {}
    for grp in BRICS.reactionDefs:
        for a, b, _ in grp:
            a, b = int(a.rstrip('ab')), int(b.rstrip('ab'))
            c.setdefault(a, set()).add(b)
            c.setdefault(b, set()).add(a)
    return c


COMPAT = _compat()

# T[a,b,k] = 1 if a cross pair with profile distances a,b falls in distance bin k:  min(a+b+1, 15)-1
_T = np.zeros((PW, PW, NB), dtype=np.float32)
for _a in range(PW):
    for _b in range(PW):
        _T[_a, _b, min(_a + _b + 1, NB) - 1] = 1.0


# ------------------------------------------------------------------ scoring ----
def fuzz(V):
    V = np.atleast_2d(V).astype(np.float32)
    out = V.copy()
    out[:, 6:] = (V[:, 6:].reshape(-1, 21, NB) @ E.F).reshape(len(V), -1)
    return out


class Target:
    def __init__(self, raw, mode='tanimoto', overshoot_weight=1.0):
        self.raw = np.asarray(raw, dtype=np.float32)
        self.t = fuzz(raw)[0]
        self.n2 = float(self.t @ self.t)
        self.mode, self.ow = mode, overshoot_weight

    def contain(self, V):
        """Containment of raw count vectors V (n, 321) in the target's raw vector. ErG counts only grow while a molecule
        is built and the distances inside a connected piece never change, so the vector of a sub-structure of the target
        is bounded by the target's, bin by bin. coverage = sum(min(V, T)) / sum(T)  (how much of the target is reached),
        overshoot = sum(max(V - T, 0)) / sum(V)  (mass that cannot belong to the target)."""
        V = np.atleast_2d(V).astype(np.float32)
        cov = np.minimum(V, self.raw[None, :]).sum(1) / max(float(self.raw.sum()), 1.0)
        ov = np.maximum(V - self.raw[None, :], 0).sum(1) / np.maximum(V.sum(1), 1.0)
        return cov, ov

    def rank_score(self, V):
        """The score used to order candidates / beam states: Tanimoto of the fuzzy vectors (default) or
        coverage - overshoot_weight * overshoot ('contain')."""
        if self.mode == 'contain':
            cov, ov = self.contain(V)
            return cov - self.ow * ov
        return self.tanimoto(V)

    def tanimoto(self, V, chunk=40000):
        out = np.empty(len(V), dtype=np.float32)
        for s in range(0, len(V), chunk):
            f = fuzz(V[s:s + chunk])
            dot = f @ self.t
            out[s:s + chunk] = dot / ((f * f).sum(1) + self.n2 - dot)
        return out


def erg_from_sites(mol, sites):
    """Same as E.erg_raw, but for a given site list (used for leave-one-fragment-out influence)."""
    S = len(sites)
    ftype = np.array([f for f, _ in sites], dtype=np.int64)
    counts = np.bincount(ftype, minlength=6)
    hist = np.zeros(21 * NB, dtype=np.int64)
    if S > 1:
        dmb = np.array(rdmolops.GetDistanceMatrix(mol))
        np.fill_diagonal(dmb, E._BIG)
        n = len(dmb)
        P = np.zeros((S, n), dtype=bool)
        for k, (_, A) in enumerate(sites):
            P[k, A] = True
        M = np.where(P[:, :, None], dmb[None, :, :], E._BIG).min(1)
        D = np.where(P[None, :, :], M[:, None, :], E._BIG).min(2)
        i, j = E._triu(S)
        d = D[i, j]
        keep = d < 1e6
        d = np.clip(d[keep], 1, NB).astype(np.int64)
        pt = E._PT_LUT[ftype[i[keep]], ftype[j[keep]]]
        hist = np.bincount(pt * NB + d - 1, minlength=21 * NB)
    return np.concatenate([counts, hist])


# ------------------------------------------------------------------ library ----
def _lib_worker(args):
    batch, allow_enamine, allow_motifs = args
    out = []
    for kid, key, cnt in batch:
        m = Chem.MolFromSmiles(key)
        if m is None:
            continue
        dums = [a for a in m.GetAtoms() if a.GetAtomicNum() == 0]
        if not dums:
            continue
        # label-7 (C=C / C=N) cuts are only re-joinable if the dummy bond is double (libraries built
        # before the cut bond order was kept store it as single: skip those blocks)
        if any(a.GetIsotope() // 10 == 7 and
               m.GetBondBetweenAtoms(a.GetIdx(), a.GetNeighbors()[0].GetIdx()).GetBondType() != Chem.BondType.DOUBLE
               for a in dums):
            continue
        heavy = m.GetNumAtoms() - len(dums)
        ok, nring = FL.block_ok(m, allow_enamine, allow_motifs)
        if not ok:
            continue
        try:
            t = F.fragment_table(m)
        except Exception:
            continue
        for a in dums:
            d = a.GetIdx()
            bt = m.GetBondBetweenAtoms(d, t['attach'][d]).GetBondType()
            p = t['prof'][d]
            p16 = p[:, :PW].copy()
            p16[:, PW - 1] += p[:, PW:].sum(1)
            out.append((kid, d, a.GetIsotope() // 10, a.GetIsotope() % 10, int(t['attach_c'][d]),
                        int(t['hyok'][d]), 2 if bt == Chem.BondType.DOUBLE else 1, len(dums), heavy,
                        cnt, t['counts'], t['internal'], p16, nring))
    return out


def load_library(path, min_count, max_heavy, max_dummies, workers, allow_enamine=False, allow_motifs=False):
    tag = os.path.splitext(os.path.basename(path))[0].replace('fragment_counts_', '')
    cache = os.path.join(HERE, f'lib_cache_{tag}_c{min_count}_h{max_heavy}_d{max_dummies}{"_enam" if allow_enamine else ""}{"_motif" if allow_motifs else ""}_f7.pkl')
    if os.path.exists(cache) and os.path.getmtime(cache) > max(os.path.getmtime(path),
                                                               os.path.getmtime(__file__)):
        with open(cache, 'rb') as fh:
            return pickle.load(fh)
    df = pd.read_parquet(path)
    df = df[df['count'] >= min_count]
    keys = df.index.tolist()
    cnts = df['count'].tolist()
    items = []
    for kid, (k, c) in enumerate(zip(keys, cnts)):
        if '*' not in k:
            continue
        ndum = k.count('*')
        if ndum > max_dummies:
            continue
        nat = Chem.MolFromSmiles(k)
        if nat is None or nat.GetNumAtoms() - ndum > max_heavy:
            continue
        items.append((kid, k, c))
    print(f'building library: {len(items)} blocks (count>={min_count}, <= {max_heavy} heavy, '
          f'<= {max_dummies} attachments)', flush=True)
    batches = [(items[i:i + 500], allow_enamine, allow_motifs) for i in range(0, len(items), 500)]
    rows = []
    with mp.Pool(workers) as pool:
        for r in pool.imap(_lib_worker, batches):
            rows += r
    lib = dict(keys=keys,
               kid=np.array([r[0] for r in rows]), dum=np.array([r[1] for r in rows]),
               lab=np.array([r[2] for r in rows]), dC=np.array([r[3] for r in rows]),
               ac=np.array([r[4] for r in rows]), hy=np.array([r[5] for r in rows]),
               bt=np.array([r[6] for r in rows]), ndum=np.array([r[7] for r in rows]),
               heavy=np.array([r[8] for r in rows]), cnt=np.array([r[9] for r in rows]),
               counts=np.array([r[10] for r in rows], dtype=np.int16),
               internal=np.array([r[11] for r in rows], dtype=np.int16),
               prof=np.array([r[12] for r in rows], dtype=np.float32),
               nring=np.array([r[13] for r in rows]))
    with open(cache, 'wb') as fh:
        pickle.dump(lib, fh)
    print(f'library: {len(rows)} attachment entries from {len(set(lib["kid"].tolist()))} blocks', flush=True)
    return lib


# -------------------------------------------------------------------- state ----
class State:
    def __init__(self, inst, links):
        self.inst, self.links = inst, links
        self.linked = {}
        for (f, d) in links:
            self.linked.setdefault(f, []).append(d)
        self.vec = F.assemble([i['table'] for i in inst], links)
        self.heavy = sum(i['heavy'] for i in inst)
        self.nring = sum(i['nring'] for i in inst)
        self.sig = (tuple(sorted(i['key'] for i in inst)), self.vec.tobytes())
        self.pose, self.pocket, self.known = None, None, None

    def open_ends(self):
        return [(f, d) for f, i in enumerate(self.inst) for d in i['table']['prof'] if (f, d) not in self.links]

    def end_info(self, f, d):
        i = self.inst[f]
        a = i['mol'].GetAtomWithIdx(d)
        at = i['table']['attach'][d]
        bt = i['mol'].GetBondBetweenAtoms(d, at).GetBondType()
        return (a.GetIsotope() // 10, a.GetIsotope() % 10, int(i['table']['attach_c'][d]),
                bool(i['table']['hyok'][d]), 2 if bt == Chem.BondType.DOUBLE else 1)

    def reach(self, g, y):
        t = self.inst[g]['table']
        H = t['prof'][y].copy()
        for z in self.linked.get(g, []):
            if z == y:
                continue
            h, w = self.links[(g, z)]
            H += F._shift(self.reach(h, w), t['dd'][(y, z)] + 1)
        return H

    def add(self, x, inst):
        """New State with `inst` (a library block instance, joined by its dummy inst['y']) bonded to open end x."""
        g = len(self.inst)
        links = dict(self.links)
        links[x] = (g, inst['y'])
        links[(g, inst['y'])] = x
        return State(self.inst + [inst], links)

    def assemble(self, partial=False):
        """(mol, keys): the molecule built from the instances. partial=True leaves open attachments as implicit H.
        keys[i] = (instance, local atom index) of heavy atom i of mol. (None, None) if it cannot be built."""
        if self.open_ends() and not partial:
            return None, None
        combo, offs = None, []
        for i in self.inst:
            offs.append(0 if combo is None else combo.GetNumAtoms())
            combo = Chem.Mol(i['mol']) if combo is None else Chem.CombineMols(combo, i['mol'])
        _drop_cached(combo)
        rw = Chem.RWMol(combo)
        kill = set()
        for (f, d), (g, e) in self.links.items():
            if (f, d) > (g, e):
                continue
            a = offs[f] + self.inst[f]['table']['attach'][d]
            b = offs[g] + self.inst[g]['table']['attach'][e]
            bt = self.inst[f]['mol'].GetBondBetweenAtoms(d, self.inst[f]['table']['attach'][d]).GetBondType()
            rw.AddBond(a, b, bt)
            kill |= {offs[f] + d, offs[g] + e}
        if partial:
            kill |= {offs[f] + d for f, d in self.open_ends()}
        amap = [(offs[f] + loc, ref_idx) for f, i in enumerate(self.inst)
                for loc, ref_idx in i.get('ref_atoms', {}).items()]
        keys = [(f, loc) for f, i in enumerate(self.inst) for loc in range(i['mol'].GetNumAtoms())
                if offs[f] + loc not in kill]
        for k in sorted(kill, reverse=True):
            rw.RemoveAtom(k)
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None, None
        ks = sorted(kill)
        mol.SetProp('anchor_map', ';'.join(f'{i - sum(1 for k in ks if k < i)}:{j}' for i, j in amap))
        return mol, keys

    def build(self):
        return self.assemble()[0]


def make_inst(lib, e):
    key = lib['keys'][lib['kid'][e]]
    mol = Chem.MolFromSmiles(key)
    t = F.fragment_table(mol)
    return dict(mol=mol, table=t, key=key, y=int(lib['dum'][e]), heavy=int(lib['heavy'][e]),
                nring=int(lib['nring'][e]))


class Ctx:
    """Everything the scoring needs besides the state."""
    def __init__(self, lib, tgt, cons, max_heavy, max_rings, deficit_weight, rng=None):
        self.lib, self.tgt, self.cons = lib, tgt, cons
        self.max_heavy, self.max_rings, self.dw, self.rng = max_heavy, max_rings, deficit_weight, rng


def expand(cx, st, x, caps_only=False):
    """Vectorised scoring of every compatible library entry bonded at open end x of state st.
    Entries that break an upper constraint (heavy atoms, rings, pocket upper bounds) are dropped.
    Returns (entry_idx, new_vec[n,321], tanimoto[n], rank[n]) with rank = tanimoto - w*lower-bound deficit."""
    lib, tgt = cx.lib, cx.tgt
    lab, dC, ac, hy, bt = st.end_info(*x)
    m = np.isin(lib['lab'], list(COMPAT.get(lab, ()))) & (lib['dC'] == ac) & (lib['ac'] == dC) & (lib['bt'] == bt)
    if hy:
        m &= lib['hy'] == 0
    m &= (lib['heavy'] + st.heavy) <= cx.max_heavy
    m &= (lib['nring'] + st.nring) <= cx.max_rings
    if caps_only:
        m &= lib['ndum'] == 1
    idx = np.nonzero(m)[0]
    if not len(idx):
        return idx, None, None, None
    P = st.reach(*x)
    P16 = P[:, :PW].astype(np.float32).copy()
    P16[:, PW - 1] += P[:, PW:].sum(1)
    Pw = np.einsum('sa,abk->sbk', P16, _T)                       # (6, PW, NB)
    Q = lib['prof'][idx]                                          # (n, 6, PW)
    cross = np.einsum('nub,sbk->nusk', Q, Pw)                     # (n, 6(u new), 6(s old), NB)
    delta = np.zeros((len(idx), 321), dtype=np.float32)
    delta[:, :6] = lib['counts'][idx]
    delta[:, 6:] = lib['internal'][idx]
    for s in range(6):
        for u in range(6):
            p = E._PT_LUT[s, u]
            delta[:, 6 + p * NB: 6 + (p + 1) * NB] += cross[:, u, s, :]
    V = st.vec[None, :].astype(np.float32) + delta
    viol, deficit, _ = cx.cons.check(V)
    keep = ~viol
    if not keep.any():
        return idx[:0], None, None, None
    idx, V, deficit = idx[keep], V[keep], deficit[keep]
    tan = tgt.tanimoto(V)
    if cx.rng is not None:                       # random-guidance control: scores are ignored for ranking
        return idx, V, tan, cx.rng.random(len(idx))
    base = tgt.rank_score(V) if tgt.mode == 'contain' else tan
    return idx, V, tan, base - cx.dw * deficit


def close_state(cx, st):
    """Cap every open end with the best mono-valent block (greedy). None if some end cannot be capped."""
    while True:
        ends = st.open_ends()
        if not ends:
            return st
        best = None
        for x in ends:
            idx, V, tan, rank = expand(cx, st, x, caps_only=True)
            if V is None:
                return None
            j = int(np.argmax(rank))
            if best is None or rank[j] > best[0]:
                best = (rank[j], x, int(idx[j]))
        st = st.add(best[1], make_inst(cx.lib, best[2]))


def closable(cx, st):
    """Every open end still has at least one cap that satisfies the upper constraints."""
    return all(expand(cx, st, x, caps_only=True)[1] is not None for x in st.open_ends())


# ------------------------------------------------------------------- anchor ----
def pick_anchor(ref, tgt, n_anchor, forced=None, keep=None):
    cuts = F.site_aware_cuts(ref)
    if not cuts:
        sys.exit('reference has no site-aware BRICS bonds: nothing to grow from')
    cm, dums = F.cut_mol(ref, cuts)
    maps = []
    frs = list(rdmolops.GetMolFrags(cm, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=maps))
    fid = {a: k for k, mp_ in enumerate(maps) for a in mp_}
    tables = []
    for fr in frs:
        fr.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(fr)
        tables.append(F.fragment_table(fr))
    sites = E.identify_sites(ref)
    site_frag = [fid[A[0]] for _, A in sites]
    full = erg_from_sites(ref, sites)
    ff = fuzz(full)[0]
    rows = []
    for k in range(len(frs)):
        rest = [s for s, f in zip(sites, site_frag) if f != k]
        v = fuzz(erg_from_sites(ref, rest))[0]
        tc = float(ff @ v) / float(ff @ ff + v @ v - ff @ v)
        n = int(tables[k]['counts'].sum())
        try:
            smi = Chem.MolToSmiles(frs[k])
        except Exception:
            smi = '?'
        rows.append(dict(frag=k, smiles=smi, sites=n, types=int((tables[k]['counts'] > 0).sum()),
                         influence=1.0 - tc))
    rk = pd.DataFrame(rows).sort_values(['influence', 'types'], ascending=False)
    print('\nreference fragments ranked by pharmacophore influence '
          '(influence = 1 - Tanimoto(ErG, ErG without the fragment\'s sites)):')
    print(rk.to_string(index=False))
    chosen = [forced if forced is not None else int(rk.iloc[0].frag)]
    nbr = {}
    for di, dj in dums:
        a, b = fid[di], fid[dj]
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    infl = dict(zip(rk.frag, rk.influence))
    if keep is not None:                       # explicit set of blocks to keep (connected), e.g. all but one leaf
        chosen = [int(c) for c in keep]
    while len(chosen) < n_anchor:
        cand = {j for c in chosen for j in nbr.get(c, ()) if j not in chosen}
        if not cand:
            break
        chosen.append(max(cand, key=lambda j: infl[j]))
    print('anchor fragments:', [(c, rows[c]['smiles']) for c in chosen])
    new = {c: i for i, c in enumerate(chosen)}
    inst = [dict(mol=frs[c], table=tables[c], key=f'anchor{c}', nring=frs[c].GetRingInfo().NumRings(),
                 heavy=sum(1 for a in frs[c].GetAtoms() if a.GetAtomicNum() != 0),
                 ref_atoms={loc: at for loc, at in enumerate(maps[c]) if at < ref.GetNumAtoms()}) for c in chosen]
    # dummy atoms are numbered per fragment by `maps` order
    where = {a: (fid[a], maps[fid[a]].index(a)) for a in (x for p in dums for x in p)}
    links = {}
    for di, dj in dums:
        (fa, da), (fb, db) = where[di], where[dj]
        if fa in new and fb in new:
            links[(new[fa], da)] = (new[fb], db)
            links[(new[fb], db)] = (new[fa], da)
    return inst, links


# ----------------------------------------------------------- 3D pocket (search) ----
_PK = None


def _pk_init(pocket):
    global _PK
    _PK = pocket


def _pk_task(task):
    mol, parent_fixed, anchor_fixed, anchor = task
    return _PK.search_place(mol, parent_fixed, anchor_fixed, anchor)


_PROBES = ('CC', 'C(C)C', 'NC', 'OC', 'SC', 'c1ccccc1', 'C(C)=O')     # first atom = the atom that would bond to the ligand


def _drop_cached(m):
    """Remove the distance-matrix cache RDKit stores as molecule properties (it is copied by RWMol / Chem.Mol)."""
    for p in ('DistanceMatrix', 'DistanceMatrix_Paths'):
        if m.HasProp(p):
            m.ClearProp(p)


def growth_vectors(ref):
    """Growth vectors of a ligand without BRICS bonds: for every atom carrying a hydrogen and every virtual substituent
    (ethyl, isopropyl, N-methylamino, methoxy, methylthio, phenyl, acetyl; BRICS environments exclude terminal atoms, so
    bare CH3 / NH2 / OH probes would never qualify) that would form a BRICS bond with it, one dict(atom, label, dC, partner).
    `label` is the BRICS label of the ligand atom for that bond, dC = 1 if the partner atom is carbon (the same flag the
    cut dummies carry)."""
    out = []
    probes = [Chem.MolFromSmiles(s) for s in _PROBES]
    for a in ref.GetAtoms():
        if a.GetTotalNumHs() == 0:
            continue
        seen = set()
        for pm in probes:
            base = Chem.RWMol(ref)
            at = base.GetAtomWithIdx(a.GetIdx())
            if at.GetNoImplicit() and at.GetNumExplicitHs() > 0:       # bracket atom such as [C@H]: one H leaves
                at.SetNumExplicitHs(at.GetNumExplicitHs() - 1)
            n = base.GetNumAtoms()
            rw = Chem.RWMol(Chem.CombineMols(base.GetMol(), pm))
            rw.AddBond(a.GetIdx(), n, Chem.BondType.SINGLE)
            m2 = rw.GetMol()
            try:
                Chem.SanitizeMol(m2)
            except Exception:
                continue
            first = pm.GetAtomWithIdx(0).GetAtomicNum()
            for (i, j), (li, lj) in BRICS.FindBRICSBonds(m2):
                if {i, j} == {a.GetIdx(), n}:
                    la = int(li) if i == a.GetIdx() else int(lj)
                    key = (la, int(first == 6))
                    if la != 7 and key not in seen:
                        seen.add(key)
                        out.append(dict(atom=a.GetIdx(), label=la, dC=key[1], partner=first))
    return out


def anchor_from_vector(ref, atom, label, dC, partner=None):
    """Anchor = the whole reference with one dummy replacing a hydrogen of `atom` (a growth vector)."""
    rw = Chem.RWMol(ref)
    at = rw.GetAtomWithIdx(atom)
    if at.GetNoImplicit() and at.GetNumExplicitHs() > 0:
        at.SetNumExplicitHs(at.GetNumExplicitHs() - 1)
    d = Chem.Atom(0)
    d.SetIsotope(10 * label + dC)
    d.SetNoImplicit(True)
    di = rw.AddAtom(d)
    rw.AddBond(atom, di, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    _drop_cached(mol)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    inst = [dict(mol=mol, table=F.fragment_table(mol), key='anchorV', nring=mol.GetRingInfo().NumRings(),
                 heavy=ref.GetNumHeavyAtoms(), ref_atoms={i: i for i in range(ref.GetNumAtoms())})]
    return inst, {}


# --------------------------------------------------------------------- main ----
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smiles', required=True, help='reference ligand (use the protonation state of the library)')
    ap.add_argument('--library', default=os.path.join(HERE, 'fragment_counts_pubchem_neutral.parquet'))
    ap.add_argument('--min-count', type=int, default=3)
    ap.add_argument('--max-block-heavy', type=int, default=20)
    ap.add_argument('--max-block-attach', type=int, default=3)
    ap.add_argument('--beam', type=int, default=40)
    ap.add_argument('--per-end', type=int, default=60, help='expansions kept per (state, open end)')
    ap.add_argument('--max-same-block', type=int, default=8, help='beam diversity: states sharing the newest block')
    ap.add_argument('--max-same-vec', type=int, default=4, help='beam diversity: states with an identical ErG vector')
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--score', choices=('tanimoto', 'contain'), default='tanimoto',
                    help="how partial molecules are ranked: 'tanimoto' (similarity of the fuzzy ErG vectors to the target) or "
                         "'contain' (coverage of the target's raw vector minus --overshoot-weight x overshoot)")
    ap.add_argument('--overshoot-weight', type=float, default=1.0)
    ap.add_argument('--extra-heavy', type=int, default=None,
                    help='allowed heavy atoms above the reference (default 4; 8 in growth-vector mode)')
    ap.add_argument('--extra-rings', type=int, default=1, help='allowed rings above the reference')
    ap.add_argument('--anchor-n', type=int, default=1, help='number of connected reference blocks kept')
    ap.add_argument('--anchor-frag', type=int, default=None, help='force the anchor block index')
    ap.add_argument('--anchor-keep', default=None, help='comma-separated block indices kept as the anchor (connected set)')
    ap.add_argument('--freq-weight', type=float, default=0.0, help='rank bonus for common blocks')
    ap.add_argument('--close-top', type=int, default=25)
    # pocket constraints
    ap.add_argument('--no-auto-constraints', action='store_true', help='disable the reference-derived envelope')
    ap.add_argument('--constraints', default=None, help='JSON file with extra pocket constraints (see filters.py)')
    ap.add_argument('--slack', type=int, default=None,
                    help='HA/HD/AR/HY slack of the auto envelope (default 1; 2 in growth-vector mode)')
    ap.add_argument('--extra-distance', type=int, default=None,
                    help='allowed growth of the largest site-site distance (default 1; 6 in growth-vector mode)')
    ap.add_argument('--grow-from-h', action='store_true',
                    help='grow from hydrogens of the reference instead of BRICS fragments (automatic when the reference '
                         'has no BRICS bond)')
    ap.add_argument('--grow-vectors', type=int, default=3,
                    help='number of reference atoms (best pocket clearance first) used as growth vectors')
    ap.add_argument('--deficit-weight', type=float, default=0.15, help='rank penalty for unmet lower bounds')
    ap.add_argument('--no-min-gate', action='store_true', help='do not reject finished molecules with unmet lower bounds')
    # molecule filters
    ap.add_argument('--sa-extra', type=float, default=2.0)
    ap.add_argument('--brenk-extra', type=int, default=0)
    ap.add_argument('--no-pains', action='store_true')
    ap.add_argument('--allow-enamine', action='store_true', help='keep enamines (C=C-N) in blocks and products')
    ap.add_argument('--allow-motifs', action='store_true',
                    help='keep nitroso, sulfoxide and vicinal/geminal dinitrile motifs')
    ap.add_argument('--no-filters', action='store_true', help='disable PAINS/Brenk/SA on finished molecules')
    # PubChem knowledge (known.py): reward molecules that exist in PubChem, derank novel ones
    ap.add_argument('--known-index', default=None, help='directory of a built known index (build_known_index.py)')
    ap.add_argument('--known-weight', type=float, default=0.1, help='score added to a molecule found in PubChem')
    ap.add_argument('--novel-penalty', type=float, default=0.0, help='score subtracted from a molecule not found')
    ap.add_argument('--known-level', choices=('closed', 'all'), default='closed',
                    help="'closed': look up finished molecules only; 'all': also every beam state at every growth "
                         "level (open attachments capped with H) and rank the beam with it")
    ap.add_argument('--known-override', action='store_true',
                    help='molecules found in PubChem are ranked above all novel ones (beam at level all, final table), '
                         'whatever their score')
    ap.add_argument('--pubchem-idx', default=paths.pubchem_idx(),
                    help='ErG index with the PubChem ids / SMILES (to report pubchem_id of known hits)')
    # excluded volume
    ap.add_argument('--complex', default=None, help='PDB of the protein-ligand complex (enables the excluded-volume check)')
    ap.add_argument('--pocket-sdf', default=None, help='SDF with the crystal ligand(s)')
    ap.add_argument('--pocket-lig', default=None, help='name of the ligand in the SDF (e.g. Y70_B_502)')
    ap.add_argument('--no-pocket-search', action='store_true',
                    help='use the excluded-volume check only as a final filter, not inside the search')
    ap.add_argument('--w3d', type=float, default=0.15,
                    help='weight of the 3D burial/contact score (0..1) in the search rank; 0 = prune only')
    ap.add_argument('--w-polar', type=float, default=0.3,
                    help='share of the polar-complementarity term inside the 3D score (0..1)')
    ap.add_argument('--pocket-max-unsat', type=int, default=-1,
                    help='hard limit on buried unsatisfied polar atoms among the new atoms (-1 = off)')
    ap.add_argument('--pocket-pool', type=float, default=2.0,
                    help='3D-checked candidates per beam slot before re-ranking with the 3D score')
    ap.add_argument('--pocket-drop-het', default='',
                    help='comma-separated residue names of co-crystallised ligands to ignore as obstacles (overlapping '
                         'ligands are dropped automatically)')
    ap.add_argument('--pocket-radius', type=float, default=14.0)
    ap.add_argument('--pocket-confs', type=int, default=20)
    ap.add_argument('--pocket-max-clash', type=int, default=2,
                    help='non-anchor heavy atoms allowed to clash (calibrated on 6h0f: known CRBN binders 0-2, groups in the cage >= 14)')
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(HERE, 'grow_results.csv'))
    return ap


_KNOWN = {}


def get_known(path):
    if path not in _KNOWN:
        import known as KN
        _KNOWN[path] = KN.KnownIndex(path)
    return _KNOWN[path]


def sort_results(df, a):
    cols, asc = [], []
    if a.known_index and a.known_override and 'pubchem_known' in df:
        cols.append('pubchem_known')
        asc.append(False)
    cols.append('score')
    asc.append(False)
    return df.sort_values(cols, ascending=asc)


def annotate_pubchem(df, a):
    """pubchem_id / pubchem_smiles of the known hits, from the ErG index (rows -> id, SMILES)."""
    if not len(df) or 'pubchem_known' not in df or not df.pubchem_known.any():
        return df
    try:
        sys.path.insert(0, os.path.join(HERE, '..'))
        import erg_index as X
        ix = X.Index(a.pubchem_idx)
        sel = df.pubchem_known.values
        ids, smi = ix.fetch(df.loc[sel, 'pubchem_row'].astype(int).values)
        df['pubchem_id'] = None
        df['pubchem_smiles'] = None
        df.loc[sel, 'pubchem_id'] = ids
        df.loc[sel, 'pubchem_smiles'] = smi
    except Exception as e:
        print(f'(could not fetch PubChem ids: {e})')
    return df


_NORM = {}


def _norm_key(k):
    r = _NORM.get(k)
    if r is None:
        m = Chem.MolFromSmiles(k)
        r = Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else k
        _NORM[k] = r
    return r


def _on_path(st, new_key, trace):
    """True if the blocks added to the anchor of `st` (plus `new_key`) are a sub-multiset of the blocks of the
    molecule being rebuilt: a necessary condition for the state to lead to it (diagnostic for rebuild benchmarks)."""
    from collections import Counter
    have = Counter(_norm_key(i['key']) for i in st.inst if not i['key'].startswith('anchor'))
    if new_key is not None:
        have[_norm_key(new_key)] += 1
    return all(trace.get(k, 0) >= c for k, c in have.items())


def run_grow(a, lib, ref_smiles, target_smiles=None, rng=None, verbose=True, pocket=None, anchor_vec=None):
    """Grow around the anchor of `ref_smiles`.
    target_smiles: ErG target (default: the reference itself; a different molecule gives the 'wrong target' control).
    rng: numpy Generator -> random-guidance control (expansions, beam and caps picked at random; constraints,
         anchor and filters unchanged).
    Returns (DataFrame of accepted molecules sorted by ERG Tanimoto to the target, info dict)."""
    log = print if verbose else (lambda *x, **k: None)
    ref = Chem.MolFromSmiles(ref_smiles)
    ref_can = Chem.MolToSmiles(ref, isomericSmiles=False)
    ref_raw = E.erg_raw(ref).astype(np.int64)
    tgt_mol = ref if target_smiles is None else Chem.MolFromSmiles(target_smiles)
    tgt = Target(E.erg_raw(tgt_mol).astype(np.int64), getattr(a, 'score', 'tanimoto'),
                 getattr(a, 'overshoot_weight', 1.0))
    vm = anchor_vec is not None
    keep_ = [int(x) for x in a.anchor_keep.split(',')] if getattr(a, 'anchor_keep', None) else None
    trace = getattr(a, 'trace_keys', None)             # {normalised block key: count} of the molecule to rebuild
    eh = a.extra_heavy if a.extra_heavy is not None else (8 if vm else 4)
    sl = a.slack if a.slack is not None else (2 if vm else 1)
    ed = a.extra_distance if a.extra_distance is not None else (6 if vm else 1)
    max_heavy = ref.GetNumHeavyAtoms() + eh
    max_rings = ref.GetRingInfo().NumRings() + a.extra_rings
    fp_ref = AllChem.GetMorganFingerprintAsBitVect(ref, 2, 2048)

    cons = FL.Constraints() if a.no_auto_constraints else FL.Constraints.from_reference(ref_raw, sl, sl, ed)
    if a.constraints:
        cons = FL.Constraints.from_json(a.constraints, base=cons)
    cons.finalize()
    log('pocket constraints:')
    for d in cons.describe():
        log('   ', d)
    mf = None if a.no_filters else FL.MolFilter(ref, a.sa_extra, a.brenk_extra, not a.no_pains)
    if mf:
        log(f'filters: PAINS={not a.no_pains}, Brenk <= {mf.ref_brenk}+{a.brenk_extra}, '
            f'SA <= {mf.ref_sa if mf.ref_sa is None else round(mf.ref_sa, 2)}+{a.sa_extra}; '
            f'rings <= {max_rings}, heavy <= {max_heavy}')

    t0 = time.time()
    if pocket is not None:
        pocket.set_reference(ref)
        log(f'{pocket}; excluded-volume rule: reject if more than {a.pocket_max_clash} non-anchor heavy atoms clash')
    search3d = pocket is not None and not a.no_pocket_search
    kidx = get_known(a.known_index) if a.known_index else None
    klevel_all = kidx is not None and a.known_level == 'all'
    use_comb = (search3d and a.w3d > 0) or klevel_all
    kadj = (lambda known: a.known_weight if known else -a.novel_penalty) if kidx is not None else (lambda known: 0.0)
    if kidx is not None:
        log(f'PubChem knowledge: {len(kidx):,} keys, reward {a.known_weight:+} for known, {-a.novel_penalty:+} for novel, '
            f'level {a.known_level}' + (', known molecules ranked first' if a.known_override else ''))
    pool = None
    if search3d:
        if a.workers > 1 and not mp.current_process().daemon:
            pool = mp.Pool(a.workers, initializer=_pk_init, initargs=(pocket,))
        else:
            _pk_init(pocket)
        pocket.score_poses = a.w3d > 0 or a.pocket_max_unsat >= 0
        pocket.w_polar = a.w_polar
        log(f'3D excluded-volume check is part of the search; 3D score weight {a.w3d} '
            f'({a.workers if pool else 1} worker{"s" if pool else ""})')
    cx = Ctx(lib, tgt, cons, max_heavy, max_rings, a.deficit_weight, rng)
    if vm:
        inst, links = anchor_from_vector(ref, **anchor_vec)
        log(f'growth vector: atom {anchor_vec["atom"]} ({ref.GetAtomWithIdx(anchor_vec["atom"]).GetSymbol()}), '
            f'BRICS label {anchor_vec["label"]}, partner {"carbon" if anchor_vec["dC"] else "hetero atom"}')
    elif verbose:
        inst, links = pick_anchor(ref, tgt, a.anchor_n, a.anchor_frag, keep=keep_)
    else:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            inst, links = pick_anchor(ref, tgt, a.anchor_n, a.anchor_frag, keep=keep_)
    anchor = State(inst, links)
    if search3d:
        anchor.pose = {(f, loc): pocket.ref_xyz[r] for f, i in enumerate(inst)
                       for loc, r in i['ref_atoms'].items() if r in pocket.ref_xyz}
    up, _, _ = cons.check(anchor.vec[None, :].astype(np.float32))
    log(f'\nanchor: {anchor.heavy} heavy atoms, {len(anchor.open_ends())} open attachments, '
        f'ErG Tanimoto to target {tgt.tanimoto(anchor.vec[None, :])[0]:.3f}; '
        f'reference {ref.GetNumHeavyAtoms()} heavy atoms (limit {max_heavy})'
        + ('   [anchor itself violates an upper constraint!]' if up[0] else '') + '\n')

    results, rejected = {}, {}
    stat = dict(mismatch=0, predicted_bad=0, unclosable=0, ref_found=False, pocket_pruned=0)

    def harvest(states):
        for st in states:
            c = close_state(cx, st)
            if c is None:
                rejected['cannot close'] = rejected.get('cannot close', 0) + 1
                continue
            mol, keys = c.assemble()
            if mol is None:
                rejected['build/sanitize'] = rejected.get('build/sanitize', 0) + 1
                continue
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            if smi == ref_can:
                stat['ref_found'] = True
                continue
            if smi in results:
                continue
            if not a.allow_enamine and mol.HasSubstructMatch(FL.ENAMINE):
                rejected['enamine'] = rejected.get('enamine', 0) + 1
                continue
            if FL.has_radical(mol):
                rejected['radical'] = rejected.get('radical', 0) + 1
                continue
            if not a.allow_motifs and FL.bad_motif(mol):
                rejected['motif'] = rejected.get('motif', 0) + 1
                continue
            _, _, unmet = cons.check(c.vec[None, :].astype(np.float32))
            if unmet[0] and not a.no_min_gate:
                rejected['pocket lower bound'] = rejected.get('pocket lower bound', 0) + 1
                continue
            info = dict(sa=np.nan, brenk=0, pains=0)
            if mf:
                why, info = mf.reason(mol)
                if why:
                    rejected[why] = rejected.get(why, 0) + 1
                    continue
            pe = {}
            if pocket is not None:
                if search3d and st.pose:
                    anc, afix = pocket.anchor_info(mol)
                    pfix = {i: st.pose[k] for i, k in enumerate(keys) if k in st.pose}
                    pe = pocket.search_place(mol, pfix, afix, anc)
                    pe.pop('xyz', None)
                else:
                    pe = pocket.evaluate(mol)
                if pe['clash'] > a.pocket_max_clash or (a.pocket_max_unsat >= 0 and pe.get('n_unsat', 0) > a.pocket_max_unsat):
                    rejected['excluded volume'] = rejected.get('excluded volume', 0) + 1
                    continue
            kn = kidx.lookup(mol) if kidx is not None else (False, -1, 0)
            ok = bool((np.minimum(c.vec, 255) == E.erg_raw(mol).astype(np.int64)).all())
            stat['mismatch'] += (not ok)
            results[smi] = dict(smiles=smi, erg_tanimoto=float(tgt.tanimoto(c.vec[None, :])[0]),
                                erg_to_ref=float(Target(ref_raw).tanimoto(c.vec[None, :])[0]),
                                heavy=mol.GetNumHeavyAtoms(), blocks=len(c.inst),
                                block_keys='|'.join(i['key'] for i in c.inst),
                                ecfp4_to_ref=DataStructs.TanimotoSimilarity(
                                    fp_ref, AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)),
                                sa=info['sa'], brenk=info['brenk'], erg_verified=ok,
                                **{('pocket_' + k): v for k, v in pe.items()},
                                pubchem_known=bool(kn[0]), pubchem_row=kn[1], pubchem_hits=kn[2],
                                score=float(tgt.tanimoto(c.vec[None, :])[0]) + (a.w3d * pe.get('s3d', 0.0) if search3d else 0.0)
                                + kadj(kn[0]))

    harvest([anchor])
    beam = [anchor]
    for step in range(1, a.steps + 1):
        cands = []
        for si, st in enumerate(beam):
            for x in st.open_ends():
                idx, V, tan, rank = expand(cx, st, x)
                if V is None:
                    continue
                rank = rank + a.freq_weight * np.log10(lib['cnt'][idx]) / 6.0
                for j in np.argsort(-rank)[:a.per_end]:
                    cands.append((float(rank[j]), si, x, int(idx[j]), V[j]))
        if not cands:
            log(f'step {step}: no expansions')
            break
        cands.sort(key=lambda c: -c[0])
        tr = None
        if trace is not None:
            pos = [k for k, c in enumerate(cands)
                   if _on_path(beam[c[1]], lib['keys'][lib['kid'][c[3]]], trace)]
            tr = dict(step=step, cands=len(cands), on_path_cands=len(pos), best_pos=(pos[0] if pos else -1))
        want = int(np.ceil(a.beam * (a.pocket_pool if use_comb else 1)))
        new, seen, per_block, per_vec, pending = [], set(), {}, {}, []

        def flush():
            """3D check of the realised states in `pending` (in parallel), keep the feasible ones."""
            if not pending:
                return
            tasks = []
            for st_, si_, key_, vk_ in pending:
                mol_, keys_ = st_.assemble(partial=True)
                if mol_ is None:
                    tasks.append(None)
                    continue
                anc, afix = pocket.anchor_info(mol_)
                par = beam[si_].pose or {}
                pfix = {i: par[k] for i, k in enumerate(keys_) if k in par}
                tasks.append(((mol_, pfix, afix, anc), keys_))
            todo = [k for k, t in enumerate(tasks) if t is not None]
            payload = [tasks[k][0] for k in todo]
            out = pool.map(_pk_task, payload, chunksize=1) if pool else [_pk_task(x) for x in payload]
            res = dict(zip(todo, out))
            for k, (st_, si_, key_, vk_) in enumerate(pending):
                if k in res:
                    r = res[k]
                    if r['xyz'] is None or r['clash'] > a.pocket_max_clash or (
                            a.pocket_max_unsat >= 0 and r['n_unsat'] > a.pocket_max_unsat):
                        stat['pocket_pruned'] += 1
                        per_block[key_] -= 1
                        per_vec[vk_] -= 1
                        continue
                    keys_ = tasks[k][1]
                    st_.pose = {keys_[i]: r['xyz'][i] for i in range(len(keys_))}
                    st_.pocket = {q: v for q, v in r.items() if q != 'xyz'}
                else:
                    st_.pose = beam[si_].pose
                new.append(st_)
            pending.clear()

        for rank, si, x, e, V in cands:
            if len(new) + len(pending) >= want:
                flush()
            if len(new) >= want:
                break
            key = lib['keys'][lib['kid'][e]]
            if per_block.get(key, 0) >= a.max_same_block:
                continue
            vk = np.rint(V).astype(np.int64).tobytes()
            if per_vec.get(vk, 0) >= a.max_same_vec:
                continue
            st = beam[si].add(x, make_inst(lib, e))
            if st.sig in seen:
                continue
            seen.add(st.sig)
            if not closable(cx, st):
                stat['unclosable'] += 1
                continue
            per_block[key] = per_block.get(key, 0) + 1
            per_vec[vk] = per_vec.get(vk, 0) + 1
            stat['predicted_bad'] += int(not (np.rint(V).astype(np.int64) == st.vec).all())
            if klevel_all:
                mp_, _ = st.assemble(partial=True)
                st.known = bool(kidx.lookup(mp_)[0]) if mp_ is not None else False
            if search3d:
                pending.append((st, si, key, vk))
            else:
                new.append(st)
        if search3d:
            flush()
        if use_comb and new:
            tn = tgt.rank_score(np.array([s.vec for s in new]))
            _, dfc, _ = cons.check(np.array([s.vec for s in new]).astype(np.float32))
            comb = tn - a.deficit_weight * dfc
            if search3d and a.w3d > 0:
                comb = comb + a.w3d * np.array([(s.pocket or {}).get('s3d', 0.0) for s in new])
            kn_ = np.array([bool(s.known) for s in new])
            if klevel_all:
                comb = comb + np.where(kn_, a.known_weight, -a.novel_penalty)
            order_ = np.lexsort((-comb, -kn_.astype(int))) if (klevel_all and a.known_override) else np.argsort(-comb)
            new = [new[i] for i in order_]
        new = new[:a.beam]
        if tr is not None:
            tr['beam'] = len(new)
            tr['beam_on_path'] = sum(1 for s_ in new if _on_path(s_, None, trace))
            stat.setdefault('trace', []).append(tr)
        if not new:
            log(f'step {step}: every expansion is unclosable or violates a constraint; stopping')
            break
        beam = new
        scores = tgt.rank_score(np.array([s.vec for s in beam]))
        order = rng.permutation(len(beam)) if rng is not None else np.argsort(-scores)
        beam = [beam[i] for i in order]
        harvest(beam[:a.close_top])
        bestc = max((r['erg_tanimoto'] for r in results.values()), default=float('nan'))
        log(f'step {step}: beam {len(beam)}, best partial {scores.max():.3f}, '
            f'{len(results)} accepted molecules, best {bestc:.3f} ({time.time() - t0:.0f}s)')

    if pool is not None:
        pool.close()
        pool.join()
    df = pd.DataFrame(list(results.values()))
    if len(df):
        df = sort_results(df, a).reset_index(drop=True)
    info = dict(stat=stat, rejected=rejected, ref_can=ref_can, anchor_heavy=anchor.heavy,
                anchor_keys=[i['key'] for i in inst])
    return df, info


def main():
    a = build_parser().parse_args()
    t0 = time.time()
    lib = load_library(a.library, a.min_count, a.max_block_heavy, a.max_block_attach, a.workers,
                       a.allow_enamine, a.allow_motifs)
    print(f'library ready in {time.time() - t0:.0f}s')
    pocket = None
    if a.complex:
        import pocket as PK
        pocket = PK.Pocket(a.complex, a.pocket_sdf, a.pocket_lig, a.pocket_radius, a.pocket_confs,
                           accept=a.pocket_max_clash, drop_het=[x for x in a.pocket_drop_het.split(',') if x])
    ref = Chem.MolFromSmiles(a.smiles)
    if a.grow_from_h or not F.site_aware_cuts(ref):
        if not a.grow_from_h:
            print('the reference has no BRICS bond: switching to growth-vector mode (growing from its hydrogens)')
        vecs = growth_vectors(ref)
        if not vecs:
            sys.exit('no growth vector found: the reference has no hydrogen that can carry a BRICS substituent')
        clr = {}
        if pocket is not None:
            pocket.set_reference(ref)
            clr = pocket.clearances(ref)
            print('atom clearances beyond the ligand (A, to the nearest protein atom):',
                  {k: round(v, 1) for k, v in sorted(clr.items(), key=lambda kv: -kv[1])})
        atoms = sorted({v['atom'] for v in vecs}, key=lambda i: -clr.get(i, 0.0))[:a.grow_vectors]
        vecs = [v for v in vecs if v['atom'] in atoms]
        print(f'{len(vecs)} growth vectors on atoms {atoms}: '
              + ', '.join(f"{v['atom']}:L{v['label']}{'C' if v['dC'] else 'X'}" for v in vecs))
        dfs, infos = [], []
        for v in vecs:
            d, inf = run_grow(a, lib, a.smiles, pocket=pocket, anchor_vec=v)
            d['vector'] = f"{v['atom']}:L{v['label']}{'C' if v['dC'] else 'X'}"
            dfs.append(d)
            infos.append(inf)
            print(f"  vector atom {v['atom']} label {v['label']}: {len(d)} accepted molecules", flush=True)
        df = pd.concat([d for d in dfs if len(d)], ignore_index=True) if any(len(d) for d in dfs) else pd.DataFrame()
        if len(df):
            df = sort_results(df, a).drop_duplicates('smiles').reset_index(drop=True)
        info = infos[0]
        for k in ('mismatch', 'predicted_bad', 'unclosable', 'pocket_pruned'):
            info['stat'][k] = sum(i['stat'][k] for i in infos)
        info['stat']['ref_found'] = any(i['stat']['ref_found'] for i in infos)
        rej = {}
        for i in infos:
            for k, c in i['rejected'].items():
                rej[k] = rej.get(k, 0) + c
        info['rejected'] = rej
    else:
        df, info = run_grow(a, lib, a.smiles, pocket=pocket)
    df = annotate_pubchem(df, a)
    df.to_csv(a.out, index=False)
    if a.known_index and len(df):
        print(f'PubChem: {int(df.pubchem_known.sum())} of {len(df)} accepted molecules are known '
              f'({a.known_index})')
    st = info['stat']
    print(f'\nincremental-vs-assembled mismatches: {st["predicted_bad"]}; '
          f'incremental-vs-from-scratch ErG mismatches among {len(df)} accepted molecules: {st["mismatch"]}')
    print(f'pruned as unclosable during growth: {st["unclosable"]}; pruned by the 3D pocket check during '
          f'growth: {st["pocket_pruned"]}; rejected at closing: {info["rejected"]}; '
          f'reference itself rediscovered: {st["ref_found"]}')
    print(f'reference: {info["ref_can"]}\n')
    pd.set_option('display.width', 250, 'display.max_colwidth', 95)
    if len(df):
        print(df.drop(columns=['block_keys', 'erg_to_ref'], errors='ignore').head(a.top).to_string(
            index=False, float_format=lambda v: f'{v:.3f}'))
    else:
        print('no molecule was accepted (see the rejection counts above)')
    print(f'\nwritten {a.out}')


if __name__ == '__main__':
    main()
