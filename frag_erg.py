"""
Fragment-additive ErG v2.

A molecule is cut at site-aware BRICS bonds. Every cut is a bridge, so for sites p (fragment F) and
q (fragment G) the topological distance is  d(p,a) + 1 + [d(b,..) chain] + d(.,q): the cross-fragment
part of the fingerprint is a tree convolution of per-fragment attachment profiles. Nothing about
(fragment A, fragment B) connectors needs to be stored.

Site-aware: a BRICS bond is skipped if both its atoms belong to one site (only HY clusters can do this),
so no site is ever split. The only context a fragment needs from across a cut is whether the
neighbouring atom is carbon (HY rule) -> encoded in the dummy isotope: iso = 10*brics_label + isC.

Per-fragment table (context free, keyed by the canonical dummy-SMILES of the fragment):
    counts[6]            site counts
    internal[315]        pairs inside the fragment
    prof[dummy][6, L]    sites by distance to the attachment atom of that dummy
    dd[dummy, dummy]     attachment-atom to attachment-atom distance
"""
import sys, os
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, rdmolops

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import erg_v2 as E

RDLogger.DisableLog('rdApp.*')
NB, L = E.NBINS, 256
_AR, _HY, _PLUS, _MINUS, _HA, _HD = E._AR, E._HY, E._PLUS, E._MINUS, E._HA, E._HD


# ------------------------------------------------------------------ cutting ----
def site_aware_cuts(m):
    """[(i, j, label_i, label_j)] BRICS bonds that do not split a site."""
    sets = [set(a) for _, a in E.identify_sites(m)]
    seen, out = set(), []
    for (i, j), (li, lj) in BRICS.FindBRICSBonds(m):
        k = (min(i, j), max(i, j))
        if k in seen or any(i in s and j in s for s in sets):
            continue
        seen.add(k)
        out.append((i, j, int(li), int(lj)))
    return out


def cut_mol(m, cuts):
    """Molecule with the cut bonds replaced by dummies. Returns (mol, [(dummy_i, dummy_j)])."""
    rw = Chem.RWMol(m)
    dums = []
    for i, j, li, lj in cuts:
        bt = m.GetBondBetweenAtoms(i, j).GetBondType()
        if bt == Chem.BondType.AROMATIC:
            bt = Chem.BondType.SINGLE
        rw.RemoveBond(i, j)
        pair = []
        for a, lab, other in ((i, li, j), (j, lj, i)):
            d = Chem.Atom(0)
            d.SetIsotope(10 * lab + (1 if m.GetAtomWithIdx(other).GetAtomicNum() == 6 else 0))
            d.SetNoImplicit(True)
            di = rw.AddAtom(d)
            rw.AddBond(a, di, bt)
            pair.append(di)
        dums.append(tuple(pair))
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(out)
    return out, dums


def fragment_keys(m, cuts=None):
    """Canonical dummy-SMILES of each fragment (the building-block identity)."""
    cuts = site_aware_cuts(m) if cuts is None else cuts
    if not cuts:
        return [Chem.MolToSmiles(m)]
    cm, _ = cut_mol(m, cuts)
    return Chem.MolToSmiles(cm).split('.')


# ------------------------------------------------------------ fragment table ----
def sites_fragment(fr, with_ok=False):
    """E.identify_sites for a fragment with dummies (Z=0, iso%10==1 -> neighbour is carbon)."""
    atoms = list(fr.GetAtoms())
    n = len(atoms)
    Z = [a.GetAtomicNum() for a in atoms]
    real = [z != 0 for z in Z]
    Zeff = [6 if (z == 0 and a.GetIsotope() % 10 == 1) else z for z, a in zip(Z, atoms)]
    sites = []
    for r in fr.GetRingInfo().AtomRings():
        if len(r) in (5, 6, 7) and all(atoms[i].GetIsAromatic() for i in r):
            sites.append((_AR, list(r)))
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in fr.GetBonds()]
    hetero = bytearray(n)
    for i, j in bonds:
        if Zeff[i] != 6 or Zeff[j] != 6:
            hetero[i] = hetero[j] = 1
    ok = [real[i] and Z[i] == 6 and not hetero[i] and not atoms[i].GetIsAromatic()
          and atoms[i].GetFormalCharge() == 0 for i in range(n)]
    if sum(ok) >= 3:
        adj = {}
        for i, j in bonds:
            if ok[i] and ok[j]:
                adj.setdefault(i, []).append(j)
                adj.setdefault(j, []).append(i)
        seen = set()
        for s in adj:
            if s in seen:
                continue
            comp, stack = [], [s]
            seen.add(s)
            while stack:
                c = stack.pop()
                comp.append(c)
                for k in adj[c]:
                    if k not in seen:
                        seen.add(k)
                        stack.append(k)
            if len(comp) >= 3:
                sites.append((_HY, comp))
    for i, a in enumerate(atoms):
        if not real[i]:
            continue
        q = a.GetFormalCharge()
        if q > 0:
            sites.append((_PLUS, [i]))
        elif q < 0:
            sites.append((_MINUS, [i]))
        elif Z[i] in E._HA_Z:
            sites.append((_HA, [i]))
            if Z[i] in (7, 8) and a.GetTotalNumHs() > 0:
                sites.append((_HD, [i]))
    return (sites, ok) if with_ok else sites


def fragment_table(fr):
    sites, hyok = sites_fragment(fr, with_ok=True)
    S = len(sites)
    ftype = np.array([f for f, _ in sites], dtype=np.int64)
    counts = np.bincount(ftype, minlength=6)
    # force=True: a molecule copied from one whose distance matrix was cached (Chem.RWMol(ref) after erg_raw(ref))
    # carries that stale cache and would return garbage for a changed atom count
    dm = np.array(rdmolops.GetDistanceMatrix(fr, force=True))
    n = len(dm)
    internal = np.zeros(21 * NB, dtype=np.int64)
    if S > 1:
        d2 = dm.copy()
        np.fill_diagonal(d2, E._BIG)
        P = np.zeros((S, n), dtype=bool)
        for k, (_, A) in enumerate(sites):
            P[k, A] = True
        M = np.where(P[:, :, None], d2[None, :, :], E._BIG).min(1)
        D = np.where(P[None, :, :], M[:, None, :], E._BIG).min(2)
        i, j = E._triu(S)
        d = D[i, j]
        keep = d < 1e6
        d = np.clip(d[keep], 1, NB).astype(np.int64)
        pt = E._PT_LUT[ftype[i[keep]], ftype[j[keep]]]
        internal = np.bincount(pt * NB + d - 1, minlength=21 * NB)
    # attachments: dummy atoms (Z==0) each hang on exactly one real atom
    dummies = [a.GetIdx() for a in fr.GetAtoms() if a.GetAtomicNum() == 0]
    attach = {d: fr.GetAtomWithIdx(d).GetNeighbors()[0].GetIdx() for d in dummies}
    prof = {}
    for d, at in attach.items():
        h = np.zeros((6, L), dtype=np.int64)
        for k, (f, A) in enumerate(sites):
            h[f, int(dm[A, at].min())] += 1
        prof[d] = h
    dd = {(a, b): int(dm[attach[a], attach[b]]) for a in dummies for b in dummies}
    return dict(counts=counts, internal=internal, prof=prof, dd=dd, attach=attach,
                hyok={d: bool(hyok[a]) for d, a in attach.items()},
                attach_c={d: fr.GetAtomWithIdx(a).GetAtomicNum() == 6 for d, a in attach.items()})


# ------------------------------------------------------------------ assembly ----
def _shift(h, s):
    out = np.zeros_like(h)
    if s < h.shape[1]:
        out[:, s:] = h[:, :L - s]
    return out


def assemble(tables, links):
    """tables[f] = fragment_table; links = {(f, dummy): (g, dummy2)} (symmetric).
    Returns raw int vector [321] (6 counts + 315), not saturated."""
    ends = {}
    for (f, d), (g, e) in links.items():
        ends.setdefault(f, []).append(d)

    def reach(g, y):
        """histogram [6,L] of sites in g and beyond, by distance to attachment atom of dummy y in g."""
        t = tables[g]
        H = t['prof'][y].copy()
        for z in ends.get(g, []):
            if z == y:
                continue
            h, w = links[(g, z)]
            H += _shift(reach(h, w), t['dd'][(y, z)] + 1)
        return H

    counts = sum(t['counts'] for t in tables)
    hist = sum(t['internal'] for t in tables).astype(np.int64)
    cross = np.zeros(21 * NB, dtype=np.int64)
    for f, t in enumerate(tables):
        for x in ends.get(f, []):
            g, y = links[(f, x)]
            far = _shift(reach(g, y), 1)
            near = t['prof'][x]
            for s in range(6):
                if not near[s].any():
                    continue
                for u in range(6):
                    if not far[u].any():
                        continue
                    c = np.convolve(near[s], far[u])
                    # c[k] = ordered pairs at distance k
                    c2 = np.zeros(NB, dtype=np.int64)
                    c2[:NB - 1] = c[1:NB]
                    c2[NB - 1] = c[NB:].sum()
                    p = E._PT_LUT[s, u]
                    cross[p * NB:(p + 1) * NB] += c2
    assert not (cross % 2).any()
    hist += cross // 2
    return np.concatenate([counts, hist])


def fragment_erg(m):
    """Assemble the raw ErG v2 vector of m from per-fragment tables only (no whole-molecule ErG)."""
    cuts = site_aware_cuts(m)
    if not cuts:
        v = fragment_table(m)
        return np.concatenate([v['counts'], v['internal']])
    cm, dums = cut_mol(m, cuts)
    maps = []
    frs = rdmolops.GetMolFrags(cm, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=maps)
    where = {a: (k, loc) for k, mp in enumerate(maps) for loc, a in enumerate(mp)}
    tables = []
    for fr in frs:
        fr.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(fr)
        tables.append(fragment_table(fr))
    links = {}
    for di, dj in dums:
        a, b = where[di], where[dj]
        links[a] = b
        links[b] = a
    return assemble(tables, links)
