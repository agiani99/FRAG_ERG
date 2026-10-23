"""
Chemistry filters and pocket constraints for the ErG grow loop.

Block filters (library level, applied to dummy-SMILES blocks):
    reactive / unstable / odd-charge groups, ring systems fused from > MAX_FUSED rings.
Molecule filters (on every closed molecule): PAINS (reject), Brenk alerts (<= reference + k),
    synthetic accessibility (<= reference + k, if RDKit's Contrib SA_Score is installed).
Pocket constraints: linear rules on the raw ErG v2 vector
        lo <= w . v <= hi,      v = [6 feature counts | 21 pair types x 15 distance bins]
    Every ErG entry only ever grows while a molecule is built, so the upper bounds (hi) are
    exact pruning rules for partial molecules; the lower bounds (lo) steer the rank (deficit
    penalty) and gate the finished molecule.
    Constraint kinds (JSON list for --constraints):
        {"type":"feature_count","feature":"HD","min":1,"max":3}
        {"type":"pair_count","pair":"HD_HA","dmin":2,"dmax":4,"min":1,"max":null}
        {"type":"max_distance","value":12}          # no site pair further apart than 12 bonds (<=14)
    and --auto-constraints derives an envelope from the reference ligand (see from_reference).
"""
import json, os, sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

import erg_v2 as E

NB = E.NBINS
MAX_FUSED = 3

# enamine: non-aromatic C=C-N, N a non-amide, non-sulfonamide amine. The first carbon may be a dummy
# (label-7 attachment) so that blocks like [71*]=C(N...) are caught before they are joined.
ENAMINE = Chem.MolFromSmarts('[#6,#0;!a]=[CX3;!a]-[NX3;!a;!$(N-C=O);!$(N-S=O)]')

_BLOCK_BAD = [Chem.MolFromSmarts(s) for s in (
    '[CX2]#[CX2]',                        # alkynes
    '[#7]=[#7]',                          # azo / diazo
    '[#8]-[#8]',                          # peroxide
    '[#6-,#7-,#16-,#15-]',                # carbanions / nitrogen anions
    '[#7]=[#6]=[#8,#16]',                 # iso(thio)cyanate
    '[CX3](=O)[F,Cl,Br,I]',               # acyl halide
    '[SX4](=O)(=O)[F,Cl,Br,I]',           # sulfonyl halide
    '[CX3H1]=O',                          # aldehyde
    '[#7;!R]-[#7;!R]-[#7;!R]',            # triazane
    '[Si,B,Se,As,Hg,Sn,Pb]',
    '[#15;!$([#15]=[#8])]',               # phosphorus other than P=O (phosphines, ylides)
    '[#1]',                               # explicit hydrogens
    '[I]-[CX3]=[CX3]',                    # vinyl iodides
    '[#7+]#[#6]',                         # nitrilium (bogus protonation of nitriles)
    '[a;r3,r4]',                          # aromatic 3/4-membered rings
)]


# other motifs ErG-guided growth likes but chemistry does not (Brenk does not flag them)
MOTIFS = [Chem.MolFromSmarts(s) for s in (
    '[#7X2]=[#8]',                        # nitroso
    '[#16X3]=[#8]',                       # sulfoxide
    '[#16+]-[#8-]',
    'N#C[#6]C#N',                         # geminal dinitrile
    'N#C[#6]~[#6]C#N',                    # vicinal dinitrile
)]


def bad_motif(m):
    return any(m.HasSubstructMatch(p) for p in MOTIFS)


def has_radical(m):
    return any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms())


def ring_systems(m):
    """Sizes (number of rings) of fused ring systems."""
    rings = [set(r) for r in m.GetRingInfo().BondRings()]
    systems = []
    for r in rings:
        merged = [s for s in systems if s['bonds'] & r]
        new = {'bonds': set(r), 'n': 1}
        for s in merged:
            new['bonds'] |= s['bonds']
            new['n'] += s['n']
            systems.remove(s)
        systems.append(new)
    return [s['n'] for s in systems]


def block_ok(m, allow_enamine=False, allow_motifs=False):
    """m: library block (with dummies). Returns (ok, n_rings)."""
    if any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms()):      # e.g. '[N]' without hydrogen
        return False, 0
    if any(m.HasSubstructMatch(p) for p in _BLOCK_BAD):
        return False, 0
    if not allow_enamine and m.HasSubstructMatch(ENAMINE):
        return False, 0
    if not allow_motifs and bad_motif(m):
        return False, 0
    rs = ring_systems(m)
    if rs and max(rs) > MAX_FUSED:
        return False, 0
    return True, m.GetRingInfo().NumRings()


# ------------------------------------------------------------ molecule level ----
class MolFilter:
    def __init__(self, ref, sa_extra=2.0, brenk_extra=0, use_pains=True):
        p = FilterCatalogParams()
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        self.pains = FilterCatalog(p)
        p = FilterCatalogParams()
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        self.brenk = FilterCatalog(p)
        self.sa = None
        try:
            sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
            import sascorer
            self.sa = sascorer.calculateScore
        except Exception:
            pass
        self.use_pains, self.sa_extra, self.brenk_extra = use_pains, sa_extra, brenk_extra
        self.ref_brenk = len(self.brenk.GetMatches(ref))
        self.ref_sa = self.sa(ref) if self.sa else None

    def info(self, mol):
        return dict(sa=self.sa(mol) if self.sa else np.nan,
                    brenk=len(self.brenk.GetMatches(mol)),
                    pains=int(self.pains.HasMatch(mol)))

    def reason(self, mol):
        """None if the molecule passes, else the name of the first failed filter. Also returns info."""
        i = self.info(mol)
        if self.use_pains and i['pains']:
            return 'pains', i
        if i['brenk'] > self.ref_brenk + self.brenk_extra:
            return 'brenk', i
        if self.sa and i['sa'] > self.ref_sa + self.sa_extra:
            return 'sa', i
        return None, i


# --------------------------------------------------------------- constraints ----
def _pair_index(name):
    a, b = name.split('_')
    for n in (f'{a}_{b}', f'{b}_{a}'):
        if n in E.PAIR_TYPES:
            return E.PAIR_TYPES.index(n)
    raise ValueError(f'unknown pair type {name}')


class Constraints:
    def __init__(self):
        self.rows, self.lo, self.hi, self.names = [], [], [], []

    def _add(self, w, lo, hi, name):
        self.rows.append(w)
        self.lo.append(-np.inf if lo is None else float(lo))
        self.hi.append(np.inf if hi is None else float(hi))
        self.names.append(name)

    def feature_count(self, feat, lo=None, hi=None):
        w = np.zeros(6 + 21 * NB, dtype=np.float32)
        w[E.FEATURES.index(feat)] = 1
        self._add(w, lo, hi, f'count_{feat}')

    def pair_count(self, pair, dmin=1, dmax=NB, lo=None, hi=None):
        w = np.zeros(6 + 21 * NB, dtype=np.float32)
        p = _pair_index(pair)
        w[6 + p * NB + dmin - 1: 6 + p * NB + dmax] = 1
        self._add(w, lo, hi, f'{pair}_d{dmin}-{dmax}')

    def max_distance(self, D):
        if D >= NB:
            return
        w = np.zeros(6 + 21 * NB, dtype=np.float32)
        for p in range(21):
            w[6 + p * NB + D: 6 + (p + 1) * NB] = 1
        self._add(w, None, 0, f'max_distance<={D}')

    @classmethod
    def from_json(cls, path, base=None):
        c = base or cls()
        for s in json.load(open(path)):
            t = s['type']
            if t == 'feature_count':
                c.feature_count(s['feature'], s.get('min'), s.get('max'))
            elif t == 'pair_count':
                c.pair_count(s['pair'], s.get('dmin', 1), s.get('dmax', NB), s.get('min'), s.get('max'))
            elif t == 'max_distance':
                c.max_distance(int(s['value']))
            else:
                raise ValueError(t)
        return c

    @classmethod
    def from_reference(cls, ref_raw, slack_pharm=1, slack_hy=1, extra_distance=1):
        """Envelope around the reference: charge counts fixed, HA/HD/AR lower-bounded at ~70 %,
        all counts upper-bounded at reference + slack, and the cavity 'length' (largest topological
        site-site distance) capped at reference + extra_distance."""
        c = cls()
        cnt = ref_raw[:6]
        for f, n in zip(E.FEATURES, cnt):
            n = int(n)
            if f in ('PLUS', 'MINUS'):
                c.feature_count(f, n, n)
            elif f == 'HY':
                c.feature_count(f, None, n + slack_hy)
            else:
                c.feature_count(f, int(np.ceil(0.7 * n)), n + slack_pharm)
        h = ref_raw[6:].reshape(21, NB).sum(0)
        nz = np.nonzero(h)[0]
        if len(nz) and nz[-1] + 1 < NB:
            c.max_distance(min(int(nz[-1] + 1) + extra_distance, NB - 1))
        return c

    def finalize(self):
        self.W = np.array(self.rows, dtype=np.float32)
        self.lo, self.hi = np.array(self.lo), np.array(self.hi)
        self.has_lo = np.isfinite(self.lo) & (self.lo > 0)
        return self

    def check(self, V):
        """V: (n,321) raw counts -> (violates_upper[n] bool, lower_deficit[n] in [0,1], unmet_lower[n] bool)."""
        S = V @ self.W.T
        up = (S > self.hi[None, :] + 1e-4).any(1)
        if self.has_lo.any():
            lo = self.lo[self.has_lo]
            d = np.maximum(0.0, lo[None, :] - S[:, self.has_lo]) / lo[None, :]
            return up, d.mean(1), (d > 1e-6).any(1)
        z = np.zeros(len(V))
        return up, z, z.astype(bool)

    def describe(self):
        return [f'{n}: [{l if np.isfinite(l) else "-"}, {h if np.isfinite(h) else "-"}]'
                for n, l, h in zip(self.names, self.lo, self.hi)]
