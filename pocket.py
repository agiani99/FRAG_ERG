"""
Excluded-volume check against a protein pocket (information no 2D fingerprint carries).

A grown molecule keeps the anchor atoms of the reference ligand. The anchor atoms are fixed at their crystal
coordinates (taken from the complex); the molecule is embedded in 3D with that constraint (many conformers), and
every non-anchor heavy atom is tested against the protein heavy atoms:

    clash:  distance < r_vdW(i) + r_vdW(j) - tol,   tol = 1.0 A if both atoms are N/O (hydrogen bonds), else 0.5 A
    clash   = number of non-anchor ligand heavy atoms with >= 1 clash       (min over conformers)
    overlap = summed penetration depth (A) of all clashing pairs            (tie-break, min over conformers)
    contacts= protein heavy atoms within 4.5 A of non-anchor ligand atoms   (max over the conformers with min clash)

Limits: no protein flexibility, no waters, no scoring of interactions; conformers are ETKDG distance-geometry
conformers with the anchor constrained (no minimisation in the field), so the check is an approximation of
'there is room for this molecule in this pocket', not a docking score.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures, RDConfig, rdMolTransforms
from rdkit.Geometry import Point3D
from scipy.spatial import cKDTree

VDW = {'H': 1.10, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85,
       'I': 1.98, 'ZN': 1.39, 'Zn': 1.39}
_NO = {'N', 'O'}


def _el(line):
    e = line[76:78].strip()
    if not e:
        e = ''.join(ch for ch in line[12:16] if ch.isalpha())[:1]
    return e.capitalize() if len(e) > 1 else e.upper()


_FACT = None


def _factory():
    global _FACT
    if _FACT is None:
        import os
        _FACT = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    return _FACT


def ligand_roles(mol):
    """{heavy atom index: (is_donor, is_acceptor)} for the N and O atoms of a ligand. Donor = N/O carrying at least one
    hydrogen (RDKit's base donor definition also lists tertiary amines, which cannot donate); acceptor = RDKit base
    feature definitions (excludes amide-type N, includes sp2 ring N, nitrile N, carbonyl / ether / hydroxyl O...)."""
    A = set()
    for f in _factory().GetFeaturesForMol(mol):
        if f.GetFamily() == 'Acceptor':
            A.update(f.GetAtomIds())
    return {a.GetIdx(): (a.GetTotalNumHs() > 0, a.GetIdx() in A) for a in mol.GetAtoms()
            if a.GetSymbol() in ('N', 'O')}


def _sphere(n=96):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


_SPH = _sphere()
PROBE = 1.4
CONTACT_CAP = 6.0     # contacts per new atom that count as fully 'in contact'


def angle(a, b, c):
    """Angle a-b-c in degrees (vertex b)."""
    u, v = a - b, c - b
    cs = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cs, -1.0, 1.0))))


HB_H_A = 2.6        # H...acceptor distance limit (A)
HB_DHA = 120.0      # donor-H...acceptor angle minimum
HB_XAH = 90.0       # X-acceptor...H angle minimum (X = atom bonded to the acceptor)
_ROT = np.radians(np.arange(0, 360, 30))


def kabsch(P, Q):
    """Rotation R and translation t so that R @ P_i + t ~ Q_i (least squares)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


class Pocket:
    def __init__(self, pdb, sdf, lig_name, radius=14.0, n_conf=20, seed=7, core_rmsd=0.6, stage2_confs=150,
                 accept=2, hopeless=25, drop_het=(), overlap=2.5):
        recs = list(Chem.SDMolSupplier(sdf, removeHs=False))
        bad = [i for i, m in enumerate(recs) if m is None]
        if bad:
            raise ValueError(f'{sdf}: record(s) {bad} cannot be read by RDKit (see the RDKit message above; usually wrong bond '
                             f'orders written by a converter such as Open Babel). Repair the pose with fix_docked_sdf.py, which '
                             f'takes the bond orders from a template SMILES and keeps the coordinates.')
        names = [m.GetProp('_Name') if m.HasProp('_Name') else '' for m in recs]
        if lig_name not in names:
            raise ValueError(f'{sdf}: no record named {lig_name!r} (pocket.py looks the ligand up by _Name, and the part before the '
                             f'first underscore is also the residue name removed from the pocket atoms of the PDB); records: {names}')
        lig = recs[names.index(lig_name)]
        self.lig_heavy = Chem.RemoveHs(lig)
        self.lig_xyz = self.lig_heavy.GetConformer().GetPositions()
        resname = lig_name.split('_')[0]
        els, xyz, rn, an, rk, het = [], [], [], [], [], []
        for l in open(pdb):
            if l[:6] not in ('ATOM  ', 'HETATM'):
                continue
            if l[17:20].strip() in ('HOH', 'WAT', resname):
                continue
            els.append(_el(l))
            xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
            rn.append(l[17:20].strip())
            an.append(l[12:16].strip())
            rk.append(f'{l[17:20].strip()}_{l[21]}_{l[22:26].strip()}')
            het.append(l[:6] == 'HETATM')
        els, xyz, rn, an = np.array(els), np.array(xyz), np.array(rn), np.array(an)
        rk, het = np.array(rk), np.array(het)
        # a docked ligand shares its site with whatever was crystallised there: drop every non-water HETATM residue of
        # two or more heavy atoms that overlaps the ligand (or is named in drop_het); single-atom ions are kept
        self.dropped = []
        lig_tree = cKDTree(self.lig_xyz)
        drop = np.zeros(len(els), dtype=bool)
        for key in np.unique(rk[het]):
            sel = rk == key
            heavy_sel = sel & (els != 'H')
            if key.split('_')[0] in set(drop_het):
                why = 'listed'
            elif heavy_sel.sum() >= 2 and lig_tree.query(xyz[heavy_sel])[0].min() < overlap:
                why = f'overlaps the ligand ({lig_tree.query(xyz[heavy_sel])[0].min():.1f} A)'
            else:
                continue
            drop |= sel
            self.dropped.append(f'{key} ({why})')
        els, xyz, rn, an, rk = els[~drop], xyz[~drop], rn[~drop], an[~drop], rk[~drop]
        hmask = els == 'H'
        # donors: N/O carrying a hydrogen (Protoss hydrogens are in the file); acceptors: every O, and the ring N of
        # histidine without hydrogen
        Hxyz = xyz[hmask]
        htree = cKDTree(Hxyz) if hmask.any() else None
        heavy = np.nonzero(~hmask)[0]
        hasH = np.zeros(len(els), dtype=bool)
        hl = [[] for _ in heavy]
        if htree is not None:
            hl = htree.query_ball_point(xyz[heavy], 1.25)
            hasH[heavy] = np.array([len(x) > 0 for x in hl])
        isD_all = hasH & np.isin(els, ['N', 'O'])
        isA_all = (els == 'O') | ((els == 'N') & ~hasH & np.isin(rn, ['HIS', 'HID', 'HIE', 'HSD', 'HSE']) &
                                  np.isin(an, ['ND1', 'NE2']))
        els, xyz, rn, an, isD_all, isA_all, rk = (v[~hmask] for v in (els, xyz, rn, an, isD_all, isA_all, rk))
        full_tree = cKDTree(xyz)
        near = np.unique(np.concatenate(full_tree.query_ball_point(self.lig_xyz, radius)))
        self.els, self.xyz = els[near], xyz[near]
        self.plabel = np.array([f'{r}:{n}' for r, n in zip(rk[near], an[near])])      # e.g. HIS_B_378:O
        self.isD, self.isA = isD_all[near], isA_all[near]
        # hydrogens of the protein donors (Protoss positions) and covalent neighbours of the polar atoms
        self.hpos, self.pnbr = [], []
        for k in near:
            self.hpos.append(Hxyz[hl[k]] if (isD_all[k] and len(hl[k])) else np.zeros((0, 3)))
            if isD_all[k] or isA_all[k]:
                nb = [m for m in full_tree.query_ball_point(xyz[k], 1.85) if m != k]
                self.pnbr.append(xyz[nb] if nb else np.zeros((0, 3)))
            else:
                self.pnbr.append(np.zeros((0, 3)))
        self.tree = cKDTree(self.xyz)
        self.rad = np.array([VDW.get(e, 1.7) for e in self.els])
        self.isno = np.array([e in _NO for e in self.els])
        self.n_conf, self.seed, self.core_rmsd = n_conf, seed, core_rmsd
        self.stage2_confs, self.accept, self.hopeless = stage2_confs, accept, hopeless
        self.ref_xyz = None
        self.score_poses = True
        self.w_polar = 0.3
        self.scan_confs, self.scan_step = 3, 30

    def __repr__(self):
        d = f'; dropped co-crystallised ligand(s): {", ".join(self.dropped)}' if self.dropped else ''
        return f'Pocket({len(self.xyz)} protein heavy atoms within the cut-off{d})'


    def clearances(self, ref, probe=1.5):
        """{ref atom index: distance (A) from a point `probe` A beyond the atom (away from its heavy neighbours) to the
        nearest protein heavy atom}: how much room a substituent on that atom has in the crystal pose."""
        out = {}
        for i, x in self.ref_xyz.items():
            nb = [self.ref_xyz[n.GetIdx()] for n in ref.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in self.ref_xyz]
            if not nb:
                continue
            d = x - np.mean(nb, axis=0)
            if np.linalg.norm(d) < 1e-3:
                continue
            out[i] = float(self.tree.query(x + probe * d / np.linalg.norm(d))[0])
        return out

    def set_reference(self, ref):
        """Map the crystal / docked ligand coordinates onto the atom indices of `ref` (a mol from SMILES). The match
        ignores formal charges and hydrogen counts (a docked ligand is often protonated, the reference neutral) and,
        as a fallback, bond orders."""
        q = Chem.RWMol(self.lig_heavy)
        for a in q.GetAtoms():
            a.SetFormalCharge(0)
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(False)
            a.SetNumRadicalElectrons(0)
        m = ref.GetSubstructMatch(q)
        if not m:
            prm = Chem.AdjustQueryParameters.NoAdjustments()
            prm.makeBondsGeneric = True
            m = ref.GetSubstructMatch(Chem.AdjustQueryProperties(q, prm))
        if not m:
            raise ValueError('reference does not match the ligand in the complex: the ligand is '
                             f'{Chem.MolToSmiles(self.lig_heavy)} ({self.lig_heavy.GetNumHeavyAtoms()} heavy atoms), '
                             f'the reference is {Chem.MolToSmiles(ref)} ({ref.GetNumHeavyAtoms()} heavy atoms). '
                             'Pass the SMILES of the ligand in the SDF (--smiles), or a reference that contains it '
                             'with the same heavy-atom graph (charges and hydrogens are ignored).')
        self.ref_xyz = {m[i]: self.lig_xyz[i] for i in range(len(m))}

    def _clash(self, pos, el):
        """clash atoms, overlap, contacts for ligand atoms `pos` with elements `el`."""
        nclash, overlap, contacts = 0, 0.0, set()
        for p, e in zip(pos, el):
            r = VDW.get(e, 1.7)
            idx = self.tree.query_ball_point(p, 4.5)
            hit = False
            for j in idx:
                d = np.linalg.norm(self.xyz[j] - p)
                contacts.add(j)
                tol = 1.0 if (e in _NO and self.isno[j]) else 0.5
                cut = r + self.rad[j] - tol
                if d < cut:
                    hit = True
                    overlap += cut - d
            nclash += hit
        return nclash, overlap, len(contacts)


    def sasa(self, X, el, atoms, with_protein):
        """Shrake-Rupley solvent accessible area (A^2) of the ligand heavy atoms `atoms` (indices into X),
        occluded by the other ligand atoms and, if with_protein, by the pocket's protein atoms."""
        rad = np.array([VDW.get(e, 1.7) for e in el])
        out = np.zeros(len(atoms))
        for q, i in enumerate(atoms):
            R = rad[i] + PROBE
            P = X[i] + R * _SPH
            free = np.ones(len(P), dtype=bool)
            others = [j for j in range(len(X)) if j != i and np.linalg.norm(X[j] - X[i]) < R + rad[j] + PROBE]
            if others:
                d = np.linalg.norm(P[:, None, :] - X[others][None, :, :], axis=2)
                free &= (d >= rad[others] + PROBE).all(1)
            if with_protein:
                nb = self.tree.query_ball_point(X[i], R + 1.9 + PROBE)
                if nb:
                    d = np.linalg.norm(P[:, None, :] - self.xyz[nb][None, :, :], axis=2)
                    free &= (d >= self.rad[nb] + PROBE).all(1)
            out[q] = 4 * np.pi * R * R * free.mean()
        return out

    def _ligand_hs(self, X, i, hinfo):
        """Candidate hydrogen positions of ligand donor i. Terminal sp3 donors (OH, NH2, NH3+ on sp3 C) rotate freely:
        12 idealised rotamers (X-D-H 109.5 deg, D-H 1.0 A); other donors use the hydrogens of the conformer."""
        if i in hinfo['rot']:
            x1 = X[hinfo['nbr'][i][0]]
            u = X[i] - x1
            u /= np.linalg.norm(u)
            a = np.cross(u, [1.0, 0.0, 0.0])
            if np.linalg.norm(a) < 1e-3:
                a = np.cross(u, [0.0, 1.0, 0.0])
            e1 = a / np.linalg.norm(a)
            e2 = np.cross(u, e1)
            c, s = np.cos(np.radians(70.5)), np.sin(np.radians(70.5))
            return [X[i] + c * u + s * (np.cos(f) * e1 + np.sin(f) * e2) for f in _ROT]
        return [X[h] for h in hinfo['H'].get(i, [])]

    def _hbond(self, X, i, j, D, A, hinfo):
        """Hydrogen bond between ligand atom i (roles D, A) and protein atom j with the angle criteria."""
        pj = self.xyz[j]
        if D and self.isA[j]:                                   # ligand donor -> protein acceptor
            nbj = self.pnbr[j]
            for h in self._ligand_hs(X, i, hinfo):
                if (np.linalg.norm(h - pj) <= HB_H_A and angle(X[i], h, pj) >= HB_DHA
                        and all(angle(nb, pj, h) >= HB_XAH for nb in nbj)):
                    return True
        if A and self.isD[j]:                                   # ligand acceptor <- protein donor
            nbi = [X[m] for m in hinfo['nbr'].get(i, [])]
            for h in self.hpos[j]:
                if (np.linalg.norm(h - X[i]) <= HB_H_A and angle(pj, h, X[i]) >= HB_DHA
                        and all(angle(nb, X[i], h) >= HB_XAH for nb in nbi)):
                    return True
        return False

    def polar(self, X, el, idx, roles, burial_i, hinfo):
        """Polar complementarity of the ligand N/O atoms `idx` (X: coordinates of all ligand atoms incl. hydrogens).
        satisfied  : a hydrogen bond with the protein: complementary partner at 2.4-3.5 A (heavy atoms) AND
                     H...A <= 2.6 A, D-H...A >= 120 deg, X-A...H >= 90 deg
        strained   : a complementary partner at 2.4-3.5 A whose geometry fails (neutral: the conformers are coarse)
        repulsive  : no partner but an acceptor-only (donor-only) protein atom of the same kind within 3.2 A
        unsatisfied: none of the above and the atom is >= 80 % buried
        Returns (n_polar, n_sat, n_strained, n_unsat, n_rep). No waters."""
        n_pol = n_sat = n_str = n_unsat = n_rep = 0
        for q, i in enumerate(idx):
            if i not in roles:
                continue
            D, A = roles[i]
            if not (D or A):
                continue
            n_pol += 1
            sat = strained = rep = False
            for j in self.tree.query_ball_point(X[i], 3.5):
                if not (self.isD[j] or self.isA[j]):
                    continue
                d = np.linalg.norm(self.xyz[j] - X[i])
                if d >= 2.4 and ((D and self.isA[j]) or (A and self.isD[j])):
                    if self._hbond(X, i, j, D, A, hinfo):
                        sat = True
                    else:
                        strained = True
                elif d < 3.2 and ((A and not D and self.isA[j] and not self.isD[j]) or
                                  (D and not A and self.isD[j] and not self.isA[j])):
                    rep = True
            if sat:
                n_sat += 1
            elif strained:
                n_str += 1
            elif rep:
                n_rep += 1
            elif burial_i[q] >= 0.8:
                n_unsat += 1
        return n_pol, n_sat, n_str, n_unsat, n_rep

    def score3d(self, X, el, new, n_contacts, roles=None, hinfo=None):
        """Burial, contact and polar terms for the `new` (non-anchor) heavy atoms of a placed molecule.
        X: coordinates of ALL atoms of the placed molecule (heavy atoms first, then hydrogens).
        burial  = 1 - SASA(complex) / SASA(ligand alone), area weighted over the new atoms           (0..1)
        cdens   = protein heavy atoms within 4.5 A of the new atoms per new atom, capped at CONTACT_CAP (0..1)
        pol01   = 0.5 * (1 + (n_sat - n_unsat - n_rep) / n_polar) over the new N/O atoms, 0.5 if there are none
                  (strained contacts count 0)
        s3d     = (1 - w_polar) * (0.5 * burial + 0.5 * cdens) + w_polar * pol01
        Returns dict(burial, cdens, pol01, n_pol, n_sat, n_str, n_unsat, n_rep, s3d)."""
        out = dict(burial=0.0, cdens=0.0, pol01=0.5, n_pol=0, n_sat=0, n_str=0, n_unsat=0, n_rep=0, s3d=0.0)
        if not new:
            return out
        Xh = X[:len(el)]
        free = self.sasa(Xh, el, new, False)
        comp = self.sasa(Xh, el, new, True)
        out['burial'] = float(1.0 - comp.sum() / free.sum()) if free.sum() > 0 else 0.0
        out['cdens'] = min(n_contacts / len(new) / CONTACT_CAP, 1.0)
        base = 0.5 * out['burial'] + 0.5 * out['cdens']
        if roles and hinfo is not None:
            bi = np.where(free > 0, 1.0 - comp / np.maximum(free, 1e-9), 0.0)
            n_pol, n_sat, n_str, n_unsat, n_rep = self.polar(X, el, new, roles, bi, hinfo)
            out.update(n_pol=n_pol, n_sat=n_sat, n_str=n_str, n_unsat=n_unsat, n_rep=n_rep)
            if n_pol:
                out['pol01'] = 0.5 * (1.0 + (n_sat - n_unsat - n_rep) / n_pol)
        out['s3d'] = (1 - self.w_polar) * base + self.w_polar * out['pol01']
        return out

    @staticmethod
    def hinfo(mol, mh):
        """Hydrogen / neighbour bookkeeping of the ligand N and O atoms (indices of the heavy atoms of mol)."""
        n = mol.GetNumAtoms()
        H, nbr, rot = {}, {}, set()
        for a in mh.GetAtoms():
            i = a.GetIdx()
            if i < n and a.GetAtomicNum() in (7, 8):
                H[i] = [x.GetIdx() for x in a.GetNeighbors() if x.GetAtomicNum() == 1]
                nbr[i] = [x.GetIdx() for x in a.GetNeighbors() if x.GetAtomicNum() > 1]
                if H[i] and len(nbr[i]) == 1 and (
                        a.GetAtomicNum() == 8 or
                        mol.GetAtomWithIdx(nbr[i][0]).GetHybridization() == Chem.HybridizationType.SP3):
                    rot.add(i)        # OH (any neighbour) and N-H on sp3 C rotate; aniline NH2 stays planar
        return dict(H=H, nbr=nbr, rot=rot)

    def evaluate(self, mol, amap=None):
        """Two stages: n_conf conformers; if the best has more than `accept` clashing atoms but not more than
        `hopeless`, redo with stage2_confs conformers (flexible groups need sampling)."""
        r = self._evaluate(mol, amap, self.n_conf)
        if self.accept < r['clash'] <= self.hopeless and self.stage2_confs > self.n_conf:
            r2 = self._evaluate(mol, amap, self.stage2_confs)
            if (r2['clash'], r2['overlap']) < (r['clash'], r['overlap']):
                r = r2
        return r

    def _evaluate(self, mol, amap, n_conf):
        """mol: heavy-atom molecule carrying the prop 'anchor_map' ('new_idx:ref_idx;...') or amap=[(i, j)]."""
        if self.ref_xyz is None:
            raise RuntimeError('call set_reference first')
        if amap is None:
            amap = [tuple(int(x) for x in t.split(':')) for t in mol.GetProp('anchor_map').split(';') if t]
        core = [(i, j) for i, j in amap if j in self.ref_xyz]
        if len(core) < 3:
            return dict(clash=99, overlap=99.0, contacts=0, confs=0)
        mh = Chem.AddHs(mol)
        cmap = {i: Point3D(*self.ref_xyz[j]) for i, j in core}
        try:
            cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=n_conf, randomSeed=self.seed, coordMap=cmap,
                                                   useRandomCoords=True, maxAttempts=5))
        except Exception:
            cids = []
        if not cids:
            return dict(clash=99, overlap=99.0, contacts=0, confs=0)
        ci = [i for i, _ in core]
        Q = np.array([self.ref_xyz[j] for _, j in core])
        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in set(ci)]
        el = [mol.GetAtomWithIdx(i).GetSymbol() for i in heavy]
        best, nok = None, 0
        for cid in cids:
            X = mh.GetConformer(cid).GetPositions()
            R, t = kabsch(X[ci], Q)
            Y = X @ R.T + t
            if np.sqrt(((Y[ci] - Q) ** 2).sum(1).mean()) > self.core_rmsd:
                continue
            nok += 1
            c, o, k = self._clash(Y[heavy], el)
            key = (c, o, -k)
            if best is None or key < best[0]:
                best = (key, c, o, k)
        if best is None:
            return dict(clash=99, overlap=99.0, contacts=0, confs=0)
        return dict(clash=best[1], overlap=round(best[2], 2), contacts=best[3], confs=nok)


    # ------------------------------------------------------------------ interaction retention ----
    def ref_hbonds(self, dmax=3.4):
        """Hydrogen bonds of the crystal ligand: [(ligand atom index in the SDF molecule, role 'D' / 'A', protein atom index,
        distance)]. A ligand donor (N/O with H) pairs with a protein acceptor and a ligand acceptor with a protein donor within
        dmax (heavy-atom distance, A); a ligand atom that is both (hydroxyl) is listed for both roles."""
        roles = ligand_roles(self.lig_heavy)
        out = []
        for i, (isd, isa) in roles.items():
            for j in self.tree.query_ball_point(self.lig_xyz[i], dmax):
                d = float(np.linalg.norm(self.xyz[j] - self.lig_xyz[i]))
                if isd and self.isA[j]:
                    out.append((i, 'D', int(j), d))
                if isa and self.isD[j]:
                    out.append((i, 'A', int(j), d))
        return sorted(out, key=lambda t: (t[0], t[3]))

    def set_hb_check(self, ref_dmax=3.25, prod_dmax=3.7):
        """Switch on the interaction-retention check inside place(): the reference bonds are the ligand-protein hydrogen bonds of the
        crystal ligand within ref_dmax (strong ones); a placed molecule keeps one if it has a polar atom of that role within
        prod_dmax of the same protein atom (sampled poses reproduce the crystal geometry only approximately, hence the larger
        tolerance). place() then reports, among the clash-free candidate poses, the one that keeps the most bonds."""
        self._hb = self.ref_hbonds(ref_dmax)
        self._hbd = prod_dmax

    def hbond_retention(self, mol, xyz, hbs=None, dmax=3.4, roles=None):
        """Which hydrogen bonds of the crystal ligand does a placed molecule still offer? mol: heavy-atom molecule, xyz: its
        coordinates (n_heavy, 3) in the pocket frame. A reference bond (role, protein atom j) is kept if the molecule has a
        heavy atom with that role within dmax of protein atom j. Returns (n_reference_bonds, n_kept, [labels of the lost ones]).
        Bonds are counted per (role, protein atom) pair, so one protein atom bonded by two ligand atoms counts once."""
        hbs = self.ref_hbonds(dmax) if hbs is None else hbs
        roles = ligand_roles(mol) if roles is None else roles
        xyz = np.asarray(xyz)
        need = sorted({(r, j) for _, r, j, _ in hbs})
        lost, kept = [], 0
        for r, j in need:
            ok = False
            for b, (isd, isa) in roles.items():
                if (r == 'D' and isd) or (r == 'A' and isa):
                    if np.linalg.norm(xyz[b] - self.xyz[j]) <= dmax:
                        ok = True
                        break
            kept += ok
            if not ok:
                lost.append(f"{self.plabel[j]}({'lig donor' if r == 'D' else 'lig acceptor'})")
        return len(need), kept, lost

    # ------------------------------------------------------------------ used inside the search ----
    def anchor_info(self, mol):
        """(anchor atom index set, {idx: crystal xyz}) from the 'anchor_map' prop of an assembled molecule."""
        amap = [tuple(int(x) for x in t.split(':')) for t in mol.GetProp('anchor_map').split(';') if t]
        fixed = {i: self.ref_xyz[j] for i, j in amap if j in self.ref_xyz}
        return set(fixed), fixed

    @staticmethod
    def _spread_subset(fixed, k):
        idx = list(fixed)
        P = np.array([fixed[i] for i in idx])
        chosen = [int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))]
        while len(chosen) < min(k, len(idx)):
            d = np.min(np.linalg.norm(P[:, None, :] - P[chosen][None, :, :], axis=2), axis=1)
            chosen.append(int(np.argmax(d)))
        return [idx[c] for c in chosen]

    def place(self, mol, fixed, anchor, n_conf, seed=None):
        """Embed `mol` (heavy atoms) with the atoms in `fixed` {idx: xyz} constrained, fit each conformer rigidly onto
        them, and count clashes of all atoms not in `anchor`. Returns dict(clash, overlap, contacts, confs, xyz) with
        xyz = (n_heavy, 3) coordinates of the best conformer, or xyz = None if nothing could be embedded."""
        bad = dict(clash=99, overlap=99.0, contacts=0, confs=0, xyz=None, burial=0.0, cdens=0.0, pol01=0.5,
                   n_pol=0, n_sat=0, n_str=0, n_unsat=0, n_rep=0, s3d=0.0)
        if len(fixed) < 3:
            return bad
        mh = Chem.AddHs(mol)
        cmap = {i: Point3D(*x) for i, x in fixed.items()}
        sd = self.seed if seed is None else seed
        try:
            cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=n_conf, randomSeed=sd, coordMap=cmap,
                                                   useRandomCoords=True, maxAttempts=5))
        except Exception:
            cids = []
        if not cids and len(fixed) > 4:
            # pinning many atoms can be infeasible for the embedder (crystal ring geometry outside its bounds):
            # pin a spread-out subset of 4, the fit and the RMSD test below still use all fixed atoms
            sub = self._spread_subset(fixed, 4)
            try:
                cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=max(n_conf, 30), randomSeed=sd,
                                                       coordMap={i: cmap[i] for i in sub}, useRandomCoords=True,
                                                       maxAttempts=5))
            except Exception:
                cids = []
        if not cids:
            return bad
        fi = list(fixed)
        Q = np.array([fixed[i] for i in fi])
        n = mol.GetNumAtoms()
        count = [i for i in range(n) if i not in anchor]
        el = [mol.GetAtomWithIdx(i).GetSymbol() for i in count]
        el_all = [a.GetSymbol() for a in mol.GetAtoms()]
        cand, nok = [], 0
        for cid in cids:
            X = mh.GetConformer(cid).GetPositions()
            R, t = kabsch(X[fi], Q)
            Y = X @ R.T + t
            if np.sqrt(((Y[fi] - Q) ** 2).sum(1).mean()) > self.core_rmsd:
                continue
            nok += 1
            c, o, k = self._clash(Y[count], el)
            cand.append((c, o, k, Y.copy(), cid, R, t))
        if not cand:
            return bad
        roles = ligand_roles(mol)
        hinfo = self.hinfo(mol, mh)
        if self.score_poses and self.scan_confs > 0:
            cand = cand + self._scan(mh, mol, fixed, count, el, el_all, roles, hinfo, cand)
        none = dict(burial=0.0, cdens=0.0, pol01=0.5, n_pol=0, n_sat=0, n_str=0, n_unsat=0, n_rep=0, s3d=0.0)
        feasible = sorted((x for x in cand if x[0] <= self.accept), key=lambda x: x[1])[:30]
        if feasible and self.score_poses:
            scored = [(self.score3d(x[3], el_all, count, x[2], roles, hinfo), x) for x in feasible]
            sc, best = max(scored, key=lambda z: (z[0]['s3d'], -z[1][1]))
        else:
            best = min(cand, key=lambda x: (x[0], x[1], -x[2]))
            sc = self.score3d(best[3], el_all, count, best[2], roles, hinfo) if self.score_poses else none
        res = dict(clash=best[0], overlap=round(best[1], 2), contacts=best[2], confs=nok, xyz=best[3][:n],
                   burial=round(sc['burial'], 3), cdens=round(sc['cdens'], 3), pol01=round(sc['pol01'], 3),
                   n_pol=sc['n_pol'], n_sat=sc['n_sat'], n_str=sc['n_str'], n_unsat=sc['n_unsat'],
                   n_rep=sc['n_rep'], s3d=round(sc['s3d'], 3))
        if getattr(self, '_hb', None) is not None:
            pool = [x for x in cand if x[0] <= self.accept] or [best]
            hb = max((self.hbond_retention(mol, x[3][:n], self._hb, self._hbd, roles) + (-x[0],) for x in pool),
                     key=lambda t: (t[1], t[3]))
            res['hb_ref'], res['hb_kept'], res['hb_lost'] = hb[0], hb[1], '; '.join(hb[2])
        return res

    def _scan(self, mh, mol, fixed, count, el, el_all, roles, hinfo, cand):
        """Systematic torsion scan of the bond(s) joining the new atoms to the fixed atoms (the sampled conformers are
        too coarse to find hydrogen-bonding orientations by chance): the best `scan_confs` conformers (by clash,
        overlap), 30-degree steps, scored with the 3D score; returns the new candidates."""
        n = mol.GetNumAtoms()
        bonds = []
        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if (u in fixed) == (v in fixed) or b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
                continue
            if v in fixed:
                u, v = v, u
            x = [m.GetIdx() for m in mol.GetAtomWithIdx(u).GetNeighbors() if m.GetIdx() != v]
            y = [m.GetIdx() for m in mol.GetAtomWithIdx(v).GetNeighbors() if m.GetIdx() != u]
            if x and y:
                bonds.append((x[0], u, v, y[0]))
        if not bonds:
            return []
        out = []
        for c0, o0, k0, Y0, cid, R, t in sorted(cand, key=lambda z: (z[0], z[1]))[:self.scan_confs]:
            mt = Chem.Mol(mh)
            mt.RemoveAllConformers()
            mt.AddConformer(Chem.Conformer(mh.GetConformer(cid)), assignId=True)
            conf = mt.GetConformer()
            for (x, u, v, y) in bonds:
                best = None
                for deg in np.arange(0, 360, self.scan_step):
                    try:
                        rdMolTransforms.SetDihedralDeg(conf, x, u, v, y, float(deg))
                    except Exception:
                        break
                    Y = conf.GetPositions() @ R.T + t
                    c, o, k = self._clash(Y[count], el)
                    sc = self.score3d(Y, el_all, count, k, roles, hinfo)['s3d'] if c <= self.accept else -1.0
                    key = (c <= self.accept, sc, -o)
                    if best is None or key > best[0]:
                        best = (key, deg, (c, o, k, Y.copy(), cid, R, t))
                if best is not None:
                    rdMolTransforms.SetDihedralDeg(conf, x, u, v, y, float(best[1]))
                    out.append(best[2])
        return out

    def search_place(self, mol, parent_fixed, anchor_fixed, anchor, n1=10, n2=60):
        """Pose of a (partial) molecule inside the search. First the parent's atoms are kept fixed and only the new
        block is sampled (n1 conformers, n2 if the first result is borderline); if that does not give an acceptable
        pose, fall back to sampling from the anchor alone (n_conf, stage2_confs). Returns the best result."""
        results = []
        if parent_fixed and len(parent_fixed) > len(anchor_fixed):
            r = self.place(mol, parent_fixed, anchor, n1)
            results.append(r)
            if r['xyz'] is not None and self.accept < r['clash'] <= self.hopeless:
                results.append(self.place(mol, parent_fixed, anchor, n2, seed=self.seed + 1))
            if any(x['xyz'] is not None and x['clash'] <= self.accept for x in results):
                return self._pick(results)
        r = self.place(mol, anchor_fixed, anchor, self.n_conf)
        results.append(r)
        if r['xyz'] is not None and self.accept < r['clash'] <= self.hopeless:
            results.append(self.place(mol, anchor_fixed, anchor, self.stage2_confs, seed=self.seed + 2))
        return self._pick(results)

    def _pick(self, results):
        ok = [x for x in results if x['xyz'] is not None]
        if not ok:
            return results[-1]
        feas = [x for x in ok if x['clash'] <= self.accept]
        if feas:
            return max(feas, key=lambda x: (x['s3d'], -x['overlap']))
        return min(ok, key=lambda x: (x['clash'], x['overlap']))
