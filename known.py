"""
'Is this molecule in PubChem?': a compact, protonation-independent lookup.

Key = the first block of the InChIKey (14 characters = hash of the main InChI layer: formula + connectivity + hydrogens
with charges / protonation moved to a separate layer, so a molecule and its protonated / deprotonated forms share it;
stereo and isotopes are not part of it). The index stores a 64-bit hash of that block for every row of the ErG index
(../pubchem_idx), sorted, with the global row number of each (-> id / SMILES through erg_index.Index.fetch):

    known_idx/keys.npy   uint64 [N] sorted        (N x 8 bytes, ~1 GB for 131M molecules)
    known_idx/rows.npy   int32  [N]               global row in pubchem_idx
    known_idx/info.json

Build (long, resumable per shard):  python build_known_index.py --idx ../pubchem_idx --out known_idx --workers 19
Why not the ErG vectors themselves: the index holds the CHARGED forms (99.8 % of rows carry a formal charge) and the ErG
vector of the neutral form of a molecule equals its indexed vector in 0 % of cases, so exact ErG matching cannot find
neutral candidates.
"""
import hashlib, json, os
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


def h64(key14):
    return int.from_bytes(hashlib.blake2b(key14.encode(), digest_size=8).digest(), 'little')


def mol_key(mol):
    """First InChIKey block of an RDKit molecule, or None."""
    try:
        k = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return k[:14] if k else None


def repair_charges(mol):
    """The charged collection writes protonated amide nitrogens as [NH+] (valence 3 + charge = a radical cation, chemically
    wrong; RDKit's Uncharger then leaves a radical [N]). Remove the charge of every N+ whose total valence is 3 (and add one H
    when it is 2); nitro (4),
    quaternary N (4) and pyridinium (4) are untouched. Returns a new molecule."""
    m = Chem.RWMol(mol)
    for a in m.GetAtoms():
        if a.GetFormalCharge() == 1 and a.GetAtomicNum() == 7 and not a.GetIsAromatic():
            v = a.GetTotalValence()
            if v == 3:                                  # [NH+](R)R' : radical cation -> neutral NH
                a.SetFormalCharge(0)
            elif v == 2:                                # [NH+]R : -> neutral NH2 (one hydrogen has to be added)
                a.SetFormalCharge(0)
                a.SetNumExplicitHs(a.GetNumExplicitHs() + 1)
    out = m.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return mol
    return out


def smiles_key(smi):
    """Key of a stored SMILES (largest component by string length: drops counter-ions)."""
    m = Chem.MolFromSmiles(max(smi.split('.'), key=len))
    return None if m is None else mol_key(repair_charges(m))


class KnownIndex:
    """Membership + row lookup. KnownIndex(path) loads a built index (memory-mapped); KnownIndex.from_smiles(list) builds a
    small in-memory one (tests)."""

    def __init__(self, path=None, keys=None, rows=None):
        if path is not None:
            self.keys = np.load(os.path.join(path, 'keys.npy'), mmap_mode='r')
            self.rows = np.load(os.path.join(path, 'rows.npy'), mmap_mode='r')
            self.info = json.load(open(os.path.join(path, 'info.json')))
        else:
            o = np.argsort(keys)
            self.keys, self.rows, self.info = keys[o], rows[o], {'n': len(keys), 'in_memory': True}
        self.cache = {}

    @classmethod
    def from_smiles(cls, smiles):
        ks, rs = [], []
        for i, s in enumerate(smiles):
            k = smiles_key(s)
            if k:
                ks.append(h64(k))
                rs.append(i)
        return cls(keys=np.array(ks, dtype=np.uint64), rows=np.array(rs, dtype=np.int32))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'keys.npy'), np.asarray(self.keys))
        np.save(os.path.join(path, 'rows.npy'), np.asarray(self.rows))
        json.dump(dict(self.info, saved=True), open(os.path.join(path, 'info.json'), 'w'))

    def __len__(self):
        return len(self.keys)

    def lookup(self, mol):
        """(known, global row of the first hit or -1, number of rows sharing the key)."""
        k = mol_key(mol)
        if k is None:
            return False, -1, 0
        r = self.cache.get(k)
        if r is None:
            h = np.uint64(h64(k))
            lo, hi = int(np.searchsorted(self.keys, h, 'left')), int(np.searchsorted(self.keys, h, 'right'))
            r = (hi > lo, int(self.rows[lo]) if hi > lo else -1, hi - lo)
            self.cache[k] = r
        return r
