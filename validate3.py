"""Exactness check for ErG v3: frag_erg3.fragment_erg(m) == erg_v3.erg_raw(m) on random ChEMBL / PubChem molecules (as given and neutralised)."""
import paths
import sys, collections
import numpy as np, pandas as pd
from rdkit import Chem
import frag_erg3 as F
import erg_v3 as E
import retro as R

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
NEUT = len(sys.argv) > 2 and sys.argv[2] == 'neutral'
chembl = pd.read_parquet(paths.data('ALL_chembl_ErG.parquet'), columns=['SMILES']).SMILES.sample(N, random_state=7)
pub = pd.read_parquet(paths.pubchem_idx() + '/shard_00005.ids.parquet', columns=['smiles']).smiles.sample(N, random_state=7)
for name, smis in (('ChEMBL', chembl), ('PubChem', pub)):
    ncut = collections.Counter(); bad = []; n = 0
    for s in smis:
        m = Chem.MolFromSmiles(s) if not NEUT else R.neutral(s)
        if m is None or m.GetNumAtoms() > 80:
            continue
        n += 1
        try:
            a = F.fragment_erg(m)
        except Exception as e:
            bad.append((s, repr(e))); continue
        ncut[len(F.site_aware_cuts(m))] += 1
        b = E.erg_raw(m).astype(int)
        if not (np.minimum(a, 255) == b).all():
            bad.append((s, 'mismatch'))
    print(f"{name}: {n} mols, {n - len(bad)} exact, {len(bad)} failed; cuts/mol histogram {dict(sorted(ncut.items()))}")
    for s, why in bad[:8]:
        print('   ', why, s)
