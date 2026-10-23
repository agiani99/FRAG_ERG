"""
Repair a docked pose whose bond orders were lost in a conversion (typically Vina PDBQT -> SDF with Open Babel): RDKit then refuses
the record ('Explicit valence ... greater than permitted'), and pocket.py fails with 'NoneType has no attribute GetProp'.

The coordinates of the pose are kept. Bond orders, aromaticity and formal charges come from a template SMILES of the same molecule
(AllChem.AssignBondOrdersFromTemplate on the heavy atoms, then hydrogens are added with coordinates). The record gets the name
given by --name: pocket.py looks the ligand up by _Name and ALSO uses the part before the first '_' as the residue name that it
removes from the pocket atoms of the PDB (for 6mo0_mono.pdb: JVJ). The original name is kept in the property ligand_name.

usage: python fix_docked_sdf.py Complex/NVP-BHG712_docked.sdf --smiles "<SMILES>" --name JVJ --out Complex/NVP-BHG712_docked_fixed.sdf
The input file is never modified.
"""
import argparse, sys
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
ap = argparse.ArgumentParser()
ap.add_argument('sdf')
ap.add_argument('--smiles', required=True, help='template SMILES of the docked molecule (correct bond orders, no 3D needed)')
ap.add_argument('--name', default=None, help='_Name for the repaired record (default: keep the original name)')
ap.add_argument('--out', required=True)
ap.add_argument('--record', type=int, default=0, help='index of the record to repair (default 0)')
a = ap.parse_args()

mols = list(Chem.SDMolSupplier(a.sdf, sanitize=False, removeHs=False))
if a.record >= len(mols) or mols[a.record] is None:
    sys.exit(f'record {a.record} of {a.sdf} cannot be read even without sanitisation')
raw = mols[a.record]
orig_name = raw.GetProp('_Name') if raw.HasProp('_Name') else ''
tmpl = Chem.MolFromSmiles(a.smiles)
if tmpl is None:
    sys.exit('template SMILES cannot be parsed')

heavy = Chem.RemoveHs(raw, sanitize=False)               # polar hydrogens of the converter are dropped and re-added below
if heavy.GetNumHeavyAtoms() != tmpl.GetNumHeavyAtoms():
    sys.exit(f'heavy atom count differs: pose {heavy.GetNumHeavyAtoms()} vs template {tmpl.GetNumHeavyAtoms()}')
for b in heavy.GetBonds():                               # the template decides the bond orders; start from single bonds
    b.SetBondType(Chem.BondType.SINGLE); b.SetIsAromatic(False)
for at in heavy.GetAtoms():
    at.SetIsAromatic(False); at.SetNoImplicit(False); at.SetNumExplicitHs(0); at.SetFormalCharge(0)
Chem.SanitizeMol(heavy, Chem.SANITIZE_NONE)
fixed = AllChem.AssignBondOrdersFromTemplate(tmpl, heavy)   # raises ValueError if the graphs do not match
Chem.SanitizeMol(fixed)
fixed = Chem.AddHs(fixed, addCoords=True)

# the pose must not move: compare heavy-atom coordinates with the input
c_in = np.array([raw.GetConformer().GetAtomPosition(i) for i in range(raw.GetNumAtoms()) if raw.GetAtomWithIdx(i).GetAtomicNum() > 1])
c_out = np.array([fixed.GetConformer().GetAtomPosition(i) for i in range(fixed.GetNumAtoms()) if fixed.GetAtomWithIdx(i).GetAtomicNum() > 1])
shift = float(np.abs(c_in - c_out).max())
assert shift < 1e-3, f'coordinates moved by {shift:.4f} A'

fixed.SetProp('_Name', a.name if a.name else orig_name)
fixed.SetProp('ligand_name', orig_name)
for p in raw.GetPropNames():
    if p in ('docking_score',):
        fixed.SetProp(p, raw.GetProp(p))
w = Chem.SDWriter(a.out)
w.write(fixed)
w.close()

chk = next(iter(Chem.SDMolSupplier(a.out, removeHs=False)))
assert chk is not None
print(f'wrote {a.out}: name {chk.GetProp("_Name")!r} (original {orig_name!r}), {chk.GetNumHeavyAtoms()} heavy atoms, '
      f'SMILES {Chem.MolToSmiles(Chem.RemoveHs(chk))}')
print(f'heavy-atom coordinates unchanged (max shift {shift:.1e} A); bond orders from the template; hydrogens re-added with coordinates')
