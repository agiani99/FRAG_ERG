"""
Locations of the large input data, in one place.

Set the environment variable FRAGERG_DATA to a folder that contains

    pubchem_idx/                     ErG index of PubChem (erg_index.py build)
    ALL_chembl_ErG.parquet           ChEMBL activities with ErG (retro.py, validate.py)
    CHEMBL37_notarget.parquet        ChEMBL compounds without target annotation (decoy_eval.py)
    MAX_FDAapproved.parquet          approved drugs, columns Name, SMILES (rebuild_bench.py)
    MAX_clinical.parquet             clinical-stage compounds, columns Name, SMILES (rebuild_bench.py)

Windows (PowerShell):  $env:FRAGERG_DATA = "D:\\fragerg_data"        Linux / macOS:  export FRAGERG_DATA=/data/fragerg

If FRAGERG_DATA is not set the original author layout is used when it exists (index and ChEMBL file one folder above this
file, the drug sets under ../../PDB_LIGAND/ERG_SWITCHER/data), otherwise `<this folder>/data/<name>`. Every script that needs
one of these inputs also takes an explicit command-line option, which always wins.

Importing this module also puts this folder and its parent on sys.path, so erg_v2.py / erg_index.py are found whether they
sit next to the scripts (repository layout) or one folder up (author layout).
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, '..')):
    if _p not in sys.path:
        sys.path.append(_p)

_LEGACY = {
    'pubchem_idx': os.path.join(HERE, '..', 'pubchem_idx'),
    'ALL_chembl_ErG.parquet': os.path.join(HERE, '..', 'ALL_chembl_ErG.parquet'),
}
_LEGACY_DRUGS = os.path.join(HERE, '..', '..', 'PDB_LIGAND', 'ERG_SWITCHER', 'data')
for _n in ('CHEMBL37_notarget.parquet', 'MAX_FDAapproved.parquet', 'MAX_clinical.parquet'):
    _LEGACY[_n] = os.path.join(_LEGACY_DRUGS, _n)


def data(name):
    """Path of an input data file / folder (see the module docstring)."""
    root = os.environ.get('FRAGERG_DATA')
    if root:
        return os.path.join(root, name)
    leg = _LEGACY.get(name)
    if leg and os.path.exists(leg):
        return os.path.normpath(leg)
    return os.path.join(HERE, 'data', name)


def pubchem_idx():
    return data('pubchem_idx')
