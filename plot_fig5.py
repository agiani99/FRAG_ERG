"""Figure 5 (2D): block swaps on pomalidomide (6H0F, cereblon). Reference + the 8 accepted purchasable swaps + 3 products removed by the
hydrogen-bond retention check. Swapped block highlighted. Data: ../BIOMIM/swaps_pomalidomide_hb.csv. usage: python plot_fig5.py"""
import os, io, re
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

here = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(here, '..', 'BIOMIM', 'swaps_pomalidomide_hb.csv'))
REF = 'Nc1cccc2c1C(=O)N(C1CCC(=O)NC1=O)C2=O'
INK, MUTED, OR, RED, GR = '#1f2328', '#6b7280', '#d9822b', '#c0392b', '#2f8f5b'
def lost_text(v):
    out = []
    for it in str(v).split(';'):
        m = re.match(r'\s*([A-Z]{3})_[A-Z]_(\d+):(\w+)\((lig \w+)\)', it)
        if m:
            out.append(f'{m.group(1).title()}{m.group(2)} {m.group(3)}')
    return 'loses ' + ' + '.join(out) if out else str(v)


def vend(n):
    return f'{int(n)} vendor' + ('' if int(n) == 1 else 's')


def block_atoms(prod, block):
    q = Chem.MolFromSmarts(re.sub(r'\[\d+\*\]', '*', block))
    if q is None:
        return []
    m = prod.GetSubstructMatch(q)
    wild = [i for i, a in enumerate(q.GetAtoms()) if a.GetAtomicNum() == 0]
    return [a for i, a in zip(range(len(m)), m) if i not in wild]


def draw(smi, hl=(), w=520, h=380):
    m = Chem.MolFromSmiles(smi)
    dr = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = dr.drawOptions(); o.bondLineWidth = 2.2; o.addStereoAnnotation = False; o.padding = 0.08
    o.setHighlightColour((0.95, 0.80, 0.55, 1.0)) if hasattr(o, 'setHighlightColour') else None
    hl = list(hl)
    bonds = [b.GetIdx() for b in m.GetBonds() if b.GetBeginAtomIdx() in hl and b.GetEndAtomIdx() in hl]
    col = {i: (0.95, 0.80, 0.55) for i in hl}; bcol = {i: (0.95, 0.80, 0.55) for i in bonds}
    rdMolDraw2D.PrepareAndDrawMolecule(dr, m, highlightAtoms=hl, highlightBonds=bonds, highlightAtomColors=col, highlightBondColors=bcol)
    dr.FinishDrawing()
    return Image.open(io.BytesIO(dr.GetDrawingText()))


acc = d[d.accepted & (d.analogues > 0)].sort_values('erg', ascending=False)
rej = d.loc[[17, 22, 24]]
cells = [dict(smi=REF, hl=[], title='pomalidomide (6H0F ligand)', line2='reference', color=INK)]
for _, r in acc.iterrows():
    prod = Chem.MolFromSmiles(r.smiles)
    cells.append(dict(smi=r.smiles, hl=block_atoms(prod, r.new_block), title=f'CID {int(r.best_analogue_cid)}',
                      line2=f'ErG {r.erg:.2f} · block ECFP4 {r.block_ecfp4:.2f}',
                      line3=f'{int(r.pocket_hb_kept)}/{int(r.pocket_hb_ref)} H-bonds kept, none lost · {vend(r.best_analogue_n_vendors)}', color=GR))
for _, r in rej.iterrows():
    prod = Chem.MolFromSmiles(r.smiles)
    cells.append(dict(smi=r.smiles, hl=block_atoms(prod, r.new_block), title=f'CID {int(r.best_analogue_cid)}  (rejected)',
                      line2=f'ErG {r.erg:.2f} · block ECFP4 {r.block_ecfp4:.2f}',
                      line3=lost_text(r.pocket_hb_lost), color=RED))

nc = 4; nr = int(np.ceil(len(cells) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(11, 3.05 * nr), dpi=300)
for ax in axes.ravel():
    ax.axis('off')
for ax, c in zip(axes.ravel(), cells):
    ax.imshow(draw(c['smi'], c['hl']), extent=(0, 1, 0.26, 1.0), aspect='auto')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.06)
    ax.add_patch(plt.Rectangle((0.01, 0.0), 0.98, 1.04, fill=False, edgecolor=c['color'], lw=1.4 if c['color'] != INK else 1.0))
    ax.text(0.04, 0.215, c['title'], fontsize=8.2, fontweight='bold', color=c['color'], va='center')
    ax.text(0.04, 0.135, c['line2'], fontsize=7.4, color=INK, va='center')
    if c.get('line3'):
        ax.text(0.04, 0.055, c['line3'], fontsize=7.0, color=c['color'], va='center')
fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.05, hspace=0.12)
fig.text(0.5, 0.965, 'Pharmacophore-preserving block swaps on pomalidomide (orange: swapped block; green: accepted and purchasable; red: removed by the H-bond check)',
         ha='center', fontsize=8.5, color=INK)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(here, f'fig5_blockswaps.{ext}'))
print(len(cells), 'cells;', len(acc), 'accepted purchasable')
