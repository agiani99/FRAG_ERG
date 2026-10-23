# FRAG-ERG

**Fragment-additive ErG fingerprints for pocket-aware ligand design, and retrieval of purchasable compounds.**

The basic idea under this fingeprint scheme is as depicted here: ![](https:://https://github.com/agiani99/FRAG_ERG/edit/main/blob/)

One engine, two ways to use it:

| | **Face A: de novo growth** | **Face B: purchasable retrieval** |
|---|---|---|
| Question | Which molecules, assembled from BRICS-type building blocks, keep the pharmacophore pattern (ErG) of a reference ligand and fit the pocket? | Which compounds that a vendor already lists, and that PubChem links to bioactivity, contain the key part of the reference ligand, resemble its pharmacophore pattern and fit the pocket? |
| Output | New molecules (usually not in any database) | Existing, listed molecules (and close analogues of generated ones) with vendors and bioassay summary |
| Main scripts | `grow.py`, `compress.py`, `pocket.py` | `anchor_scan.py`, `retrieve.py`, `analogues.py`, `ecfp_store.py` |
| Strength | explores new chemistry around a kept anchor | nothing has to be synthesised to test the result |
| Honest caveat | ErG alone does not steer growth to a specific molecule (see Benchmarks) | a vendor listing is evidence of an offer, not of stock |

Both faces share the same ErG engine, the same anchor selection (the most pharmacophore-rich BRICS block of the
reference, kept at its crystal coordinates) and the same 3D pocket check (excluded volume, burial, contacts, polar
complementarity with hydrogen-bond angles).

---

## 1. The engine

**ErG v2** (`erg_v2.py`): 6 feature counts (HA, HD, AR, HY, PLUS, MINUS) plus 21 feature-pair types x 15 topological-distance
bins (321 values, raw uint8; fuzzy +-1 bin spill-over 0.3 applied to the query). Similarity is a continuous Tanimoto.

**Fragment additivity** (`frag_erg.py`): BRICS bonds are acyclic bridges, so cutting them splits a molecule into a tree of
blocks. Pharmacophore sites are assigned per block and pair distances compose through the attachment points, which gives
the exact ErG vector of an assembled molecule from per-block tables by tree convolution, without building the molecule.
Cuts are site-aware (a bond is skipped if cutting it would split one pharmacophore site). The result is verified against
the from-scratch ErG of the assembled molecule: **0 mismatches** in every test (`validate.py`; `erg_verified` column in
every `grow.py` output).

**Block library** (`compress.py`): BRICS blocks of a PubChem sample (neutralised), kept if seen >= 3 times, <= 20 heavy
atoms, <= 3 attachments and passing substructure filters (about 48k blocks). One build, reused by all growth runs.

---

## 2. Face A: de novo growth

```bash
python grow.py --smiles "<reference SMILES>" --anchor-frag 1 --steps 6 --beam 40 --workers 8 \
       --complex Complex/6h0f.pdb --pocket-sdf Complex/6h0f.sdf --pocket-lig Y70_B_502 \
       --w3d 0.15 --w-polar 0.3 --out grow_6h0f.csv
python analogues.py grow_6h0f.csv        # link to Face B: purchasable analogues of every generated molecule
```

How it works: the anchor block(s) of the reference are kept; a beam search adds one block per step. The ErG vector of every
candidate is updated incrementally (exact), candidates that can no longer be closed are pruned, partial molecules are
placed in the pocket (anchor atoms fixed, ETKDG conformers) and ranked by ErG Tanimoto to the reference + 3D score.
Finished molecules pass PAINS / Brenk / SA / motif filters and are verified. Modes: BRICS-anchor growth, growth from
hydrogens for small ligands without BRICS bonds (`--grow-vectors`), docked ligands, JSON constraint rules
(`--constraints`), optional reward for molecules found in PubChem (`--known-index`, `--known-weight`).

---

## 3. Face B: purchasable retrieval

Retrieval is a pipeline of one-time data builds and then fast queries.

**One-time builds** (all resumable or single-pass; times are measured on a 19-worker desktop unless marked estimate):

| Step | Command | Result |
|---|---|---|
| ErG index of PubChem (131M rows, 42 GB) | `python erg_index.py build --csv <pubchem csv> --smiles-col <col> --id-col <id> --out pubchem_idx` | exact ErG nearest-neighbour search |
| Known-molecule index (InChIKey block) | `python build_known_index.py --idx pubchem_idx --out known_idx --workers 19` | "is this molecule in PubChem?" (131,039,076 keys) |
| Vendor sources | `python source_table.py` (after saving the PubChem source table, see header) | `pubchem_sources.csv` |
| Purchasable flag | `python purchasable_flag.py scan SID-Map.gz --out purchasable` then `... join --idx pubchem_idx --out purchasable` | tight / medium / ZINC-only flag per CID and per index row (scan 5.3 min) |
| ECFP4 store | `python ecfp_store.py build --idx pubchem_idx --flags purchasable_rows.npy --out ecfp_purch --workers 19` | 66.7M rows, 17 GB |
| Vendor map | `python vendor_annotate.py build SID-Map.gz --out vendor_map` | CID to vendor sources and SID (8.1 min) |
| Bioactivity map | `python bio_annotate.py build bioactivities.tsv.gz --out bio_map` | per-CID active / tested counts, best potency, genes |

Inputs come from the PubChem FTP (`Substance/Extras/SID-Map.gz` 3.3 GB, `Bioassay/Extras/bioactivities.tsv.gz` 3.0 GB).

**Purchasable tiers** (a CID is flagged when a source lists a substance that standardises to it): *tight* = named
aggregators and screening-library suppliers (Molport, Mcule, Enamine, Chem-Space, eMolecules, ChemBridge, ChemDiv, Vitas-M,
Life Chemicals, Interbioscreen, Specs, Asinex, MolMall); *medium* = tight plus every "Chemical Vendors" source with at
least 100,000 live substances; *ZINC-only* = listed only through ZINC (second-hand evidence). The default store is medium + ZINC-only.

**Queries**

```bash
# (1) anchor-first retrieval: every purchasable molecule that contains the anchor, ranked by ErG and pocket fit
python anchor_scan.py --smiles "<reference>" --anchor-frag 1 --workers 19 --out hits.parquet     # exact, ~35 min for 66.7M rows
python retrieve.py   --smiles "<reference>" --hits hits.parquet --anchor-frag 1 \
       --complex Complex/6h0f.pdb --pocket-sdf Complex/6h0f.sdf --pocket-lig Y70_B_502 --pocket-top 100 --workers 8 --out retrieve.csv

# (2) ErG-neighbour retrieval (no anchor scan; works without a pocket)
python retrieve.py --smiles "<reference>" --k 30000 --out retrieve_nn.csv

# (3) purchasable analogues (ECFP4 Tanimoto >= 0.85) of any molecule list, e.g. grow.py output
python analogues.py grow_6h0f.csv
python ecfp_store.py search --store ecfp_purch --idx pubchem_idx --vendors vendor_map --bio bio_map --smiles-file cands.smi
```

Each result row carries CID, tier, number of vendors (and how many are tight-tier), the PubChem substance link of a
listing, bioactivity counts (active / tested results, distinct active genes, best potency) and the 3D pocket scores.
A high active/tested ratio over many unrelated assays means promiscuity, not a repurposing lead.

---

## 4. Benchmarks and what they show

All numbers were produced with the scripts in this repository on the data described above; each script writes its own
summary file. Treat the small samples accordingly.

| Question | Script | Result |
|---|---|---|
| Is the incremental ErG exact? | `validate.py`, `erg_verified` | 0 mismatches |
| Does ErG separate actives from size-matched inactive decoys? | `decoy_eval.py` | pooled MCC 0.488 (ErG) vs 0.476 (ECFP4) vs about 0 (size only); AUC is reported only as a side column |
| Does ErG guidance help growth beyond the anchor? | `retro2.py` | +0.014 to +0.018 ECFP4 to held-out actives over a wrong-target control; the reference was rediscovered in 0 of 70 targets |
| Are generated drug-sized molecules known? | `grow.py --known-index` | 0 of 1,007 found in PubChem |
| Can growth rebuild a known molecule? | `rebuild_bench.py` | 100 molecules (99 drugs), largest terminal block missing: **63/100 with the standard PubChem block library** (77/100 if the molecule's own blocks are added; random-choice control 1-2/100). Only the most influential block given: 1/100; two blocks: 2/100. Failures are lost at the first expansion (the correct block is ranked out of the beam). |
| Does the ECFP store pre-screen an anchor search? | `ecfp_store.py` | not exactly: 99 % recall keeps 10-50 % of rows, so the anchor scan is an exact SMARTS scan |
| How big is the purchasable set? | `purchasable_flag.py` | 59.5M of 131.0M index rows (medium), 13.6M tight, 7.3M ZINC-only |
| Anchor scan, glutarimide of pomalidomide | `anchor_scan.py` | 63,067 purchasable molecules contain it (66.7M scanned, 34.8 min) |
| Purchasable analogues of generated molecules | `analogues.py` | 4 of 15 small pomalidomide-like molecules; drug-sized molecules are almost never listed |

**Reading these together.** ErG is a coarse, non-identifying descriptor (drug-like molecules share their exact vector with
a median of about 100 others in PubChem), so it is good at ranking within a chemical series and poor at pointing growth
at one specific molecule. That is why the repository has two faces: growth explores around an anchor, retrieval uses ErG
only to rank molecules that already contain the anchor and already exist.

---

## 5. Install

**Data locations.** Large inputs are found through `paths.py`. Set `FRAGERG_DATA` to a folder containing
`pubchem_idx/`, `ALL_chembl_ErG.parquet`, `CHEMBL37_notarget.parquet`, `MAX_FDAapproved.parquet` and
`MAX_clinical.parquet` (only the ones a given script needs); command-line options such as `--idx` override it.

```bash
export FRAGERG_DATA=/data/fragerg          # PowerShell:  $env:FRAGERG_DATA = "D:ragerg_data"
```


```bash
conda create -n frag-erg python=3.12 rdkit numpy pandas pyarrow scipy numba scikit-learn tqdm
# optional C++ ErG kernel: cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build   (pybind11 needed;
# cpp/build.bat is the author's Windows/MSVC script with a hard-coded conda path: adapt it)
```

Python 3.12 (the tested environment), RDKit 2024.09, numba 0.65, NumPy 2.1. The C++ kernel (`erg_kernel`) only speeds up
index building; searches and growth do not need it. Pocket scoring expects a protonated protein (hydrogens added, e.g. by
Protoss); the example complexes in `Complex/` are prepared that way.

---

## 6. Repository layout

```
core        erg_v2.py  erg_index.py  frag_erg.py  filters.py  paths.py  cpp/
face A      compress.py  grow.py  pocket.py  known.py  build_known_index.py
face B      source_table.py  purchasable_flag.py  ecfp_store.py  vendor_annotate.py  bio_annotate.py
            anchor_scan.py  retrieve.py  analogues.py
benchmarks  validate.py  retro.py  retro2.py  decoy_eval.py  rebuild_bench.py
tests       test_erg_v2.py  test_index.py  test_ecfp_search.py
examples    Complex/ (6h0f, 6VUE, 6mo0, 7CR2 with ligands)
docs        README.md  docs/DEVELOPMENT_LOG.md (the detailed lab notebook, sections 2.1-2.26)
```

---

## 7. Limitations

- The 3D check is an approximation: rigid protein, ETKDG conformers with the anchor fixed, no minimisation, no waters, no scoring function. It answers "is there room and are the polar groups satisfied", not "how strongly does it bind". Stereochemistry is not part of the anchor match.
- ErG is topological (2D); the pocket enters only through the excluded-volume / burial / polar terms and the anchor coordinates.
- The index holds charged forms; the ErG neighbour search is approximate for ionisable references (neutral ErG is recomputed afterwards). Isotopologues, salts and protonation forms share an InChIKey block and are merged to one parent form.
- A vendor listing means a compound is or was offered. Make-on-demand and virtual catalogue entries exist in the medium tier; use the tight tier for a stricter reading. No stock check is made.
- Rebuild benchmark: 100 molecules (99 drugs), one seed, success judged before the property filters, no pocket, no random control for the one- and two-block conditions.
- PubChem identity is by InChIKey first block (no stereo, no isotopes).

## 8. Related work

Descriptor-guided search of fragment spaces without enumeration is established: FTrees-FS (exact, reduced-graph feature
trees), SpaceLight (ECFP / CSFP), SpaceMACS (maximum common substructure) and SynthonOR (approximate synthon fingerprint
composition) search make-on-demand spaces, and V-SYNTHES docks synthons of Enamine REAL Space. FRAG-ERG differs in using
an exact fragment-additive ErG with a pair-distance histogram, in keeping an anchor at crystal coordinates, and in
targeting compounds that are already listed and linked to PubChem bioactivity rather than make-on-demand products. No
head-to-head comparison with those tools has been made. See `docs/DEVELOPMENT_LOG.md` section 2.26 for the sources.

## 9. Citation and licence

To be added by the authors. Data: PubChem (public domain), PDB entries as cited in `Complex/`. BRICS: Degen et al.,
ChemMedChem 2008 (RDKit implementation).
