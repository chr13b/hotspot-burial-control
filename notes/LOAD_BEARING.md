# LOAD-BEARING FILE SET — the de-identified submission repo (packaging step 2)

*Decision doc for the anon fresh-init. Written 2026-09-17. Precondition met: numbers FROZEN (exhaustive
audit clean, 7a3e170). This is the prune that gates the fresh-init — once reviewed, the fresh-init can fire.*

## The rule
Ship ONLY what the paper needs to be **read** and **reproduced from committed CSVs** (no GPU/LFS/Sherlock).
Everything else is excluded unless an `INDEX.md` row needs it. Concretely the de-identified repo carries:

```
paper/     PAPER_DRAFT.md (scrubbed) + built main.pdf + latex/ (scrubbed)
src/       load-bearing scripts only (see below)
results/   load-bearing CSVs only (see below)
figures/   results/figures/*.{pdf,png}
environment/  README.md + frozen env spec (scrubbed of scratch paths)
README.md  INDEX.md  reproduce.sh  DATA.md  BRIEF.md
```
Excluded from the repo entirely: git history (70 tracked files leak `bertsch`/`cbertsch` + author identity →
**fresh `git init`, never a clone**), `CLAUDE.md`/`notes/` internal working files, the FoldX binary + rotabase
(license), model/RFdiffusion weights (third-party), and the big regenerable artifacts (→ Zenodo, below).

## SHIP — CSVs  = (cited by paper) ∪ (read by a figure script)
Mechanically, the ship-CSV set is the union of these two greps (run at fresh-init):
```
cited=$(grep -oE '[a-z0-9_]+\.csv' notes/PAPER_DRAFT.md | sort -u | grep -v '^file\.csv$')
fig=$(grep -rhoE 'results/[A-Za-z0-9_./-]+\.csv' src/fig*.py | sed 's#results/##' | sort -u)
ship_csv=$(printf '%s\n%s\n' "$cited" "$fig" | sort -u)   # ~78 files
```
- **Paper-cited: 72** CSVs (the `→ file.csv` traces; `file.csv` is a code-example false hit, drop it).
- **Figure-input additions not already cited** (must ship or a figure won't build): `hero_pdbs/1GL0_E_I_meta.csv`,
  `iptm_steer.csv`, `iptm_steer_120.csv`, `matched_recovery.csv`, `p0_dssp_summary.csv`,
  `expD_af2_of3_corr.csv`, `expD_af2_of3_corr_percomplex.csv`, `leverage_skempi_positions.csv`.

## EXCLUDE — big regenerable intermediates (→ Zenodo, NOT the repo)
All `results/*.csv > 2 MB` **except `leverage_skempi_positions.csv`** (3.2 MB, a figure input → ships).
The 23 to relocate (regenerable from committed inputs + `src/`, and archived in the Zenodo bundle):
`p0_positions.csv` (130 MB), `p0_n002_positions.csv` (56 MB), `leverage_bennett_pairs.csv`,
`bennett_knows_where_pairs.csv`, `p0_dssp_interface_resid.csv`, `p0_interface_resid.csv`,
`leverage_pq_skempi{,_pifold,_esmif,_mif}.csv`, `frustration_monomer_joined.csv`, `kl_detector_positions.csv`,
`bennett_occlusion_allatom_pairs.csv`, `skempi_conservation_positions.csv`, `p0_n002_interface_resid.csv`,
`panel_{mpnn_soluble,pifold,mif}_positions.csv`, `leverage_{pifold,esmif,mif}_positions.csv`,
`catalytic_positions.csv`, `frustration_monomer_positions.csv`.
*(The paper cites the SUMMARY CSVs these produce, so excluding them costs the reader nothing; the reproduce
path regenerates them from the scripts + smaller committed inputs, or pulls them from Zenodo.)*

## EXCLUDE — pilots / smoke / superseded shards
`results/_*.csv` (`_abbind_xcheck`, `_cfg_pred_{af2,crystal,of3}`, `_pilot_esmif_sigma0`),
`*_audit2.csv` / `*_audit3.csv` / `*_audit4.csv` (`kl_readout_audit{2,3,4}`, `abbind_readout_audit2`),
and any intermediate not in `ship_csv` and not an INDEX reproduce-input. Default = exclude.

## SHIP — scripts
Ship `src/x.py` iff it (a) writes a ship-CSV, (b) is a `fig*.py`, or (c) is a shared util
(`ftax_common.py`, `figstyle.py`, `validate.py`, `patch_ss.py`, `mcsa_build_labels.py`). Exclude pure
pilots/debug (`*_audit2/3/4.py`, `expC2_kl_debug.py`, `expC2_slope_diag.py`, smoke drivers). The exact set
is emitted by INDEX.md at fresh-init; ~120 of 156 scripts ship.

## Directory + de-id checklist (at fresh-init) — see SUBMISSION_RECON.md
1. Fresh `git init` (no history). 2. Copy the ship set into `paper/ src/ results/ figures/ environment/`.
3. Scrub `bertsch`/`cbertsch`/email/`/scratch/users/...` from every shipped file. 4. Delete the `⟨✎ …⟩`
marker in PAPER_DRAFT.md §8. 5. Write `INDEX.md` (paper number → CSV → script; seed from FOLDED_STATUS.md)
+ `README.md` + `reproduce.sh`. 6. `verify-references` on the .bib. 7. Data-Availability line → Zenodo
(reserved DOI, restricted now; public+named at camera-ready).

## STATUS
Prune **decided**. This is the last precondition before the fresh-init. Awaiting the user's go
("now — do the fresh init and send me the link") + the empty repo link; then populate as above.
