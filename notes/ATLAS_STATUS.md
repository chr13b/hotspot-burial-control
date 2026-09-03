# ATLAS second-fixture attempt — status (bonus; honest stop at the numbering blocker)

Pre-registration: `results/PREREG_atlas.md` (committed, frozen). This note records how far the attempt got and
the exact blocker, so the decision is documented and the work is resumable.

## What is SOLVED
- **Viability (pre-registered overlap control = GO).** ATLAS `www/structures/true_pdb` = **121 structures**, of
  which **90 are NON-overlapping with SKEMPI** (the `www/all` subset that showed only 16 was misleading). So it
  is a genuine independent fixture, not a SKEMPI re-run.
- **Labels.** `www/scoring/Mutants_052915.tsv` gives ΔΔG **directly** (`Delta_DeltaG_kcal_per_mol`): **310/418
  rows carry a real ΔΔG**; ~300 fall on non-overlapping structures (AB-Bind scale).
- **Chain groups (partner-ablation operator).** `www/scoring/build_models.py` fixes the ATLAS convention —
  `tcr_chain_map = {'A':'D','B':'E'}`: MHC = chain A(+B), peptide = C, TCR α = D, TCR β = E. So g1 (TCR) = {D,E},
  g2 (pMHC) = {A,B,C}. `www/scoring/CDR_seqs.txt` gives a per-PDB chain map as a cross-check (with class-II
  variation, e.g. 3PL6 lists TCR=CD). `ftax_common.load_complex(path, pdb, g1, g2)` accepts these groups
  directly, so the leverage machinery is reusable unchanged.
- **Tooling.** biopython 1.83 present; network fetch of the ATLAS repo works.

## The BLOCKER — mutation numbering ≠ raw-PDB residue numbering
The mutation strings (`TCR_mut`, e.g. `S25A`) use ATLAS's **canonical/IMGT-style numbering**, which does **not**
match the residue numbers in the raw `true_pdb` structures. Direct test on 3PL6: using the mutation number as a
PDB residue number, **only 1/6 mutations landed on a residue whose WT identity matched** (e.g. "Y46A" → PDB
residue 46 is L, not Y). `build_models.py` writes the mutation number straight into a Rosetta resfile, i.e. it
operates on **ATLAS-preprocessed structures that were renumbered/rechained to the canonical A/B/C/D/E scheme** —
which the raw `true_pdb` files do not carry. `wtCDRseq`-anchoring alone is insufficient: the stated CDR loops do
not contain all mutated residues (the numbering spans framework-adjacent positions the short CDR string omits).

Scoring on a wrong position map would produce a **false** ATLAS result — exactly the failure mode the project's
ground rules forbid ("never a false positive"; "run positive controls before trusting"). So the attempt STOPS
here rather than force a fragile mapping.

## Two ways to resolve it (if the bonus is pursued)
1. **IMGT-renumber the TCR chains with ANARCI** (or reconstruct ATLAS's `build_models` preprocessing): renumber
   each raw `true_pdb` TCR chain to the ATLAS/IMGT scheme so the mutation number aligns to a PDB residue, then
   validate per mutation that the WT identity matches (a hard gate). ANARCI is not installed here; this is a
   Sherlock/env task.
2. **Use ATLAS's pre-built processed structures** (`www/structures/designed_pdb`, 520 models in the canonical
   scheme) as the WT/mutant backbones — but these are Rosetta-repacked, a modeled (not crystal) backbone, which
   changes the experiment slightly and needs its own justification.

## Recommendation
Given ATLAS is a **bonus** with **modest expected power** (~300 non-overlapping mutations, AB-Bind scale, likely
"indeterminate"), and the numbering reconciliation is a correctness-critical multi-step effort, the disciplined
default is to **rest on the §9 framing** (which already names ATLAS as a non-overlapping-but-narrow replication
target, with the honest note that no clean large natural fixture exists). Pursue path 1 only if a clean second
fixture is judged worth the ANARCI-renumbering investment. Either way, **no ATLAS number is claimed** until a
mutation-position map passes the WT-identity gate on every mutation.
