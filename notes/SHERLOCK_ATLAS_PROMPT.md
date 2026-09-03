# Sherlock task — ATLAS (TCR–pMHC) second fixture via ANARCI renumbering (path 1)

**Pre-registered:** `results/PREREG_atlas.md` (frozen — do not edit). Bonus replication: does the
confidence–leverage decomposition hold on a genuinely non-overlapping natural ΔΔG_bind fixture? ATLAS is TCR–pMHC
(narrow biology, ~300 non-overlapping mutations → likely modest/indeterminate power — that is fine and expected).
`SEED=20260803`. The blocker that stopped the local build was the **mutation numbering ≠ raw-PDB numbering**; this
handoff resolves it with **ANARCI IMGT renumbering + a hard WT-identity gate**. Repo:
`/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`.

## The correctness guarantee (read first)
**No ATLAS number is claimed until EVERY scored mutation passes a WT-identity gate**: after mapping a mutation
`S25A` to a PDB residue, the residue's actual identity in the structure MUST equal the mutation's wild-type letter
(`S`). Drop (and log) any mutation that fails. Report the fraction that pass. If the pass rate is low (<~80%), the
numbering scheme is wrong — try a different ANARCI scheme before proceeding, and disclose the final rate. This is
what makes ATLAS defensible; without it, do not report a result.

## Phase 0 — sync, env, data
```bash
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
git pull --no-edit origin main
# ANARCI (fast, HMMER-based): conda install -c bioconda anarci  (or pip install anarci + hmmer in the env)
python3 -c "import anarci; print('anarci OK')"
mkdir -p data/atlas/pdb
base=https://raw.githubusercontent.com/weng-lab/ATLAS/master/www
curl -sSL $base/scoring/Mutants_052915.tsv        -o data/atlas/Mutants.tsv     # 418 rows; Delta_DeltaG_kcal_per_mol
curl -sSL $base/scoring/CDR_seqs.txt              -o data/atlas/CDR_seqs.txt    # per-PDB chain map: pMHC | TCR
# structures (121): loop the true_pdb dir listing (GitHub API) and fetch each *.pdb into data/atlas/pdb/
```
`data/atlas/` is third-party — keep it **gitignored** (do not commit it).

## Phase 1 — chain groups (authoritative, no classifier)
- Convention from ATLAS's own `www/scoring/build_models.py`: `tcr_chain_map = {'A':'D','B':'E'}` → MHC = chain
  A(+B), peptide = C, TCR α = D, TCR β = E. `CDR_seqs.txt` gives a per-PDB `pMHC` / `TCR` chain string as a
  cross-check (with class-II variation, e.g. 3PL6 lists TCR=CD). **Use CDR_seqs's TCR/pMHC columns per PDB** where
  present; fall back to {D,E}/{A,B,C}. So g1 (scored) = TCR chains, g2 (partner, deleted for the monomer) = pMHC
  chains. `ftax_common.load_complex(path, pdb, g1_str, g2_str)` takes these directly.

## Phase 2 — ANARCI renumber + map + WT-identity gate (the crux)
For each mutation row with a real `Delta_DeltaG_kcal_per_mol` on a **non-overlapping** PDB (exclude SKEMPI codes —
see `results/leverage_skempi_positions.csv` complex prefixes):
1. Parse `TCR_mut` = `{wt}{num}{mut}` (single-point; skip multi-mut for the first pass). Chain = the TCR PDB chain
   (`TCR_PDB_chain`, or map `TCR_mut_chain` A/B → D/E).
2. **ANARCI-renumber that TCR chain to IMGT** (try `scheme='imgt'` first; if the gate fails broadly, try
   `kabat`/`chothia`). This yields, per structure residue, its scheme number. ATLAS's `num` is in that scheme.
3. Map `num` → the structure residue whose scheme-number == `num` on that chain.
4. **GATE:** assert the structure residue's amino acid == `wt`. Also sanity-check the `wtCDRseq` appears in the
   chain. Drop + log failures. Keep only passing mutations.
Write `results/atlas_fixture.csv`: `complex_id (pdb_g1_g2), chain, resnum, icode, wt, mut, ddG, cdr, gate_pass`.
Print the pass rate and the per-PDB mutation counts.

## Phase 3 — score leverage (GPU), reusing the committed machinery
Write `src/leverage_atlas.py` (a thin adapter — mirror `leverage_decomposition.py`'s per-position scoring but
over the ATLAS fixture + chain groups):
- For each complex: `cx = load_complex(pdb, g1=TCR, g2=pMHC)`; run ProteinMPNN (complex pass P) and the
  partner-ablated monomer pass (TCR only, pMHC deleted → Q), 8 decode orders averaged, sequence-free marginals —
  exactly as `leverage_decomposition.py` does. Compute `L_i(a) = (logP_i(a)−logP_i(wt)) − (logQ_i(a)−logQ_i(wt))`.
- Repeat with ESM-IF1 (`leverage_esmif.py` scorer) for the 2-model check.
- Emit per-mutation `L` (at the mapped position/substitution) + geometry (burial, nbr, ΔSASA) + `conf`, `klP`,
  `negH`, joined to `ddG` — the same columns `leverage_skempi_mutations.csv` has, so the analysis is identical.
- **Positive control (rule 6):** ProteinMPNN interface native recovery on a held ATLAS complex in the normal
  range (~0.3–0.5); a bit-identical re-score reproduces.

## Phase 4 — analyse (identical to SKEMPI), on the non-overlapping subset
Reuse `leverage_decomposition.py`'s CPI machinery (`LD.cpi`) + the placebo floor:
- **H1:** mutation-level **Spearman(L, ΔΔG) < 0** and **CPI(L | burial+nbr+ΔSASA) > the placebo floor**;
- **H2:** position-level **CPI(confidence | geometry) ≈ 0**;
- report the placebo floor on this fixture, `n` mutations / `n` complexes, the SKEMPI-overlap fraction, and the
  WT-gate pass rate. **Falsifier (verbatim if it fires):** Spearman(L,ΔΔG) ≥ 0 or CPI(L|geom) not clearing the
  floor → the decomposition does not replicate on TCR–pMHC (a bounded-generalization result, not a refutation).

## Phase 5 — deliverables
Commit **only**: `src/leverage_atlas.py`, `results/atlas_fixture.csv`, `results/atlas_leverage.csv`,
`results/FINDINGS_atlas.md` (the H1/H2 numbers + CIs, the WT-gate pass rate, the overlap fraction, `SEED`, exact
commands). `git add` by name (not `-A`); keep `data/atlas/` gitignored; commit with the two trailer lines; push;
message me the Spearman(L,ΔΔG) + CPI(L|geom) + the gate pass rate.

## Guardrails
- The **WT-identity gate is mandatory** — a mutation that fails it is dropped, never mapped by force.
- Report power honestly: ~300 mutations / ~90 complexes is AB-Bind scale; an **indeterminate/wide-CI result is a
  valid, reportable outcome** (do not massage the scheme or subset to manufacture significance).
- Do not touch `results/PREREG_atlas.md`. Do not commit the ATLAS third-party data.
