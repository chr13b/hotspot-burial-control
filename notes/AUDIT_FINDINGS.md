# Numbers-vs-CSV audit — light pass (2026-09-06)

Scope: every quantitative claim in the **Abstract, §1 (contributions), and §4** of `notes/PAPER_DRAFT.md`
(~150 claims), each checked against its `results/*.csv`, with sign conventions cross-checked against the
generating `src/*.py`. Full systematic pass (all sections, on the trimmed text) scheduled for the ~20 Sept freeze.

## Bottom line
130+ values matched exactly, **including every load-bearing one** — R²(L|P) 0.37/0.36 → ~63%; ground-truth ΔΔG
floor 88%/90%; the full SKEMPI feature-class-law battery (Spearman −0.30, CPI +0.059, 62/72, 18/19,
partial-Spearman, monomer-inert, Pearson +0.64); all four cross-architecture replications (not transposed);
the ipTM figure (**+0.226** reconfirmed) with pAE/pLDDT/composite/pTM; both dose laws (retention-vs-RMSD sign
confirmed negative); the entire epistasis section; Bennett P1–P3 + occlusion + non-parent; the three-point
gradient; catalytic dissociation; reciprocity; and the **MIF/PiFold burial-matched "advantage" sign**
(`gap_recovery = hit(hotspot) − hit(control)`, positive = advantage — matches the corrected §5).

## FIXED this pass (commit — 2 substantive errors)
1. **AB-Bind CPI (§4) — unsourced number removed.** Paper claimed the per-mutation distribution "adds
   **+0.031 [+0.015, +0.045]** (and +0.042 beyond geometry alone)" — **not present in any committed CSV**
   (`abbind_cpi.csv` holds only `logP | burial+ΔSASA+BLOSUM+vol = +0.00935 [−0.00128, +0.01877]`, "conditionally
   INDEPENDENT (CI spans 0)"; `src/abbind_cpi.py` hard-codes 5 tests, none producing +0.031/+0.042). Corrected
   to the committed **+0.009 [−0.001, +0.019]** (spans zero); AB-Bind is underpowered/indeterminate, SKEMPI
   settles it. The unsupported number had been presented as the main result and the real null as the caveat.
2. **ESM-IF1 dose-law redraw (§4) — σ/value transposition.** "σ = 0.99/1.00/1.01 give +0.0114, +0.0019, −0.0002"
   → correct mapping is σ=1.00→+0.0114, σ=0.99→+0.0019 (`leverage_noise_ladder_esmif_redraw.csv`:
   σ0.99→+0.00188, σ1.01→−0.00019; σ1.00→+0.0114 per `FINDINGS_esmif_dose_law.md`). Reordered to
   **+0.0019, +0.0114, −0.0002**. Conclusion (~0.012 spread, unstable) unchanged.

## DEFERRED to the full pass (numbers correct; traces/rounding only — no conclusion affected)
**Traceability (wrong/missing → citation):**
- §4 feature-class-law **table** prints `→ nugget_cpi.csv` nearby (5,742-pos / 141-cx sample for the adjacent
  ΔSASA figure), but the table's real source is **`w_placebo_ladder.csv`** (13,401-pos / 343-cx, matching the
  caption). Add a direct `→ w_placebo_ladder.csv` under the table.
- Partial-Spearman −0.147 / −0.094 cited to `w2_monomer_inert.csv` but live in **`w2_onepass_control.csv`**.
- Abstract/§1 "AUROC 0.69 / KL 0.68 / geometry 0.66 / confidence 0.51" has no inline cite (values correct in
  `leverage_triage.csv`, named only in §8) — add inline.

**Rounding-boundary slips (all at x.xxx5; a rounding-function difference, not re-run data):**
- feature-class table: confidence CI-lo −0.0003→−0.0002; scalar-KL +0.0009/+0.0016→+0.0010/+0.0017;
  leverage CI-lo +0.0033→+0.0034 (`w_placebo_ladder.csv`).
- "quadratic or cubic +0.0047": quadratic 0.00466 ✓, cubic 0.00463→+0.0046.
- PiFold position-level confidence CPI +0.0000→+0.0001 (`leverage_pifold.csv`).
- predicted-backbone OF3 CPI CI-lo +0.027→+0.026 (`leverage_predicted.csv`).
- three-point protease-inhibitor leverage-AUROC CI-hi 0.811→0.812 (`threepoint_law.csv`).
- "~62% even when wt identity added": ProteinMPNN 62.17% ✓ but ESM-IF1 63.32% → say "~62–63%".
- occlusion "earlier proxy 0.587": not in current `bennett_occlusion_allatom.csv` (superseded run) — UNVERIFIABLE,
  low-stakes; re-source or drop in the trim.

No further instance of the ipTM-value or deficit/advantage-sign error class was found.
