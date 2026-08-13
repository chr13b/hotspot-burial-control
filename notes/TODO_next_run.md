# TODO — next run (handoff, 2026-08-13)

Session landed the full science + the abstract/title/contributions reframe. What remains (none blocks the
story; ordered by leverage):

## 1. Finish the §-body reframe to match the new abstract  (writing, ~1 session)
The **title + one-paragraph abstract + §1 contributions** are reframed (nugget-forward + Big-Idea-1-forward,
KL demoted to "learned frustratometer", matched-pair = diagnostic *protocol*). The **section bodies §3–§7
still read KL-detector-forward** and must be collapsed to match:
- Add a dedicated section for **Big Idea 1** ("the model knows binding, in its distribution not its
  confidence": P1/P2/P3 + the occlusion-vs-energetics +0.025 + per-target robustness). → FINDINGS_knows_where.md,
  bennett_knows_where.csv, bennett_occlusion_energetics.csv.
- Demote the KL-detector material to one paragraph ("a learned frustratometer ≈ ΔSASA on all 4 backbone
  classes", R1 = kl_geometry_control_predicted.csv). **Cut** capture@k tables, Lever-2 kcal/mol, the
  four-class KL "money" panel.
- Fold in the **combiner-free nugget** via CPI (nugget_cpi.csv: confidence CPI 0.000) and the **designer
  table** (baseline_audit.csv) as the §4 core.
- **Figure inventory:** replace with (a) burial-confound + 5-model forest; (b) the designer table
  (confidence below random, geometry best); (c) Big Idea 1 dissociation (core vs interface) + the
  energetics +0.025; (d) the AF2-vs-OF3 ρ=0.57 scatter. Retire all KL-triage figures.

## 2. Per-target robustness of Big Idea 1 — DONE, just fold in
All 4 targets consistent: P2 (core 0.66–0.74 > interface 0.60–0.63), P3 (P−Q +0.067/+0.094/+0.081/+0.089).
Put the per-target table in the appendix (recompute traceably or commit the inline numbers).

## 3. Optional theoretical ceiling-raiser — constraint-vs-leverage + confidence-decay gradient
Theory: model confidence estimates positional *constraint* (H(a_i|X)); hotspot-ness is *leverage*
(∂ΔG_bind/∂a_i); they coincide only under binding-dominated selection. **Testable prediction:** confidence's
hotspot AUROC should decay obligate → transient → de-novo. **Feasibility TBD** — needs SKEMPI complex-type
(obligate/transient) annotations; Bennett de-novo is the low-selection end. If the gradient appears, the
paper gains a genuine model, not just measurements. Next-cycle-caliber; not required.

## 4. Cheap hardening / honesty items
- **Verify search-only citation URLs** before submission (Frustratometer, HotPoint, DBAC, BindCraft,
  Surf2Spot, ProtDBench, target-conditioned-IF, UMA-Inverse) — flagged in §7 with ✎. BAIF/DeSAE/CPI fetched.
- **Conservative-substitution ΔΔG re-scoring** (Big Idea 4b): does burial's hotspot advantage decay when the
  label uses volume-matched substitutions (removing the truncation term)? Power-check first (may be sparse).
- **Occlusion caveat**: our clash baseline is volume/contact-based; note a richer all-atom clash model could
  narrow the +0.025.
- **Zenodo archival** (task #11) before the ~2026-10-09 SCRATCH purge and at submission.
- Polish the abstract from "sketch" to final length; retitle-check.

## Committed artifacts this session (all trace to CSVs)
kl_geometry_control{,_predicted}.csv (R1), nugget_partner_sensitivity.csv, nugget_cpi.csv, baseline_audit.csv,
dsasa_matched_sens.csv, composition_confound.csv, probid_gap_estimators.csv, bennett_kl_detector.csv,
bennett_knows_where.csv (+_pairs), bennett_occlusion_energetics.csv, PREREG_knows_where.md, FINDINGS_{bennett,
knows_where,kl_triage}.md.
