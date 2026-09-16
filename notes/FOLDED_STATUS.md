# FOLDED STATUS — what is in the paper, where, and its source CSV

*Index so we never lose track of a folded result or a framing decision. "Loc" = section(s) in
`notes/PAPER_DRAFT.md`. Every number traces to the listed CSV. Updated 2026-09-13.*

## Results folded (number → location → source)
| Result | Loc | Source CSV / FINDINGS |
|---|---|---|
| Confidence–leverage decomposition + Prop 1 no-go | §3, §4 | r2_leverage_from_P.csv |
| Feature-class law / CPI beyond geometry | §4 | (CPI tables) nugget_cpi.csv, leverage_esmif.csv |
| Scalar KL ≈ ΔSASA (crystal AND predicted; demoted) | §3, §8 | kl_geometry_control{,_predicted}.csv |
| **Beyond-X ladder (matched metric)** | §8, Fig. L | beyond_x_ladder.csv |
| FoldX Lane A detection (L −0.30 vs FoldX +0.43) | §4, Fig. P | foldx_detection.csv, foldx_laneA_deepen.csv |
| **L beyond fitted physics (partial −0.17)** | §4, §8 | foldx_laneA_deepen.csv |
| FoldX Lane B steering specificity (L−naive) | §4, Fig. P | foldx_steer.csv, foldx_steer_robustness.csv |
| **best-of-k design yield (+1.86)** + mean-over-k primary (+1.24) | §4, Fig. P | foldx_steer_robustness.csv |
| AB-Bind 2nd fixture (trend, underpowered) | §4 | foldx_detection_abbind.csv |
| L→kcal/mol calibration (0.42/unit, appendix-level) | §9(c) | laneA_calibration.csv |
| **Predicted-backbone steering (judge +0.70–0.76 n=106; ipTM +0.18 [+0.14,+0.22] n=93)** | §4 | cfg_steer_predicted.csv, iptm_predicted.csv |
| Judge matrix (4 architectures) + ipTM (AF2/Boltz-2) | §4 | cfg_judge_matrix{,_pifold}.csv, iptm_summary_*.csv |
| Generalization triad (SKEMPI/AB-Bind/ATLAS + catalytic) | §9 | atlas_summary.csv, foldx_detection_abbind.csv |
| Dose law (detection + intervention both survive predicted) | §4, §6 | rmsd_leverage_bridge.csv, cfg_steer_predicted.csv |

## Framing decisions folded (do not re-litigate; keep coherent)
- **best-of-k is the design-relevant yield** (a designer samples a few, keeps the best); mean-over-k is
  the pre-registered primary, kept visible in text/table/Fig. P. Transparent dual-report, not a goalpost move. (§4)
- **AB-Bind signs**: L's −0.18 / FoldX's +0.27 are the *same* correct ranking in opposite conventions
  (L = favorability, ΔΔG = cost) — the negative sign is right; magnitude, not sign, is accuracy. Sign-convention
  note in §4. Principle: explain enough that a reviewer feels smart for getting it.
- **"beyond physics"** = L adds binding rank-signal FoldX's fitted ΔΔG does not contain (partial −0.17); we do
  **not** compete with FoldX on absolute ΔΔG (it ranks better, 0.43 vs 0.30). (§4, §8)
- **(c) kcal/mol** modest: good ranker, rough absolute predictor (±1.8 too coarse for single mutations); L
  roughly *linear* in ΔΔG. (§9c)
- **Calibration is not "fitting"** the predictor: a 2-param post-hoc affine (scale/offset only); every headline
  stays zero-shot/scale-invariant. Appendix-level; bridge to the thermocycle paper. (§9c)
- **CFG mapping**: naive = unguided/marginal baseline; L = guidance direction → naive (not random) is the
  CFG-demanded control. (§4)
- **Naive-strong-baseline insight** (confidence carries 64–77% of L−random) is foregrounded WITH an explicit
  anti-contradiction guard: confidence-as-readout is at chance and L still beats the confidence *direction* on
  binding-specificity → the title "confidence is not competence" holds. (§4) **Title: KEEP.**
- **Fractal motif × beyond-X ladder** = two axes of one dissociation (levels × controls). (§1, §8)
- **Consensus tally**: 7 independent readouts × 3 modalities agree on +α·L, crystal AND predicted. (§8)

## Figures
- Fig. 0 overview (schematic) · Fig. P FoldX (detection + steering) · Fig. L beyond-X ladder — all Fable-polished,
  pass the 5.5in guard, numbers from CSV. Fig. 1–5 (decomposition/feature-class/dose-law/coupling/gradient) pre-existing.
- Hero: PyMOL (lean) vs biotite (backup) — decide at figure-selection.

## Deferred / next-paper (do NOT fold into this paper)
naive+L design recipe · tuned-L (invites "you fit it") · 2nd-derivative co-design · CFG to other conditioners ·
thermocycle fold+bind · per-burial-stratum temperature · Rosetta flex_ddG (optional, only if a reviewer presses).
