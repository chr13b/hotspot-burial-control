# ICLR figure plan + design system (rendering blueprint)

Design-audited plan, every number verified against a committed CSV. Render in matplotlib → PDF, at TRUE final
width (ICLR is single-column, 5.5in text block; full-width = 5.5in, half = ~2.65in). Never `\includegraphics`
a 10in figure down to 5.5 — it shrinks 8pt type to 4.4pt (the #1 tell of a rushed submission). Load the
`dataviz` skill before rendering.

## Data gaps to close first
- [x] **Per-class leverage-AUROC** — added to `p_3point_law.py` → `threepoint_law.csv` (`leverage_auroc_Lrms`).
      Flat+high (0.64/0.63/0.70) while confidence tracks the gradient. **DONE — highest-ROI number, rescues Fig 5.**
- [x] **Within-decile IQR of |L|** = 1.087 [1.057,1.127]× overall → `conf_decile_leverage.csv`. For Fig 1b. DONE.
- [ ] Conservation ranker BASE AUROCs (0.714→0.731) live only in `FINDINGS_conservation.md`; plot the Δ (in CSV).
- [ ] `capture@3` appears in two CSVs on DIFFERENT complex sets (`bindcraft_triage` 141 vs `leverage_triage`
      108). Never mix on one axis; use `leverage_triage.csv` (108, the only one with leverage).
- [ ] `leverage_noise_ladder_extra.csv` (MPNN redraw) is n=1,097, σ=0 CPI +0.092 (not 2,949/+0.058) — inset/
      normalised only; caption mention, never on the main ladder axis.

## The figure set (4 main + 1 optional half-width)

### Figure 1 — The operator (full width, 5.5×4.0in). Takeaway: *confidence is a scalar of one distribution; binding is how it moves when you delete the partner, and no scalar sees it.*
- **1a schematic** (top, full width): two passes on ONE frozen f_θ → P=p(·|X_complex), Q=p(·|X_monomer); confidence=diagonal φ(P) (grey, thin border, no fill) vs leverage=mixed 2nd derivative (blue, heavy border, 6% fill). Bottom strip = the CFG punchline: `ε_θ(x|c) − ε_θ(x|∅)` aligned vertically above `logp(·|X_complex) − logp(·|X_monomer)`, thin blue connectors between aligned pairs. Real 20-simplex bars for `3SZK/C/44` from `leverage_pq_skempi.csv` (exp+renorm lP_*/lQ_*). ✗ blind-by-construction / ✓ Spearman(L,ΔΔG)=−0.30. Ink + blue only, no red/green.
- **1b identifiability scatter** (bottom-left): 13,401 positions, x=confidence logP(wt), y=|L|_rms; non-hotspots grey rasterized, hotspots black; 10 confidence-decile bars spanning 5–95th pct of |L| — visibly SAME length. Annotate: "Spearman(conf,|L|)=+0.075; within-decile IQR = 1.09× overall — conditioning on confidence removes none of the spread." Two exemplars (3SZK C-Phe44 conf−0.97 |L|4.76 hotspot; 2QJ9 B-Asp82 conf−0.93 |L|0.05 non): "same confidence · 100× the leverage." Source `leverage_skempi_positions.csv` + `conf_decile_leverage.csv`.
- **1c non-vacuity** (bottom-right): R²(L|P) — max-flexible learner (GBM+RF) recovers ~0.37 (both models) → ~63% irreducible. Bars with CI, ref line at "P determines L" (R²=1) vs "irreducible" (0). Source `r2_leverage_from_P.csv`. [UPडATED from the old TV-matched 47/79 — that was peakedness-confounded.]

### Figure 2 — The feature-class law (full width, 5.5×4.4in). Takeaway: *every scalar of P sits at the placebo floor; only the mixed derivative clears it — two families, past geometry AND conservation, and it improves the ranker.*
- **2a placebo ladder** (top, full width — THE panel): horizontal bars+CI, ONE sample (13,401 pos/343 cx/327 hot). Rows ordered by derivative order: confidence +0.00019 [−0.00024,+0.00061] (grey) / negentropy +0.00087 / scalar-KL +0.00098 / **leverage −L(→Ala) +0.00485 [+0.00337,+0.00647]** (blue) / L|quadratic-geom +0.00466 / L|cubic +0.00463. Floor band x∈[0,+0.00072] `#E4E7E9`, "pure noise" tick at −0.00002, "6.7×" bracket to leverage. Class cues: "scalars of P — one pass" vs "mixed derivative — two passes". Source `w_placebo_ladder.csv`.
- **2b model class** (bottom-left): paired bars 4 feats × 2 models. MPNN (blue) 0.00019/0.00084/0.00093/**0.00485** (`leverage_decomposition.csv`); ESM-IF1 (vermillion) −0.00004/0.00060/0.00103/**0.00424** (`leverage_esmif.csv`). Print both n (13,401/344 vs 13,037/337).
- **2c beyond standard feature set** (bottom-mid): conservation-masked|geom +0.00635; L|geom +0.00484; **L|geom+conservation +0.00585** (LONGER — "undiminished" connector); conservation|geom+L +0.00830. "near-orthogonal Spearman(L,cons)=−0.08". Source `skempi_conservation_masked_cpi.csv`.
- **2d actionable** (bottom-right): Δ-AUROC bars. +|L|_rms on geom +0.0125 [+0.0007,+0.0246] (0.704→0.717, `w4_combined_ranker.csv`); on geom+conservation +0.0161 [+0.0042,+0.0290] (`skempi_conservation.csv`); optional +(−L_ala) +0.0220.

### Figure 3 — The dose law (full width, 5.5×2.5in, 3 panels). Takeaway: *survives sub-Å error, then a model-dependent cliff — so the fragility is the backbone's, not the network's.*
- **3a two ladders**: x=σ≈RMSD(Å) 0→2, y=CPI(L|geom) mut-level. MPNN (blue solid ●, n=2,949): .0575/.0588/.0474/.0321/.0024/−.0012. ESM-IF1 (vermillion dashed □, n=2,809): .0362/.0350/.0266/.0115/.0177/.0020/.0011. Marker fill ramps with σ (open→saturated). Floor shading σ≥1.0 (MPNN) + σ≥1.5 (ESM-IF1). Sources `leverage_noise_ladder{,_075full}.csv` + `leverage_noise_ladder_esmif_{all285,tail}.csv`.
- **3b model-free**: y=−Spearman(L,ΔΔG). MPNN .301/.294/.293/.195/.077/.060; ESM-IF1 .252/.170/.172/.118/.169/.096/.103. Annotate ESM-IF1 raw corr more jitter-robust than its CPI (untested hypothesis — the honesty panel).
- **3c realization variance @1Å**: 3 ESM-IF1 draws (200-cx subsample) +.0114/.0019/−.0002, spread ~0.012 = size of estimate; "1.0Å straddles the floor". Sources `leverage_noise_ladder_esmif{,_redraw}.csv`. Do NOT add the MPNN extra-redraw (diff sample).

### Figure 4 — The second mixed derivative (full width, 5.5×2.2in, 3 panels). Takeaway: *second order predicts binding epistasis — magnitude clearly, sign barely, ablation surfaces it.*
- **4a ablation forest**: partial-Spearman(C,g|dist). cross-interface ablated −0.129 [−0.253,−0.039] (blue FILLED); same-side un-ablated +0.014 [−0.152,+0.147] (grey OPEN, on zero); same-side ablated −0.118 [−0.253,+0.019] (blue open). Source `p3_coupling_summary.csv`.
- **4b magnitude**: binned-median (tertiles of |g|) of |C| with CI — NOT raw scatter. partial ρ(|C|,|g||dist)=+0.21. Source `p3_coupling.csv`+`p3_sign_verify.csv`.
- **4c honest sign bound**: model vs majority sign-accuracy, 4 subsets — all .542/.531, |g|>1 .625/**.694**, |C|>p75 .650/.621, |C|>p90 .679/**.696**. Baseline WINS on 2 (open diamond). "chance-corrected sign channel survives: partial ρ=+0.079 [+0.012,+0.172]". Source `p3_sign_verify.csv`. KEEP — a limitation panel buys trust.

### Figure 5 (optional, half-width 2.65×2.4in) — constraint vs leverage across interface types. NOW UPGRADED by the leverage-AUROC row.
- x = pre-registered transience order. **confidence-AUROC** (grey ●): TCR/pMHC 0.430 [.355,.508] · AB/AG 0.457 [.383,.518] · Pr/PI 0.554 [.465,.615] · de-novo 0.596 [.567,.624]. **leverage-AUROC** (blue ■, NOW COMPUTED): 0.641 [.539,.765] · 0.628 [.544,.716] · 0.701 [.599,.811] · (de-novo pending). burial-residualized conf (grey open dashed): .413/.426/.551. burial-AUROC (green △, runs OPPOSITE): .606/**.750**/.481. Chance line 0.5. Sources `threepoint_law.csv`+`bennett_conf_fork.csv`. Takeaway upgrades to "confidence tracks fold-coupling; leverage does not care" — the two feature classes diverge across a controlled axis. **This upgrade moves Fig 5 from cut-first toward level with Fig 4.**

## Design system
```python
INK="#1A1A1A"; RULE="#4D4D4D"; MUTED="#6B7379"
CONF="#9AA3AA"; NEGENT="#7E888F"; KL="#636D74"          # scalars-of-P ladder (greyer = more inert = the thesis)
LEVERAGE="#0B6FA4"                                       # mixed derivative; also = ProteinMPNN
ESMIF="#C0561F"                                          # ESM-IF1 / 2nd model
GEOM="#1C7C68"; CONSERV="#6D4E9C"                        # baselines
FLOOR_FILL="#E4E7E9"; FLOOR_EDGE="#B4BBC0"; GHOST="#C8CDD1"; PANEL_TINT="#F4F6F7"
RAMP_MPNN =["#D6E7F0","#A9CBDF","#7BAECC","#4B90B9","#0B6FA4","#08526F"]           # σ=0,.25,.5,.75,1,1.5
RAMP_ESMIF=["#F6DFD1","#EDBFA3","#E19E77","#D47C4A","#C0561F","#96411A","#6E2F12"] # +2.0
# diverging (signed heatmaps): #0B6FA4 <- #F2F2F0 -> #C0561F, mid pinned at 0
```
- **No red, no green-as-good/red-as-bad.** Okabe–Ito hues, deuteranopia+protanopia safe.
- LEVERAGE & ESMIF close in luminance → whenever both appear, also differ by line style + marker fill (MPNN solid+filled ●, ESM-IF1 dashed(4,2)+open □). Greyscale-safe (reviewers print).
- CONSERV & ESMIF never co-occur.
- Typography: `font.sans-serif=["Nimbus Sans","Helvetica","Arial","DejaVu Sans"]` (Nimbus Sans = URW Helvetica clone, verified renders ΔΔG/ρ/Å/− with no fallback; DejaVu is THE unstyled-matplotlib tell), `mathtext.fontset="stixsans"`, `pdf.fonttype=42`. Sizes @ true width: panel letter 9pt bold (lowercase, no paren/period, flush left, consistent x-offset), panel title 8.5pt (states the FINDING not the variable, ≤60 char, no in-image bold title — caption carries it), axis 8, tick 7.5, annotation 7, n=/note 6.5 MUTED. Nothing <6.5pt.
- **Significance = filled marker when CI excludes reference, open when it spans it** (stated once per caption; NO asterisks — this is a CI paper).
- CI: bars/dots thin lw0.9 INK no caps; curves 12%-alpha band. All 95% complex-clustered bootstrap (3000 for CPI, 2000 for AUROC/Spearman, seed 20260803) — boilerplate once per caption.
- Placebo floor: identical every time (band 0→floor, hairline edge, one 6.5pt inline label; never a coloured line/legend). Chance: dashed RULE lw0.8 "chance". Zero: solid RULE lw0.8.
- **Sign convention (most important): up/right = MORE binding signal everywhere** → plot −L, −Spearman(L,ΔΔG), −L(→Ala); say so in axis labels.
- Every panel prints its own `n=…, … complexes` (samples genuinely differ: 13401/13037/5742/2949/2809/557/418).
- Chartjunk banned: no grid (optional faint horizontal behind dense CPI bars only), no top/right spines, no bar edges/hatch/gradient/shadow/3D, direct-label over legend. `savefig(format="pdf", bbox_inches="tight", pad_inches=0.02)`; rasterize only the 1b scatter (dpi=600).

## Priority (if only 3): **Fig 2 (the law) > Fig 1 (the operator + CFG hook) > Fig 3 (dose law = honesty + reach).** Cut order beyond 3: Fig 5 first UNLESS the leverage-AUROC upgrade is used (then ~level with Fig 4); Fig 4 to appendix at full size.
