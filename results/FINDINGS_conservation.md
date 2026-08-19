# Ceiling-raiser (audit W3): does leverage add beyond CONSERVATION + geometry?

**Why.** The paper controls leverage against cheap geometry (burial+nbr+ΔSASA). But every *published* hotspot
predictor uses **evolutionary conservation**, and our own catalytic section shows conservation (ESM-2) finds
functional sites where inverse-folding confidence is blind. A skeptical reviewer will demand the symmetric
control for binding. This closes it: it upgrades the claim from "beyond cheap geometry" (a strawman) to
"beyond the **standard hotspot feature set** (geometry + conservation)."

**Method.** Score every SKEMPI interface position's sequence conservation with ESM-2 (150M) negentropy
(−H of its 20-aa distribution; higher = more conserved), reusing the catalytic pipeline. Join to the committed
leverage positions and run the project CPI estimator (cross-fit, conditional-permutation within geometry
strata, complex-clustered bootstrap). 166,520 scored positions → 13,160 interface positions matched over 343
complexes (308 hotspots). Features are z-scored before the CPI (matching `position_level_cpi`; `cpi()` is not
scale-invariant), so CPI(L|geometry) here == the §4 table's +0.00485. The conservation comparison **excludes**
135 interface positions / 9 hotspots on chains shorter than 10 residues (MHC peptides, short inhibitors): a
PLM's per-position entropy there is a systematic outlier (mean −1.64 vs −0.85 nats) and not a meaningful
conservation estimate — a pre-outcome structural exclusion (chain length); those positions retain valid
ProteinMPNN leverage. Only 4 chains exceed the ESM-2 1022-residue context limit. SEED=20260803.

```
python3 src/skempi_conservation.py --stage score   --out results/skempi_conservation_positions.csv
python3 src/skempi_conservation.py --stage analyse --out results/skempi_conservation_positions.csv
```

## Result — leverage SURVIVES conservation, and they are complementary

| CPI (beyond the listed controls, z-scored) | value | verdict |
|---|---|---|
| **conservation** \| geometry | **+0.00338 [+0.00177, +0.00514]** | conservation is a real baseline (ADDS) |
| leverage −L \| geometry | +0.00484 [+0.00328, +0.00657] | reference (== §4 table +0.00485) |
| **leverage −L \| geometry + conservation** | **+0.00510 [+0.00309, +0.00756]** | **SURVIVES** — undiminished |
| conservation \| geometry + leverage | +0.00412 [+0.00144, +0.00803] | conservation ALSO adds beyond L |

Drop-3-influential on the headline (1EMV, 1KBH, 4BFI): **+0.00362 [+0.00226, +0.00505] — SURVIVES.**
Concentration: the top-3 complexes contribute 27% of the estimate, top-10 = 52% (131/343 negative) — not
dominated by a few complexes.

**Actionable ranker (the payoff against the standard feature set).** Cross-fit hotspot AUROC:
geometry+conservation 0.714 → **+|L|_rms 0.731 (Δ +0.0161 [+0.0042, +0.0290], P=0.996)**; +(−L_ala) 0.737
(Δ +0.0220 [+0.0051, +0.0393]). **Both** leverage variants add significantly on top of geometry+conservation —
a larger, more robust effect than against bare geometry (§8's +0.0125), and the direct answer to "you defended
against a baseline the field does not use."

**Interpretation.**
1. **Conservation genuinely predicts hotspots** beyond geometry (+0.0034, CI>0) — so the control is
   non-trivial, exactly the baseline a reviewer would insist on.
2. **The mixed derivative survives it**: adding conservation to the geometry control leaves leverage's
   contribution undiminished (+0.00485 → +0.00510, same z-scored convention), and it survives dropping the 3
   most influential complexes. The binding signal in the mixed derivative is **not** a repackaging of conservation.
3. **They are complementary, nearly orthogonal**: each adds beyond the other, and Spearman(conservation, −L) =
   **−0.137** — leverage and conservation are measuring largely different things. (The weak *negative* sign is
   itself telling: conserved positions are not preferentially the binding-leverage ones.)

## Bottom line
Leverage adds binding information **beyond geometry, beyond the one-pass log-odds (W2), beyond a second IF
model family (ESM-IF1), and beyond evolutionary conservation** — the full standard hotspot-prediction feature
set. This is the control that turns the decomposition from a control study into a result, and it lands in the
strong direction (survives, not attenuates-to-null). → results/skempi_conservation.csv,
skempi_conservation_positions.csv.
