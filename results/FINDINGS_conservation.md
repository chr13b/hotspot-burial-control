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
complexes (308 hotspots); 33 chains >1022 res skipped (ESM-2 context limit). SEED=20260803.

```
python3 src/skempi_conservation.py --stage score   --out results/skempi_conservation_positions.csv
python3 src/skempi_conservation.py --stage analyse --out results/skempi_conservation_positions.csv
```

## Result — leverage SURVIVES conservation, and they are complementary

| CPI (beyond the listed controls) | value | verdict |
|---|---|---|
| **conservation** \| geometry | **+0.0057 [+0.0016, +0.0123]** | conservation is a real baseline (ADDS) |
| leverage −L \| geometry | +0.0072 [+0.0037, +0.0126] | reference (the paper's number) |
| **leverage −L \| geometry + conservation** | **+0.0063 [+0.0034, +0.0106]** | **SURVIVES** — barely attenuated |
| conservation \| geometry + leverage | +0.0042 [+0.0013, +0.0087] | conservation ALSO adds beyond L |

Drop-3-influential on the headline (1EMV, 1KBH, 4BFI): **+0.0040 [+0.0025, +0.0055] — SURVIVES.**

**Interpretation.**
1. **Conservation genuinely predicts hotspots** beyond geometry (+0.0057, CI>0) — so the control is
   non-trivial, exactly the baseline a reviewer would insist on.
2. **The mixed derivative survives it**: adding conservation to the geometry control barely moves leverage's
   contribution (+0.0072 → +0.0063), and it survives dropping the 3 most influential complexes. The binding
   signal in the mixed derivative is **not** a repackaging of sequence conservation.
3. **They are complementary, nearly orthogonal**: each adds beyond the other, and Spearman(conservation, −L) =
   **−0.137** — leverage and conservation are measuring largely different things. (The weak *negative* sign is
   itself telling: conserved positions are not preferentially the binding-leverage ones.)

## Bottom line
Leverage adds binding information **beyond geometry, beyond the one-pass log-odds (W2), beyond a second IF
model family (ESM-IF1), and beyond evolutionary conservation** — the full standard hotspot-prediction feature
set. This is the control that turns the decomposition from a control study into a result, and it lands in the
strong direction (survives, not attenuates-to-null). → results/skempi_conservation.csv,
skempi_conservation_positions.csv.
