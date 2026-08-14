# FINDINGS — BindCraft demonstration: IF-confidence is worse than random for hotspot triage (field-question #3)

**Script:** `src/bindcraft_triage.py`. **Output:** `results/bindcraft_triage.csv`. SKEMPI interface positions
(141 complexes, 5742 positions, 325 hotspots). Capture@k = fraction of a complex's experimental hotspots
captured in the top-k interface positions by a selection rule; complex-clustered bootstrap, seed 20260803.

## Question
BindCraft (leading one-shot binder pipeline) hard-codes a 4 Å interface FREEZE forbidding inverse folding at
the interface — an implicit admission that IF confidence is untrustworthy there. Give it a measurement: at a
matched budget of k interface positions per complex, which selection rule captures the most experimental
hotspots — the model's CONFIDENCE, free geometry (ΔSASA / contact), the learned KL, or random (= the freeze
treated uniformly)?

## Result
| budget | confidence | ΔSASA | KL | burial | nbr(contact) | random(=freeze) |
|---|---|---|---|---|---|---|
| @3 | **0.064** [.039,.092] | **0.233** [.176,.297] | 0.219 | 0.214 | 0.139 | 0.089 |
| @5 | **0.125** (below rand) | 0.342 | 0.310 | 0.292 | 0.233 | 0.138 |
| top-25% | 0.293 | 0.494 | 0.543 | 0.466 | 0.444 | 0.241 |

## Reading
**Two measured facts turn the BindCraft hook from rhetoric into a result.** (1) Ranking interface positions
by IF **confidence captures FEWER hotspots than random** at tight budgets (@3: 0.064 vs 0.089; @5: 0.125 vs
0.138) — so a designer who trusts IF confidence to prioritise the interface does worse than chance, which
*justifies* BindCraft's decision to freeze the interface rather than trust IF there. (2) But **free ΔSASA
captures ~3× more hotspots than confidence** at the same budget (0.233 vs 0.064 @3), well above random — so
freeze-then-prioritise-by-geometry beats both trusting confidence and the uniform freeze. The field's implicit
hack is right about confidence and improvable with free geometry. (KL ≈ ΔSASA here too, consistent with the
learned-frustratometer result.)
