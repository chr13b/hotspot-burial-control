# HANDOFF — paste this into a fresh Claude Code session

Open a new session in this directory and paste the block below. Everything else is in `BRIEF.md`
and `CLAUDE.md`.

---

```
Read BRIEF.md and CLAUDE.md in full before writing any code.

We are testing one claim: the staged backbone→sequence pipeline cannot reach interface hotspots.
Inverse-folding models maximise p(sequence | backbone) with NO binding-energy term, and hotspots
are frequently FRUSTRATED (buried polars, strained rotamers, entropically expensive aromatics that
buy affinity rather than stability) — so at exactly the positions that make a binder a binder, the
model's mode is the wrong residue and the right one is in the tail. Factorisation then turns one
joint choice into k independent tail draws, N_hot = exp(Σδ_i/T).

Phase 0 is inference-only on CPU and it decides everything. ProteinMPNN is ~1.7M params; 345
complexes is minutes, not hours.

Do these in order and STOP after step 5.

STEP 0 — FETCH THE UNRESOLVED PREPRINT FIRST, BEFORE ANY CODE.
bioRxiv 10.64898/2026.05.09.722041, "Redesign selective protein binders using contrastive
decoding." A prior sweep read metadata only. It reportedly frames ProteinMPNN's blindness to
target side chains at the interface as a decoding problem — same diagnosis, different fix. If it
already contains the burial-matched analysis or the commitment-ordering result, STOP and tell me.
Report exactly what you could and could not fetch.

STEP 1 — PHASE 0, THE BURIAL-MATCHED CONTROL. This is the whole experiment.
Data: SKEMPI 2.0 (~7,000 experimental ΔΔG_bind over ~345 solved complexes). Hotspot label:
alanine-substitution ΔΔG > 1 kcal/mol; ALSO report the strict ΔΔG > 2 threshold that ProBID-Net
used, for direct comparability. Null label: |ΔΔG| < 0.25.
Model: public ProteinMPNN checkpoint. Compute per-position native log-probability conditioned on
the BOUND complex backbone, teacher-forced on the true rest of the sequence. Average over ≥8
decoding orders and report the spread — a result inside decoding-order variance is not a result.
THE MATCHED-PAIR DESIGN IS THE EXPERIMENT: pair each hotspot position with a non-hotspot interface
position FROM THE SAME COMPLEX with relative SASA within ±0.05, the same secondary-structure
class, and neighbour count within ±1. Report the PAIRED log-probability difference with a
COMPLEX-LEVEL bootstrap (complexes are the independent unit, not positions).
Note the direction of the confound: buried positions are where inverse folding is MOST confident,
so an uncontrolled comparison HIDES the effect rather than inventing it. ProBID-Net's published
0.334 vs 0.472 is exactly that uncontrolled comparison.
PRE-REGISTERED KILL (F0): if the burial-matched gap's 95% complex-level bootstrap CI contains
zero, ProBID-Net's headline is a burial artifact and the factorisation tax dissolves. THIS IS
PUBLISHABLE — report it as a finding, not a failure.

STEP 2 — THE CAUSAL DISCRIMINATION, same pass.
ProBID-Net attributes the gap to conformational DYNAMICS, not frustration. Test both: frustration
proxies (buried-polar fraction, rotamer strain, monomer-vs-complex local energy) against dynamics
proxies (crystallographic B-factors, predicted flexibility). Does the residual post-matching gap
track frustration or dynamics?
PRE-REGISTERED KILL (F1): burial-controlled partial Spearman between inverse-folding log-odds and
SKEMPI ΔΔG_bind ≥ 0.35 → the model is not blind to binding energy and the mechanism is refuted.

STEP 3 — PHASE 1, N_hot.
Compute the constellation cost at T=0.1. Do BOTH: direct measurement by sampling K sequences and
counting full-constellation recoveries (valid only where N_hot is small enough to observe), AND
the analytic product exp(Σδ_i/T) for the large cases. REPORT THE DISCREPANCY where both work —
that discrepancy IS the positional correlation the analytic form assumes away, and reporting it is
better than pretending it is zero.
PRE-REGISTERED KILL (F2): median log10 N_hot < 2 AND the burial-matched gap CI contains zero.

STEP 4 — Note for the record that PDB training leakage runs AGAINST this hypothesis (the model has
seen these complexes), so a positive result is conservative. Say so explicitly in the write-up.

STEP 5 — Write results/FINDINGS.md: what you ran, exact commands, raw numbers with uncertainty,
which falsifiers fired, and a one-line verdict SUPPORTED / REFUTED / INCONCLUSIVE derived strictly
from the pre-registered thresholds. If INCONCLUSIVE, state exactly what was underpowered.

Do NOT move a falsifier after seeing a number. Do NOT start Phase 2 — it needs a GPU this machine
does not have, and it depends on an unverified assumption about MultiFlow's released code. An
honest null is valid and expected; here it is a paper in both directions.

Two standing rules from this project's lineage: (1) a negative from any search or filter is only as
good as a positive control run through the same path — two agents here were saved by exactly that
after a silently-broken full-text operator returned zero for every query and produced four false
negatives; (2) cite only URLs you actually fetched.
```

---

## If Phase 0 passes — Phase 2 on Sherlock

First verify the implementation assumption:

```
Read the released MultiFlow code and confirm that the discrete (CTMC) and continuous (SE(3) flow)
corruption processes are coupled ONLY through a shared time index, with independently specifiable
rate functions. BRIEF.md §4 depends on this. If they are entangled, report before proceeding.
```

Then:

```bash
cd /mnt/c/Users/chris/Desktop/python_projects/personal_projects/factorization-tax
git init && git add -A && git commit -m "factorization tax: brief + phase 0/1 results"
gh repo create factorization-tax --private --source=. --push
```

On Sherlock:

```
Read BRIEF.md, CLAUDE.md and results/FINDINGS.md. Phase 0 passed. Run Phase 2 from BRIEF.md §4:
measure commitment times t*_seq and t*_str on a public MultiFlow-family checkpoint (0.5-crossing of
normalised agreement between the endpoint prediction and the realised value — token argmax agreement
for sequence, TM-score or contact-map overlap for structure). Then sweep the discrete unmasking rate
exponent AT INFERENCE with the continuous schedule fixed, no retraining, and measure
hotspot-restricted recovery: recovery at SKEMPI hotspots minus recovery at burial-matched controls.
PRE-REGISTERED KILLS: F3 — t*_seq ≤ t*_str + 0.05 under the default schedule, stable across ≥3
seeds and ≥2 length bins, means the diagnosis is factually wrong. F4 — a full-range sweep moving
hotspot-restricted recovery by less than seed-to-seed SD means the knob is inert.
Note that this contradicts MultiFlow's published finding that purity (confidence-ordered) unmasking
is beneficial. The burden of proof is on us; the comparison must be scrupulously matched.
```

**Sherlock notes:** one GPU (`--gres=gpu:1`) suffices — this is sampling, not training. Phases 0 and
1 need no data movement to Sherlock.

---

## Do not forget

**BRIEF.md §3** before writing any paper text. ProBID-Net already published the *uncontrolled*
phenomenon (0.334 vs 0.472) and attributed it to dynamics; MultiFlow already has decoupled schedules
*and* reports purity unmasking as beneficial; StaB-ddG already occupies the Tsuboyama + SKEMPI
fixture. Lead on the burial-matched control, `N_hot`, and the commitment ordering.
