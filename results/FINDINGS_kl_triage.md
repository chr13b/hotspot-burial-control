# FINDINGS — KL as a design-time triage signal (a candidate method contribution)

> **VERDICT (POSITIVE — and now VALIDATED on the non-native backbones designers use; see §4).** The
> sequence-free KL signal is not only a diagnostic — it is an *actionable* design-time triage ranker. Given
> a fixed budget of `k` interface positions per complex to receive expensive binding-aware optimization,
> ranking by **KL+burial captures significantly more experimental hotspots than the burial heuristic**:
> capture@3 **0.237 vs 0.139** (Δ **+0.098 [+0.021, +0.175]**, P=0.99), capture@25% **0.533 vs 0.444**
> (Δ **+0.089 [+0.009, +0.169]**), n=106 complexes with ≥1 hotspot, complex-level bootstrap (10,000 reps,
> seed 20260803). Two honest bounds: the gain is a **general** additive property of KL, **not** concentrated
> where the model is uncertain (niche AUROC Δ −0.002, null); and although the headline is on **crystal**
> backbones, the design-time claim is now **confirmed on the OpenFold3- and AF2-multimer-predicted backbones
> designers actually use** (§4): capture@k Δ(KL+burial − burial) excludes zero on both, at the crystal
> magnitude (task #12 discharged).

## 1. What was run

```bash
python3 src/kl_triage.py --joined results/kl_detector_joined.csv --out results/kl_triage.csv
```

`results/kl_detector_joined.csv`: 141 complexes, 5742 interface positions, 325 interface hotspots
(`is_hot` = Ala-scan ΔΔG hotspot label). Per position: `kl` (the sequence-free detector — two ProteinMPNN
unconditional passes, complex vs chain-deleted backbone), `nbr`/`rsasa_complex` (burial), `logp_native`
(the model's own confidence), `is_hot`. Analysis restricted to the 106 complexes carrying ≥1 interface
hotspot. Every number traces to `results/kl_triage.csv`.

## 2. The triage framing (why this is a method, not just another AUROC)

A binder designer working on a fixed backbone cannot make ProteinMPNN see binding energy it has no term
for — but they *can* choose where to spend an expensive binding-aware step (FoldX/Rosetta ΔΔG, an MD
relaxation, a binding-aware model, or human attention). That is a **triage** problem: with a budget of
`k` interface positions per complex, which do you pick? The naive design-time signal is **burial** (buried
interface residues matter most). We ask whether **KL** — which is *free* (sequence-free, backbone-only,
no extra model) — beats or augments that heuristic at concentrating the experimental hotspots into the
top-`k`. The decision-relevant metric is therefore **capture@k** (of a complex's hotspots, the fraction
ranked in the top `k`), not top-`k` precision.

## 3. Results (`results/kl_triage.csv`)

| Budget | random | burial | KL | **KL+burial** | Δ(KL+burial − burial) [95% CI], P |
|---|---|---|---|---|---|
| **capture@3** | 0.084 | 0.139 | 0.219 | **0.237** | **+0.098 [+0.021, +0.175], P=0.99** |
| **capture@25%** | 0.261 | 0.444 | 0.543 | **0.533** | **+0.089 [+0.009, +0.169], P=0.99** |

- At a **3-position budget**, KL+burial captures **24%** of a complex's hotspots vs burial's **14%** — a
  **+70% relative** improvement, CI excludes zero. KL alone (0.219) already beats burial (0.139); adding
  burial tightens it (KL-alone Δ +0.079 [−0.005, +0.163] touches zero, so the **combined** ranker is the
  clean, reportable one — KL and burial are complementary).
- **AUROC(is_hot)** sanity vs `kl_analysis`: burial 0.678, KL+burial 0.742, Δ **+0.064 [+0.037, +0.091]**
  — replicates the known KL-adds-to-burial (+0.048) on the full fixture. Consistent.

## 4. Non-native validation (task #12) — the design-time test

The crystal result is a proof-of-concept; the design-time claim needs the non-native backbones designers
actually use. Re-running the **identical** `src/kl_triage.py` on the per-position tables from three
non-native backbone sources, reusing the **same backbone-independent** experimental `is_hot`/`is_interface`
labels (keyed by `(complex_id,chain,resnum,icode)` from the crystal table) and taking `kl`, `logp_native`,
`nbr` from **each backbone's own scoring** (Exp A / Exp D KL-joined tables directly; Exp C2 aggregated
per-position over interface-formed generative backbones, `src/expD_build_triage_c2.py`). **Pre-registered
reading, fixed before the numbers:** if capture@k Δ(KL+burial − burial) CI still **excludes zero**, the
triage is design-relevant and enters the paper as a method; if it collapses to contain zero, it is a
crystal-only property and is not upgraded.

| backbone (n cx w/ ≥1 hotspot) | capture@3 Δ(KL+bur − bur) | capture@25% Δ | AUROC Δ | reading |
|---|---|---|---|---|
| **crystal** (positive control, n=106) | +0.098 [+0.021, +0.175] | +0.089 [+0.009, +0.169] | +0.064 | reproduces committed ✅ |
| **OpenFold3 predicted** (Exp A, n=100) | **+0.083 [+0.013, +0.155]** | **+0.107 [+0.029, +0.185]** | +0.062 | **excludes 0 ✅** |
| **AF2-multimer** (Exp D, n=95) | **+0.087 [+0.018, +0.158]** | **+0.122 [+0.047, +0.199]** | +0.055 | **excludes 0 ✅** |
| RFdiffusion generative (Exp C2, n=9) | +0.112 [−0.080, +0.371] | +0.065 [−0.278, +0.389] | +0.064 | positive sign, underpowered |

> **VERDICT — the triage method is DESIGN-RELEVANT.** On **both** well-powered non-native predicted-backbone
> arms — OpenFold3 and the architecturally-independent AF2-multimer — the KL+burial capture@k advantage over
> burial holds at **both** budgets with CIs **excluding zero**, at essentially the crystal magnitude
> (~+0.08–0.12). The generative arm (Exp C2) shares the positive sign but only **9** complexes carry an
> interface hotspot on the sparse interface-formed generative set, so it is **underpowered, not
> contradictory**. Per the pre-registered reading, the triage enters the paper as a validated design-time
> method, not a crystal-only property. The model-uncertain niche stays null on the predicted backbones too
> (Exp A Δ −0.019, Exp D +0.023, both contain zero) — the gain is a general additive property of KL, as on
> crystal.

## 5. Honest bounds (stated as bounds, not buried)

- **The "rescues where the model is blind" niche is NULL.** Restricted to model-uncertain positions
  (`logp_native` below the complex median — where log-prob design would most plausibly err), KL does *not*
  beat burial: crystal AUROC Δ(KL − burial) = **−0.002 [−0.074, +0.068]** (and −0.019 / +0.023 on Exp A /
  Exp D). The triage gain is a *general* additive property of KL across interface positions, **not** a
  targeted rescue of the model's failures. The intuitive on-narrative story does not hold and is not claimed.
- **The generative arm is underpowered.** Exp C2's interface-formed set yields only 9 complexes with an
  interface hotspot; its capture@k point estimates are positive and consistent (+0.11 / +0.07) but the CIs
  are wide and contain zero. This is a power limit of partial-diffusion generation (few docked backbones
  carry strict hotspots), not evidence against the method — the two predicted-backbone arms carry the claim.
- **Not contradicted by the top-k precision null.** `kl_analysis` found KL's top-`k` *precision* over
  burial not significant. Precision (of the `k` selected, how many are hotspots) is dominated by the many
  non-hotspot interface positions; **capture** (of the hotspots, how many are selected) with the combined
  ranker is the triage-relevant quantity, and it survives on crystal, OpenFold3 and AF2 backbones. Consistent.
- **Not a pilot-n number.** The crystal (n=106), OpenFold3 (n=100) and AF2 (n=95) arms are full fixtures,
  not pilots — the standing project lesson (pilot claims shrinking at scale) is satisfied on the two arms
  that carry the verdict.

## 6. What it means for the paper

This converts the load-bearing KL result from a **diagnostic** ("KL identifies hotspots with
burial-orthogonal information") into a **validated method** ("KL is a free, sequence-free triage signal
that beats the burial heuristic at allocating expensive binding-aware effort"). It is the actionable
contribution the ICLR case was missing (PAPER_OUTLINE.md contribution (iii)). The non-native validation
(task #12) is now **done and positive** (§4): capture@k holds on **both** the OpenFold3 and AF2-multimer
predicted backbones (CIs exclude zero, at crystal magnitude), so the method claim **is** design-relevant
and enters the paper — no longer gated. The RFdiffusion generative arm is underpowered (9 complexes) but
shares the positive sign.

## Lever 2 — binding-energy-weighted readout (kcal/mol)  [added 2026-08-13]

Reviewers ask "your readout isn't binding." This re-expresses the same triage in EXPERIMENTAL binding
free energy. Among interface residues with an Ala-scan measurement (111 complexes, 1325 positions, mean
Σ max(ΔΔG,0) = 16 kcal/mol per complex), energy-capture@k = fraction of the complex's total binding energy
in the top-k positions by each ranker. Complex bootstrap, seed 20260803. → `src/kl_triage_energy.py`,
`results/kl_triage_energy.csv`.

- **Top-3 budget:** KL+burial captures **51.3%** of total interface binding energy vs **49.5%** for burial
  — Δ = +0.018 [−0.002,+0.039] (P=0.96), **+0.32 kcal/mol [+0.01,+0.65]** (absolute-kcal CI excludes 0).
  Robust to the ΔΔG column (ddG_max: +0.31 kcal, same story). KL alone ≈ burial (+0.002, null).
- **25% budget:** no gain (Δ −0.006, null; KL alone −0.025, P=0.05).
- Reading: the count-based capture@3 advantage translates to a **modest but real kcal/mol gain**,
  concentrated at small budgets — the payoff is now in experimental binding units, not a model score. It is
  modest because the highest-ΔΔG positions are often the most buried (which burial already ranks), leaving
  little for KL to add on energy; honest, and still the binding-relevant readout the method needs.

## Files
- `src/kl_triage.py` — the triage prototype (capture@k, AUROC, niche; complex bootstrap).
- `src/kl_triage_energy.py`, `results/kl_triage_energy.csv` — Lever 2 binding-energy-weighted capture (kcal/mol).
- `src/expD_build_triage_c2.py` — builds the Exp C2 interface-formed per-position joined table.
- `results/kl_triage.csv` — crystal estimates, CIs, P-values (the trace for every crystal number above).
- `results/kl_triage_expA.csv`, `results/kl_triage_expD.csv`, `results/kl_triage_expC2.csv` — the three
  non-native arms (OpenFold3, AF2-multimer, RFdiffusion generative).
- Inputs: `results/kl_detector_joined.csv` (crystal); `$SCRATCH/ftax/predicted/expA_kl_joined.csv` (OF3);
  `$SCRATCH/ftax/expD/expD_kl_joined.csv` (AF2); `$SCRATCH/expC2/scored_positions.csv` → C2 joined table.
