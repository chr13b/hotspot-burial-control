# BRIEF — The factorization tax at interface hotspots

> Self-contained scientific brief. A reader who has seen none of the prior work should be able to
> run the whole thing from this file. Written 2026-07-30.

---

## 1. The claim, in one sentence

The staged **backbone → sequence** pipeline that dominates binder design imposes a sampling barrier
that grows exponentially in the number of energetically load-bearing interface positions — so at
the positions that make a binder a binder, **the right residue lives in the tail of the model's
distribution**, and every downstream filter is sorting among designs that never contained the
chemistry.

---

## 2. Why this could be true

### 2.1 The conditioning set is missing the energy term

Inverse-folding models maximise `p(sequence | backbone geometry)` on native pairs. **Nothing in
that objective references binding free energy.** For most positions this is harmless — the
stability-optimal residue and the native residue coincide.

At interface **hotspots** they systematically diverge. Alanine-scanning biology (Clackson & Wells
1995; Bogan & Thorn 1998) establishes that a handful of residues carry most of ΔG_bind, and those
residues are frequently **frustrated**: buried polars, strained rotamers, entropically expensive
aromatics that exist because they buy *affinity*, not because they stabilise the monomer. So at
exactly the positions that matter, the model's mode is the wrong residue.

### 2.2 Factorisation turns one joint choice into *k* independent tail draws

Designers sample inverse folding at `T ≈ 0.1` and take modes. If the native hotspot residue at
position *i* has log-probability deficit `δ_i` relative to the mode, and a functional interface
needs `k ≈ 3–6` such positions **simultaneously**, then

```
    N_hot  =  exp( Σ_i δ_i / T )
```

is the expected number of draws to recover the constellation. **If `N_hot` is 10⁴–10⁸, no amount of
oversampling recovers native-grade interfaces** — and the entire in-silico filtering stack is
operating downstream of a sampler that cannot reach the answer.

### 2.3 "Joint" models inherit the tax through their schedule

In a coupled discrete–continuous flow, each channel has a **commitment time** `t*`: the time at
which the model's prediction of the final value stops changing (0.5-crossing of normalised
agreement between `x̂₁(t)` and the realised `x₁` — token argmax agreement for sequence, TM-score or
contact-map overlap for structure).

**If `t*_str < t*_seq`,** the fold topology and interface geometry are fixed while the sequence is
still mostly mask — the model is *choosing the shape first and fitting chemistry to it*, which is
the staged pipeline with extra steps and no explicit factorisation to point at.

**The standard confidence-ordered ("purity") unmasking heuristic makes this worse:** it unmasks
easy, stability-determined positions first, so the frustrated hotspots are decided **last**, against
a backbone that can no longer move to accommodate them. The prescription is the opposite ordering —
decide the highest-influence discrete variables while the continuous channel is still hot.

---

## 3. Prior art — read this before designing anything

Fetched and verified by an adversarial sweep on 2026-07-30.

### The paper that already published the phenomenon

**ProBID-Net** — [PMC11575592](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11575592/) (*Chem. Sci.*
2024). Using the MIX hotspot set (hotspot = interfacial residue with alanine-scan ΔΔG > 2 kcal/mol;
**440 hotspot vs 1902 non-hotspot**), they measured sequence recovery:

> **0.334 at hotspots vs 0.472 at non-hotspots.**

So *"sequence design models recover hotspots worse"* is **already measured and published.** Two
things make this an opening rather than a wall:

1. **They attribute it to conformational dynamics, not frustration.** The causal mechanism is
   contested and untested.
2. **They do not control for burial.** — and burial is exactly the confound that could produce the
   whole gap (§5.1).

### The published finding this work contradicts

**MultiFlow** — [arXiv 2402.04997](https://arxiv.org/abs/2402.04997). Verified in full text: it
already trains with **decoupled time schedules** for sequence and structure — *"we have freedom to
arbitrarily sample with any combination of (t, t̃)"* — but spends that freedom only on conditional
inpainting, **never on which modality commits first.** It also **uses purity (confidence-ordered)
unmasking and reports it as beneficial**, and samples at temperature 0.1.

So §2.3 directly contradicts a published, load-bearing design choice. That is a feature: ML venues
reward a measurement that overturns a specific choice. It also means the burden of proof is on you,
and the comparison must be scrupulously matched.

### Adjacent, and one unresolved scoop risk

- **StaB-ddG** — [arXiv 2507.05502](https://arxiv.org/abs/2507.05502) (ICML 2025). Occupies the
  exact fixture (Tsuboyama 776,298 + SKEMPI 2.0) and parameterises binding as a difference of
  folding energies. Motivated by binding-data scarcity versus copious folding data — overlapping
  premise. **Every paper in this space must distinguish itself from this one.**
- **Refolding-limitations** — [PMC13147959](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13147959/)
  (*Protein Science* 2026). The canonical statement that self-consistency oracles are biased.
- Unmasking-order work exists but is **language-domain only**:
  [2510.05725](https://arxiv.org/abs/2510.05725), [2606.00295](https://arxiv.org/abs/2606.00295),
  [2605.24697](https://arxiv.org/abs/2605.24697).
- **⚠ UNRESOLVED — fetch this first.** bioRxiv `10.64898/2026.05.09.722041`, *"Redesign selective
  protein binders using contrastive decoding."* It reportedly frames ProteinMPNN's blindness to
  target side chains at the interface as a **decoding problem** — the same diagnosis, a different
  fix. The sweep read **metadata only** and did not fetch the full text. **If this already contains
  the burial-matched analysis or the commitment-ordering result, the project changes. Resolve before
  writing code.**

### What is genuinely unshipped

1. **The burial-matched control.** ProBID-Net's raw 0.334/0.472 gap is precisely the uncontrolled
   comparison; nobody has separated *"hard because buried"* from *"hard because frustrated."*
   **This is the strongest and cleanest contribution, and it is also the cheapest experiment.**
2. **`N_hot` as a computed expected-draws number.** Recovery *rates* exist; nobody converts them
   into the constellation cost where the rhetorical punch lives.
3. **Measuring `t*_str` vs `t*_seq`** and reallocating discrete commitment ahead of structural
   commitment.
4. **Frustration as the tested causal mechanism**, against ProBID-Net's dynamics explanation.

---

## 4. The experiment

**Phase 0 is inference-only, runs on CPU, and decides everything.**

### Phase 0 — the burial-matched control (laptop, hours)

**Data:** SKEMPI 2.0 — roughly 7,000 experimental ΔΔG_bind measurements (ITC, SPR, fluorescence)
across ~345 complexes with solved structures. Hotspot label: alanine-substitution ΔΔG > 1 kcal/mol
(**also report the strict ΔΔG > 2 threshold ProBID-Net used**, for direct comparability). Null
label: |ΔΔG| < 0.25.

**Model:** public ProteinMPNN checkpoint (~1.7M params — CPU inference on 345 complexes is minutes,
not hours). Compute per-position native log-probability conditioned on the **bound complex**
backbone, teacher-forced on the true rest of the sequence. Average over ≥8 decoding orders and
report the spread.

**The matched-pair design — this is the whole experiment.** Pair each hotspot position with a
non-hotspot interface position **from the same complex** having:
- relative SASA within ±0.05,
- the same secondary-structure class,
- matched neighbour count (within ±1).

Report the **paired** log-probability difference with a **complex-level bootstrap** (complexes are
the independent unit, not positions).

> **KILL — F0.** If the burial-matched hotspot-minus-control log-probability gap has a 95%
> complex-level bootstrap CI containing zero, then **ProBID-Net's 0.334/0.472 is a burial artifact**,
> the factorisation tax dissolves, and the mechanism is refuted.
>
> **This outcome is publishable.** *"The hotspot recovery gap is explained by burial"* is a genuine
> correction to a published result and belongs at MLSB or TMLR. **Both directions are a paper — which
> is the main reason to run this first.**

**Run in the same pass — the causal discrimination.** ProBID-Net attributes the gap to
conformational *dynamics*. Test that against *frustration*:
- Frustration proxies: buried-polar fraction, rotamer strain, monomer-versus-complex local energy.
- Dynamics proxies: crystallographic B-factors, predicted flexibility, ensemble variance.
- Does the residual (post-burial-matching) gap track frustration or dynamics?

> **KILL — F1.** Burial-controlled partial Spearman between inverse-folding log-odds and SKEMPI
> ΔΔG_bind ≥ 0.35 → the model is **not** blind to binding energy, and §2.1 is refuted.

### Phase 1 — `N_hot`, the constellation cost (laptop, hours)

Compute `N_hot` per complex at T = 0.1. **A methodological trap to handle explicitly:** the analytic
product `exp(Σδ_i/T)` assumes positional independence, which is false. So do both:
- **Direct measurement** by sampling K sequences and counting full-constellation recoveries — valid
  only where `N_hot` is small enough to observe.
- **Analytic product** for the large cases.
- **Report the discrepancy** on the overlap where both work. That discrepancy *is* the positional
  correlation, and reporting it honestly is better than assuming it away.

Headline: median log₁₀ `N_hot`, and the burial-matched paired gap from Phase 0.

> **KILL — F2.** Median log₁₀ `N_hot` < 2 on held-out complexes **and** the burial-matched gap CI
> contains zero → the factorisation is not costly where claimed.

### Phase 2 — commitment ordering (**GPU → Sherlock**, only if Phase 0 passes)

Public MultiFlow-family co-design checkpoint. Measure `t*_seq` and `t*_str` as defined in §2.3.
Then sweep the **discrete unmasking rate exponent at inference with the continuous schedule fixed**
(no retraining), on held-out complexes with a real binder chain. Metric: **hotspot-restricted
recovery** — recovery at SKEMPI-defined hotspots *minus* recovery at burial-matched controls. This
metric exists because plain sequence recovery is degenerate (many sequences fold to one backbone);
experiment tells you where it is *not* degenerate.

> **KILL — F3.** `t*_seq ≤ t*_str + 0.05` under the default schedule, stable across ≥3 seeds and ≥2
> length bins → the diagnosis is factually wrong; joint models already decide sequence first.
>
> **KILL — F4.** A full-range sweep of the discrete rate moves hotspot-restricted recovery by less
> than seed-to-seed SD → the knob is inert.

**Verify before relying on it (the sweep flagged this as an implementation assumption):** in the
released MultiFlow code, the discrete (CTMC) and continuous (SE(3) flow) corruption processes must
be coupled only through a shared time index with independently specifiable rate functions. **Read
the code before committing to Phase 2.**

---

## 5. Pitfalls that will produce a false positive

1. **Burial — and it cuts *against* naive expectation.** Hotspots are buried, and buried positions
   are where inverse folding is **most confident**. So an uncontrolled comparison will show hotspots
   with *lower* NLL and **hide** the effect. This is the analogue of the molecular-size confound
   measured elsewhere in this project's lineage (apparent docking advantage of −0.25 kcal/mol
   reversing to +0.30 once size was matched). Matched pairs within the same complex, always.
2. **Native amino-acid identity.** Trp, Arg and Tyr are hotspot-enriched and have distinctive
   priors. Include wild-type and mutant identity as fixed effects; report within
   hydrophobicity-matched subsets.
3. **PDB training leakage runs *against* the hypothesis.** The model has seen these complexes, so a
   positive result is **conservative**. Say so — it is a rare case where leakage helps your honesty.
4. **Assay heterogeneity.** SKEMPI pools ITC, SPR and fluorescence. Report strict and loose ΔΔG
   thresholds; restrict to 293–303 K and pH 6–8 for the headline.
5. **Positional independence in `N_hot`.** Handled explicitly in Phase 1 — do not let the analytic
   product stand alone.
6. **Decoding-order variance.** ProteinMPNN's autoregressive order changes conditional
   probabilities. Average over ≥8 orders and report the spread; a result inside decoding-order
   variance is not a result.

---

## 6. What each outcome means

| Outcome | Meaning |
|---|---|
| Burial-matched gap survives, tracks frustration, `N_hot` large | **The result.** The factorisation tax is real and mechanistically located. Phase 2 becomes a method paper contradicting a published choice. |
| Gap survives but tracks **dynamics**, not frustration | ProBID-Net's attribution was right and yours is wrong — but the burial-matched measurement still corrects the literature. Solid short paper. |
| **Gap collapses under burial matching** | **ProBID-Net's headline is a burial artifact.** The idea dies, and the correction is publishable. MLSB or TMLR. |
| Gap survives, Phase 2's `t*` ordering is already sequence-first | Diagnosis half-wrong; the tax is real but the schedule fix is unavailable. Report the measurement. |

**An honest null is a valid result — and here, unusually, it is a paper in both directions.**

---

## 7. Venue

- **Best realistic case: NeurIPS / ICML / ICLR main track.** This is a clean method paper — a
  sampling-schedule contribution to joint discrete–continuous generative models, anchored to
  experimental ΔΔG, that **overturns a published design choice**. That combination is what ML venues
  reward.
- **Floor, if the burial control kills the gap:** *"the hotspot recovery gap is a burial artifact"* —
  **MLSB** (NeurIPS workshop) or **TMLR**. A real service to the field.
- **Domain alternative:** *Chemical Science* (ProBID-Net's home) or *JCIM*.

---

## 8. Compute

**Phases 0 and 1 are inference-only on a ~1.7M-parameter model over ~345 complexes — CPU, minutes
to hours.** No GPU, no training. Only Phase 2 (MultiFlow sampling sweeps) needs a CUDA GPU, which
this machine does not have. See `CLAUDE.md`.

---

## 9. Provenance

Output of a third directed-evolution ideation pass (`innovator` skill, research profile) plus an
adversarial prior-art sweep, 2026-07-30. Full lineage across three passes — including two plateaued
topics, one empirically refuted idea, and several corrections to the author's own claims — in
`notes/lineage-provenance.md`.

**Two standing rules inherited from that lineage:**

- **A negative from any search or filter is only as good as a positive control run through the same
  path.** Two independent agents in this project hit a silently-broken Europe PMC full-text operator
  returning zero for *every* query; four "strong negatives" were false, and only a control caught it.
- **Weak negatives are unresolved, not cleared.** The contrastive-decoding preprint in §3 is exactly
  such a case — metadata only. Fetch it first.
