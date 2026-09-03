# Confidence is not competence: inverse-folding models know binding in the mixed derivative

*Working prose draft, 2026-08-14. Expands notes/PAPER_OUTLINE.md (Spine B). Every quantitative claim carries
a `→ file.csv` trace to a committed result. Sections marked ⟨PENDING …⟩ await a running analysis.*

---

## Abstract

Staged binder design — generate a backbone, then inverse-fold a sequence — is believed to stumble at
protein–protein interface *hotspots*, and a prominent report (ProBID-Net) quantifies this as inverse-folding
recovery of 0.334 at hotspots versus 0.472 elsewhere, attributed to dynamics. The phenomenon is real but
misread: the model's binding knowledge was being read from the wrong place. **Confidence and competence are two
different derivatives of the same inverse-folding likelihood.** A residue's *confidence* is the *diagonal* term
— a scalar summary of the bound-conditioned distribution — and measures fold-stability constraint; its binding
*leverage* is the *mixed second derivative*, the response to ablating the binding partner,
`[log p(a|complex)−log p(wt|complex)] − [log p(a|monomer)−log p(wt|monomer)]`, and measures the binding effect.
Because the partner-ablated structure is *determined* by the complex, leverage is computable from the structure
but **not** from the bound distribution alone — so every scalar the field reads off the model (recovery,
confidence, the complex-vs-monomer KL) is a lossy projection, blind to binding **by construction, not by
failure**. This is no vacuous identity: holding the bound distribution fixed, much of the leverage spread
survives — a flexible learner (the best of gradient boosting and random forests) trained on the *entire* bound
distribution recovers only ~37% of the leverage, so **~63% is irreducible from the bound distribution** in both
inverse-folding families (~62% even when wt identity is added);
the binding-specific component provably requires the partner-ablated second pass. And the consequence is measurable and large. On our main fixture (SKEMPI natural complexes)
confidence ranks hotspots at or barely above chance across five architectures and adds nothing beyond geometry
(position-level conditional predictive impact 0.000), while the mixed derivative adds binding information **beyond
geometry, beyond the standard one-pass log-odds readout, and beyond evolutionary conservation** — the full
feature set of published hotspot predictors (mutation-level CPI +0.059; Spearman with experimental ΔΔG −0.30;
per-position ~5× the best scalar, where confidence is conditionally independent) — and the whole result
replicates in three further inverse-folding families (ESM-IF1: CPI +0.035, Spearman −0.26; PiFold: +0.050, −0.33; MIF: +0.058, −0.27). Three consequences follow. It is **actionable**: the mixed derivative is among the strongest single features
for ranking interface hotspots (AUROC 0.69, on par with the learned KL detector at 0.68 and above geometry's
0.66 and confidence's 0.51), and adds
**+0.016 AUROC [+0.004, +0.029]** on top of the full feature set published predictors already use —
geometry *and* conservation. It is a **dose law** of backbone accuracy
— the signal survives ≤0.5 Å of error and then collapses, at ~1 Å for ProteinMPNN and ~1.5 Å for ESM-IF1 — and
because the sensitivity is to the *backbone the derivative is read from*, not to the network, the fragility
(though not its exact threshold, which we measure to be model-dependent) is a prediction every method built on
the same mixed derivative (BA-Cycle, RedNet, StaB-ddG) inherits on predicted backbones. And it reaches the **second** mixed
derivative: the model's partner-ablated pairwise couplings predict experimental binding *epistasis*. The
published deficit is then largely a corollary — mostly a burial confound that attenuates sharply under matching, across five architectures plus ProBID-Net's own released
model — and the blindness generalizes from binding to *catalytic* residues (confidence blind, sequence
conservation predicts). The leverage operator is BA-Cycle (Jiao et al. 2024); our contribution is the
decomposition, the identifiability result and its *measured* non-vacuity, the first beyond-geometry (and
beyond-conservation) control on any inverse-folding binding signal, and the feature-class law. The recipe is
general and already familiar: the mixed derivative is the model's **classifier-free-guidance direction** (the
binding partner as conditioner), and to read an un-trained quantity off a conditional generative model one
ablates the conditioner and reads that direction — not the conditional marginal the field has been mistaking
for competence. And the direction is *actionable*: biasing a frozen off-the-shelf ProteinMPNN's interface
logits by `+α·L` raises the binding-leverage an independent model (ESM-IF1) assigns the sampled residues —
monotonically, while a matched-magnitude random direction *lowers* it — with native recovery preserved.

## 1. Introduction

The dominant paradigm for designing protein binders is staged: a generative model proposes a backbone, and an
inverse-folding model assigns a sequence by maximising `p(sequence | backbone)`. This factorisation is
convenient but it omits the very quantity a binder exists to optimise — there is no binding-energy term
anywhere in `p(sequence | backbone)`. The concern this raises is sharpest at *interface hotspots*: the small
set of residues that contribute most of the binding free energy, and which are frequently *frustrated* —
buried polar residues or strained rotamers that are locally unfavourable for the monomer's fold but bought
because they pay off in binding. If inverse folding optimises fold-compatibility, the reasoning goes, it
should systematically miss exactly these residues, and a hotspot should sit in the tail of the model's
distribution rather than at its mode. A prominent measurement appears to confirm this: ProBID-Net reports
inverse-folding recovery of 0.334 at hotspots versus 0.472 elsewhere, and attributes the gap to dynamics.

There is a confound that no prior analysis controls, and it runs opposite to intuition. Hotspots are, on
average, *more deeply buried* than other interface residues, and burial is precisely where inverse folding is
*most* confident and accurate. An uncontrolled hotspot-vs-rest comparison therefore mixes a putative binding
effect with a large burial effect of the opposite sign — so the naive comparison **hides** any real deficit
rather than inventing one. Controlling burial is not a detail; it is the experiment.

We make five contributions.

**(i) The Confidence–Leverage Decomposition (a theorem).** A model's per-residue *confidence* is the *diagonal*
term of the inverse-folding log-likelihood — a scalar functional of one conditioning — and estimates
fold-stability constraint; a residue's binding *leverage* is the *mixed second derivative*, the per-substitution
response to ablating the partner (the BA-Cycle operator of Jiao et al. 2024). Confidence is blind to leverage
*by construction* — an identical bound distribution yields identical confidence but arbitrary leverage; we
verify this is non-vacuous *and measured*: a flexible learner over the entire bound distribution recovers only
~37% of the leverage, leaving **~63% irreducible from `P`** in both families (→ r2_leverage_from_P.csv). This
yields a **feature-class law**: on natural complexes a scalar read off the bound distribution *alone* recovers
little beyond cheap geometry — confidence, negentropy, and the scalar KL all sit **at or below the CPI
estimator's calibrated false-positive floor** (+0.0007, the score of a placebo feature that is a deterministic
function of geometry) — whereas the *two-pass* mixed derivative does not reduce: per interface position it adds
+0.0048, ~7× the floor (CI disjoint from every scalar's, robust to a nonlinear geometry control), and on SKEMPI at the mutation level CPI +0.059
with Spearman −0.30 against experimental ΔΔG, robustly (a *dissociation between feature classes*, not a claim
that `L` is a large-magnitude predictor). The model knows binding on natural complexes; the knowledge
was invisible to every scalar readout the field has used. → leverage_decomposition.csv, nugget_cpi.csv.

**(ii) Confidence is not competence — a property of inverse folding, with a practical consequence.** The
diagonal is blind across five architectures (interface-hotspot AUROC 0.50–0.54; conditionally independent of
hotspot-ness given geometry, position-level CPI 0.000; the mutation-level confidence CPI is small but nonzero,
+0.010, so the blindness claim is specifically about *position-level* scalars). This measures the field's implicit BindCraft interface-freeze: ranking
interface positions by confidence captures *fewer* hotspots than random (capture@3 0.064 vs 0.084, overlapping intervals; a trend, not yet significant), while free
ΔSASA captures ~3× more (0.233) — so rank by geometry — or, better, by the mixed derivative itself, the
among the strongest single features for interface triage (AUROC 0.69, level with the learned KL detector, vs confidence's 0.51; §4) — not by confidence. De-novo designs corroborate the positive
of (i) with even the scalar distribution: it beats substitution baselines (0.615), dissociates stability from
binding, and adds +0.018 beyond an all-atom rotamer-repacked occlusion baseline, reproduced by a non-parent
scorer. → xmodel_confidence.csv, baseline_audit.csv, bindcraft_triage.csv, bennett_occlusion_allatom.csv, bennett_nonparent.csv.

**(iii) The blindness generalises across function types.** On catalytic residues (M-CSA), structure-conditioned
confidence is blind (within-amino-acid-type AUROC ≈ 0.44–0.50) while a sequence language model's conservation
predicts them (0.77) — a dissociation surviving composition, burial, and chain-truncation controls.
Inverse-folding confidence is blind to functional importance in general, not only binding. → catalytic_audit.csv.

**(iv) The published deficit is largely a burial confound.** Under a pre-registered burial-matched matched-pair
design (matching within-complex on relative SASA, secondary-structure class, and neighbour count), the
crystal-backbone hotspot deficit attenuates sharply across five inverse-folding architectures — most of the gap
is burial, though a residual persists for two of them (MIF, PiFold) on the strict tier — and ProBID-Net's own
released voxel-CNN reproduces its published deficit and then dissolves it under joint burial-and-composition
matching. We offer the matched-pair design as a reusable diagnostic *protocol*. → probid_gap_estimators.csv,
composition_confound.csv.

**(v) The residual tax lives in the conditioning set.** On the predicted backbones designers actually use —
from two architecturally-independent structure predictors — a burial-matched deficit reappears, and the two
predictors' per-complex deficits agree (ρ = 0.57): the *same* complexes are hard under both. This agreement
survives residualising on interface burial (partial ρ = 0.53), so it is not a recursive burial effect, and it
is absent on noised-crystal backbones at matched distance — it tracks *independent reconstruction*, not
distance-from-native. → deficit_burial_residualize.csv. And on those same predicted backbones the *mixed
derivative itself* survives: recomputed on the OpenFold3/AF2 structures with geometry taken from the predicted
backbone, CPI(L | geometry) is +0.039/+0.032 (69–84% of the matched crystal value, CI>0), so the binding signal
is design-time-usable, not a crystal artifact — while the confidence readout on the same backbones degrades into
the deficit, exactly as the decomposition predicts. → FINDINGS_leverage_predicted.md. Two competing mechanisms —
a low-temperature constellation cost and a commitment-ordering schedule — are separately measured and refuted.

## 2. Setup and pre-registration

**Fixtures.** Our primary fixture is SKEMPI 2.0, from which we take single-mutation binding data and define a
hotspot as an alanine-scan ΔΔG_bind > 1 kcal/mol (and a strict variant > 2, ProBID-Net's threshold), with a
null set of |ΔΔG| < 0.25. Hotspots are rare — 2.4% of interface positions (327/13,401), label entropy 0.115
nats — which sets the scale for the effect sizes below. Because SKEMPI complexes are crystal structures of *natural* complexes, we add two
independent fixtures of different character: **Bennett-2023 de-novo designed binders**, which carry
experimental site-saturation binding measurements over four targets and constitute a true design-regime test;
and **AB-Bind**, antibody–antigen ΔΔG over 27 analysed complexes, a second SKEMPI-class fixture with distinct
biophysics.

**The matched-pair protocol.** The core of the burial analysis is a within-complex optimal 1:1 matching of
hotspots to null residues on relative complex SASA (±0.05), secondary-structure class, and neighbour count
(±1). Effects are aggregated by complex-level bootstrap; every seed is fixed (20260803) and reported with its
bootstrap replicate count. → PREREG.md, PREREG_knows_where.md, PREREG_bennett_hardening.md.

**Statistical stance (multiple comparisons).** Headline claims are *pre-registered* (P1–P3 and the PREREG files
above), so their tests are confirmatory rather than selected after seeing the data; the placebo floor calibrates
the CPI estimator's false-positive rate directly (a feature must clear the floor, not merely exceed zero); and
every interval is a complex-clustered bootstrap. The many secondary and cross-architecture analyses are reported
without a global family-wise correction and are treated as descriptive support, not independent confirmatory
tests.

**Models.** Five inverse-folding architectures span the design space: ProteinMPNN (vanilla and soluble
variants), ESM-IF1 (a 142M-parameter GVP-transformer), PiFold (a one-shot GNN), MIF (masked inverse folding),
and ProBID-Net (a voxel CNN). A positive control gates every scoring path.

## 3. Confidence is not competence

The quantity a practitioner is most tempted to trust — the model's own confidence that the native residue
belongs at a position — is useless for locating hotspots. On SKEMPI interface positions, ProteinMPNN's
per-residue confidence attains an AUROC of 0.538 for hotspots, barely above chance, and for a fixed-budget
triage (the top-3 interface positions per complex) it captures *fewer* hotspots than random selection
(0.064 vs 0.084). → baseline_audit.csv, confidence_antipredicts.csv.

One might object that any single scalar can be rescued by combining it with structure. It cannot. We apply a
conditional predictive impact (CPI) test: cross-fit a model of hotspot-ness on cheap geometry (burial,
neighbour count, ΔSASA), then measure how much predictive information confidence adds when its
geometry-conditional information is destroyed by permutation within geometry strata. The estimate is
**0.000 [−0.0003, +0.0003]**: confidence is conditionally independent of hotspot-ness given structure. By the
same test, ΔSASA adds real information (+0.013) and the KL detector adds a token amount (+0.002). → nugget_cpi.csv.

This is not a quirk of one network. Across all five architectures, interface-hotspot confidence-AUROC lies in
0.50–0.54 (ProteinMPNN 0.538, ESM-IF1 0.517, PiFold 0.499, MIF 0.509, ProBID-Net 0.536), each 0.15–0.19 below
what trivial burial alone achieves. *Confidence is not competence* is thus a property of inverse folding, not
an artefact of a particular model. → xmodel_confidence.csv.

What does predict hotspots is free geometry. Burial alone reaches 0.689; ΔSASA — the partner-contact area,
computable without any neural network — reaches 0.673; and a cheap-geometry combination reaches 0.734. The
obvious learned alternative is the sequence-free divergence between the model's complex- and
monomer-conditioned distributions (a KL detector), which one might hope captures partner-induced frustration
beyond geometry. It captures a *small* one: under a combiner-free conditional test, KL adds CPI = +0.002
[+0.0006, +0.0034], P=0.998 beyond full geometry, and its within-geometry-stratum AUROC is 0.60 (vs 0.50
leakage) — a genuine learned-frustratometer signal, but ~6× smaller than ΔSASA's contribution and not worth
the network as a standalone ranker. It is the same signal §4 reads *at the placebo floor*: the scalar KL is a
contraction of the leverage vector (`E_P[L]` up to a constant) — the single best scalar summary of the two-pass
signal — and even it captures only a sliver, a hair above the floor on this matched 5,742-position sample and
sitting on it on the full 13,401-position sample, an order of magnitude below the full mixed derivative it
summarises. (We are careful here about readout: the unfitted z-sum ΔAUROC we first
used has a −0.021 noise floor — it penalises adding *any* feature — so the earlier "KL adds ≈0 / actively
hurts" reading measured the combiner, not KL; we retire that estimator, as we do the sibling
ΔAUROC-over-one-hot in §4.) So KL is a learned frustratometer that *largely* recapitulates the classical
geometry, adding only a small real increment beyond it. → kl_geometry_control{,_predicted}.csv, nugget_cpi.csv.

Finally, none of this is specific to SKEMPI. On AB-Bind (antibody–antigen ΔΔG), the identical pattern holds:
confidence-AUROC for hotspots is 0.560 (chance; CI includes 0.5), burial 0.728 and ΔSASA 0.604 predict, and
confidence adds +0.008 (indistinguishable from zero) over full geometry. → abbind_nugget.csv.

## 4. The Confidence–Leverage Decomposition: the model knows binding — in the mixed derivative

The results so far are corrective, and every quantity the field has read off these models — recovery,
confidence, the complex-vs-monomer KL — is a *scalar summary* of the distribution. There is a specific reason
they all fail at hotspots, and stating it turns the corrections into a theorem.

**The decomposition.** Write the model's per-position information as two orthogonal terms of the
inverse-folding log-likelihood's interaction expansion in the partner. **Confidence** is the *diagonal* term —
a scalar functional of the single bound-conditioned distribution `p(·|X_complex)` (log p(native), negentropy) —
and estimates positional fold-stability constraint. **Leverage** is the *mixed second difference*, the
per-substitution response to *ablating the partner*:
`L_i(a) = [log p(a|X_complex) − log p(wt|X_complex)] − [log p(a|X_monomer) − log p(wt|X_monomer)]`,
which by the standard binding thermodynamic cycle estimates −ΔΔG_bind up to a single unknown positive scale
(the model's effective inverse temperature). This is exactly the **classifier-free-guidance / contrastive-decoding**
direction: writing a guided logit as `logit(·|X_complex) + α·[logit(·|X_complex) − logit(·|X_monomer)]`, the
bracket equals `L` up to a position-constant `wt`-reference shift that the softmax absorbs — the guided
distribution is identical — whereas the KL contraction below is a shift the softmax does *not* kill. The binding
partner is the conditioner and confidence is the conditional marginal, so the general recipe for reading an
*un-trained* quantity off a conditional generative model is to ablate the conditioner and take this mixed
derivative, not the marginal. Read information-theoretically, each bracket is a difference of pointwise mutual
informations between residue identity and partner presence (against a common reference marginal, which cancels),
and `L` contrasts that quantity at `a` against `wt`. The scalar KL is one contraction of this vector:
`KL(P‖Q) = E_{a∼P}[L(a)] + [log P(wt) − log Q(wt)]` (identity verified to 1e-6) — the offset is
*position-dependent*, so the scalar detector is a lossy, position-shifted summary of `L`, not a rescaling of it. A methodological aside that
also motivates L: the per-position softmax normaliser `log Z_i` contaminates confidence but *cancels* in L
(each bracket is within-conditioning), so L is better-posed. **Confidence is blind to leverage by
construction:** two positions with an identical bound distribution have identical confidence yet can differ
arbitrarily in L — and this is not hypothetical. We measure how much of the mixed derivative the *whole* bound distribution can determine: a flexible learner
(the max over gradient boosting and random forests, out-of-sample under complex-clustered cross-validation)
trained on the full 20-vector `P` recovers **only ~37%** of the leverage — R²(L_rms | P) = 0.37 [0.34, 0.40]
(ProteinMPNN), 0.36 [0.34, 0.39] (ESM-IF1) — so **~63% is irreducible from `P`** in both families (~62% even
when wt identity is added, the fair `φ(P, wt)` class since confidence itself uses `log P(wt)`); a *linear*
readout recovers less than half, R² ≈ 0.15. The recoverable part is the one-pass complex log-odds *vector*; the
*scalar* one-pass magnitude alone recovers far less, and the irreducible majority is the partner-ablation term
that lives in `Q`. A second, threshold-free reading agrees: within deciles of confidence the interquartile
range of `|L|` is **1.09× [1.06, 1.13]** the overall — so conditioning on confidence removes essentially none
of the leverage spread. → r2_leverage_from_P.csv, conf_decile_leverage.csv.

We state the decomposition as a proposition; it is short, and it is what turns the corrections above into a
theorem rather than a list.

> **Proposition 1 (Confidence is blind to leverage).** Fix an inverse-folding model and an interface
> position `i`. Let `P_i = p(·|X_complex)` and `Q_i = p(·|X_monomer)` be its bound- and partner-ablated
> per-position distributions, and define
> - *confidence* `C_i = φ(P_i)` — **any** scalar functional of the bound distribution alone (recovery
>   `1[argmax P_i = wt]`, log-likelihood `log P_i(wt)`, negentropy `−H(P_i)`);
> - *leverage* `L_i(a) = [log P_i(a) − log P_i(wt)] − [log Q_i(a) − log Q_i(wt)]`.
>
> For position-level statements the 20-vector `L_i` is reduced to a scalar by a declared map (`L_rms`, or the
> alanine leverage `L_i(→A)`). Then **(i) [cycle]** `L_i(a)` estimates `−ΔΔG_bind(i, wt→a)` up to a single
> unknown positive scale — a *monotone surrogate*, not a calibrated kcal/mol reading — via the binding
> thermodynamic cycle; **(ii) [non-identifiability]** `L` is not a function of `P`: `Var(L_i | P_i) > 0`, so two
> positions with the same bound distribution `P_i = P_j` have identical confidence `φ(P_i) = φ(P_j)` for
> **every** functional `φ`, yet differ in leverage. Hence for any `P`-only estimator, `inf_φ E[(ΔΔG_bind −
> φ(P))²] ≥ E[Var(ΔΔG_bind | P)] > 0` — an error floor that `L`, which uses information beyond `P`, is not bound by;
> **(iii) [readout, not model]** `X_monomer` is a deterministic function of `X_complex` (delete the partner's
> atoms), so `L` is computable — by a second, partner-ablated forward pass — yet by (ii) is **not a function of
> `P`**. The blindness is a property of the *readout*, not the model: the binding term is in the weights,
> reachable only by re-querying on `X_monomer`, and no transform of the bound distribution recovers it.

*Proof.* (i) Under the model's Boltzmann reading of `log p`, the bound-minus-unbound difference of the `wt→a`
log-odds identifies with `log(K_a^{a}/K_a^{wt}) = −ΔΔG_bind/kT_model`; the per-position normaliser `log Z_i`
cancels because each bracket lies within a single conditioning. It is an *estimate*, not an equality — our
measured Spearman(`L`, experimental ΔΔG_bind) = −0.30 is what makes "estimates" the honest verb — and it
carries three assumptions we state rather than hide: `kT_model` is taken *position-independent* (needed for `L`
to *order* positions; an assumption we flag, testable as per-burial-stratum calibration slopes); the readout is
the *sequence-free marginal* `p(·|X)`, a mean-field approximation that buys decoding-order invariance; and
`X_monomer` is the *bound* backbone with the partner deleted, so `L` estimates the **interaction component** of
ΔΔG_bind — not monomer refolding or conformational relaxation. (ii) The equality `φ(P_i)=φ(P_j)` is
definitional; the content is that `P` does **not** determine `Q`, which would hold iff the map `X ↦ P` were
injective — and it is not. That `Var(L|P) > 0` is measured directly: a flexible learner (gradient boosting) trained on the full 20-vector
`P` recovers only R²(L|P) ≈ **0.37** out-of-sample in both families (max over gradient boosting and random
forests), so **~63% of the mixed derivative is irreducible from `P`** even under a nonlinear readout, and only
~15% under a linear one (`r2_leverage_from_P.csv`). And the floor holds against *ground truth*, not just the
`L` proxy: regressing **experimental** ΔΔG_bind directly on `P` (the →Ala substitutions, the readout comparable
to `L(→A)`) leaves **88% irreducible** for ProteinMPNN and **90%** for ESM-IF1 (R² = 0.12 [0.06, 0.16] and 0.10 [0.05, 0.14]) — even more than the mixed derivative's own 63–66% (66% for the strictly comparable →Ala readout), as expected once
experimental measurement noise enters, with substitution identity alone explaining ~1.5% of ΔΔG (both floors hold on a second model's distribution). → r2_ddg_from_P.csv, r2_ddg_from_P_esmif.csv. (iii) `X_monomer` is `X_complex` with the partner deleted, a deterministic map; `Q =
model(X_monomer)` costs a second forward pass, which by (ii) no function of `P` reproduces. ∎

The empirical sections instantiate the proposition — and they trace a single arc: `L` is the model's
classifier-free-guidance direction (this section), it is among the best *training-free* readouts for locating
interface hotspots (§8), and it is exactly the mixed derivative Proposition 1 proves confidence cannot see. The
feature-class law below is (ii) measured on natural
complexes; the no-go for scalar readouts is its immediate corollary; and §5 (ProBID-Net), §8 (BindCraft) and
the KL detector are three scalars-of-`P` the field met separately, each blind for exactly this reason.

**The feature-class law (on the main fixture, natural complexes).** On SKEMPI, the mixed derivative adds
binding information beyond cheap geometry where every scalar summary does not. Per interface position
(conditional predictive impact over burial+neighbours+ΔSASA, 13,401 interface positions; confidence here is the
same diagonal §3 finds conditionally independent):

| feature (all functionals of the same distribution) | CPI beyond geometry |
|---|---|
| *placebo floor* — a deterministic function of geometry | *+0.0007* — the estimator's false-positive floor (pure noise ≈0) |
| **confidence** — the diagonal | **+0.0002 [−0.0003, +0.0006]** — below the floor; conditionally independent |
| negentropy — one-pass | +0.0009 [+0.0003, +0.0015] — at the floor |
| scalar KL — a contraction of L | +0.0009 [+0.0003, +0.0016] — at the floor |
| **leverage L** — the mixed derivative | **+0.0048 [+0.0033, +0.0065]** — ~7× the floor; CI disjoint from every scalar; robust to a nonlinear geometry control (+0.0047); survives dropping the 3 most influential complexes |

On the identical 5,742-position sample §3's confidence test uses, the same ordering holds and sharpens —
leverage +0.0092 [+0.0062, +0.0124], ~5× the scalar KL, while confidence stays conditionally independent (CI
spans 0). → leverage_nugget_match.csv. For an interpretable scale: on that same sample the zero-shot mixed
derivative contributes **~71%** of what the partner-contact area (ΔSASA) — an explicit geometric measurement of
the interface — contributes beyond burial and neighbour count (ΔSASA +0.0129; → nugget_cpi.csv). And because
the hotspot label is rare (base rate 2.4%, entropy 0.115 nats), these CPIs are small in absolute terms but not
in relative: leverage's +0.0048 is **4.2%** of the label's entropy — against 0.2% for confidence, 0.8% for the
scalar KL — an order of magnitude more than either scalar of the bound distribution (the comparison to ΔSASA,
~71%, is given above on its own sample). → effect_size_normalized.csv. At the mutation level the effect is large: Spearman(L, experimental ΔΔG_bind) = **−0.30**, and CPI(L | geometry)
= **+0.059 [+0.046, +0.073]**, surviving controls from substitution similarity (BLOSUM, volume, hydropathy) and
from L's own scalar components. → leverage_decomposition.csv, FINDINGS_leverage.md. Critically, the second pass
earns its keep against the *standard* zero-shot readout — the one-pass complex log-odds
`logP(mut|complex) − logP(wt|complex)`: controlling for it *and* geometry, leverage still tracks ΔΔG
(partial Spearman = **−0.147 [−0.190, −0.108]**, P<0=1.0); and the reverse holds too — the one-pass readout
retains signal after controlling for leverage (−0.094 [−0.142, −0.047]). So the two passes are *not* redundant:
each carries binding signal the other misses, and in particular the partner-ablation pass is not subsumed by
the standard one-pass readout (we do not claim one dominates — the paired difference is not significant). The
two-pass *structure* is what carries the signal, and this is its cleanest statement: the monomer pass on its
own is inert (Spearman(monomer log-odds, ΔΔG) = **+0.04**), and the one-pass and full leverage, while
correlated (Pearson **+0.64**), are genuinely distinct — yet subtracting the individually-inert monomer pass
*improves* the tracking of ΔΔG (one-pass Spearman −0.26 → leverage −0.30 on this sample). Leverage
works precisely because the second pass *corrects* the first for the fold-stability constraint that the
one-pass score conflates with binding. → w2_monomer_inert.csv. This scopes the
feature-class law precisely: it is about *position-level* scalars of P (recovery, confidence, entropy); a
*per-substitution* one-pass readout legitimately carries constraint and substitution-similarity information,
but the *binding-specific* increment requires the second pass. The direction is robust, not a residue-type
artifact — Spearman(L, ΔΔG) is negative in 62 of 72 complexes with ≥15 measured mutations and in 18 of 19
wild-type residue types (n-weighted −0.26), and holds at −0.25 on alanine substitutions alone (n=2,327), so L
is not a side-chain-volume or truncation proxy. → w2_onepass_control.csv. And the increment is not evolutionary
conservation in disguise — the one control every *published* hotspot predictor uses. We score each interface
position's sequence conservation with the **field-standard masked-marginal** ESM-2 estimator (mask the
position, read the model's distribution) — itself the *stronger* baseline, adding **+0.0064 [+0.0028, +0.0113]**
beyond geometry — and leverage's contribution is *undiminished*: CPI(L | geometry) +0.0048 → CPI(L | geometry +
conservation) **+0.0059 [+0.0031, +0.0097]** (surviving the drop of its 3 most influential complexes at +0.0040 [+0.0027, +0.0054],
those three contributing 33% of the estimate). The simpler unmasked estimator agrees
(+0.0051; the two conservation estimators correlate at +0.71), and conservation adds beyond leverage in turn
(+0.0083) — the two are nearly orthogonal (Spearman −0.08 masked, −0.14 unmasked). And the *actionable* payoff
sharpens against the conservation baseline: adding the mixed derivative on top of geometry **and** conservation
(the unmasked-negentropy estimator, for which we ran the hotspot ranker) lifts hotspot AUROC by
**+0.016 [+0.004, +0.029]** (both the |L|_rms and −L(→Ala) variants significant), larger than against geometry
alone. So the mixed derivative adds beyond the **standard hotspot feature set** —
geometry *and* conservation — not merely cheap geometry. → skempi_conservation.csv, skempi_conservation_masked_cpi.csv, FINDINGS_conservation.md. To calibrate the effect
size: L is a *zero-shot* readout of a model never trained on binding, yet it adds **+0.030 interface AUROC**
(0.700→0.730) beyond a *supervised* geometry+substitution baseline fit directly on the binding labels, and on
its own reaches AUROC 0.647 — near that supervised baseline. That a zero-shot readout beats a supervised one is
exactly what the ground-truth floor predicts: with ~88% of experimental ΔΔG irreducible from `P` and ~98% from
substitution identity alone (Prop 1(ii)), there is little in geometry+substitution for a supervised model to
fit — the binding signal is reachable only through the second pass. → leverage_effect_size.csv, r2_ddg_from_P.csv. **This is not a
ProteinMPNN artifact:** it replicates under ESM-IF1 — a GVP-transformer with a native-conditioned (not sequence-free) readout
— where confidence is again blind to hotspots and leverage again adds beyond geometry and beyond every scalar
including confidence *at the position level* (337/344 complexes; mutation Spearman −0.26, CPI +0.035; position
confidence CPI −0.0000). Honesty on the second family: at the *mutation* level ESM-IF1's confidence is less
inert than ProteinMPNN's — it adds (confidence CPI +0.023 [+0.012, +0.033]) and absorbs about half of
leverage's mutation-level increment (leverage +0.035 → +0.018 when confidence is controlled, vs ProteinMPNN's
+0.059 → +0.056); the *position-level* blindness the feature-class law is about holds for both. It replicates a
**third and fourth** time under PiFold (a graph message-passing model; mutation Spearman(L, ΔΔG) =
**−0.33 [−0.39, −0.26]**, CPI(L | geometry) = **+0.050 [+0.039, +0.061]**) and MIF (a masked-inverse-folding
model; Spearman **−0.27 [−0.36, −0.18]**, CPI **+0.058 [+0.045, +0.070]**), each surviving substitution,
confidence and scalar-KL controls, with confidence again position-blind (CPI +0.0000). So the feature-class law
is a property of the inverse-folding *class*, now across **four** architectures — the same panel on which
confidence is blind at hotspots. → leverage_esmif.csv, leverage_pifold.csv, leverage_mif.csv. So the model *does*
know binding on natural complexes — the knowledge was invisible to every scalar readout the field used. The law is
not about a *regime* but a *feature class*: on natural complexes scalar summaries reduce to geometry
(confidence exactly, KL nearly); the mixed derivative does not.

**Corollary 1 (no-go for scalar readouts).** This is Proposition 1(ii) made empirical. The blindness is not special to confidence. Write any scalar the field reads
off an inverse-folding model as a functional `φ(P)` of the bound-conditioned distribution `P = p(·|X_complex)`
*alone* — sequence recovery (an argmax match), confidence (`log P(native)`), or entropy/perplexity. Binding
leverage is a functional of the *pair* `(P, Q = p(·|X_monomer))`, and no functional of `P` alone can express it
wherever `P` does not determine `Q` — which the distribution-matched pairs above show is generic. Empirically
the tiers are distinct — and we read them against a *calibrated* floor. The CPI estimator has a non-zero
false-positive floor: a placebo feature that is a deterministic function of the geometry controls, carrying no
information beyond them, still scores **+0.0007** (a duplicate of ΔSASA; pure noise correctly scores ≈0). →
w_placebo_ladder.csv. Against that floor the scalars of the bound distribution are indistinguishable from
noise-beyond-geometry: confidence **+0.0002** (CI spans zero), one-pass negentropy **+0.0009**, and even
leverage's own P-weighted contraction — the KL detector, algebraically `E_P[L] + const` — **+0.0009**, all at or
below the floor. Only the *two-pass* mixed derivative L(→Ala) clears it decisively: **+0.0048**, ~7× the floor,
CI disjoint from every scalar's, robust to a *nonlinear* (quadratic or cubic) geometry control (+0.0047), and
surviving the drop of its 3 most influential complexes. So collapsing the leverage vector to any scalar of the
bound distribution discards **essentially all** of its conditional signal. (That confidence and sequence recovery track
burial is itself long known — Dauparas et al. 2022, Hsu et al. 2022; what is new here is the *formal*
feature-class law, the conditional-independence control, and that the mixed derivative alone escapes it.) So any method that ranks interface positions by
a scalar summary of the bound distribution is, on natural complexes, a geometry detector in disguise. This is
one statement with several corollaries the field has met separately: ProBID-Net's recovery deficit is a burial
confound (§5), BindCraft is right not to trust interface confidence (§8), and the learned KL detector merely
recapitulates ΔSASA (§8). Only the partner-ablation mixed derivative escapes the reduction.

As a physical check that the mixed derivative behaves like a *shared* interface energy rather than a per-chain
scoring artifact, contacting cross-interface residues have positively correlated leverage magnitudes: Spearman
**+0.094** [0.066, 0.122], P(>0) > 0.999, over 6,391 cross-interface contact pairs. The correlation is modest
per pair — as expected when a hotspot faces a structural scaffold residue — but robustly positive; a binding
energy should be reciprocal across a contact, a per-chain confidence artifact would not be. (We report the
per-*pair* statistic deliberately: a per-complex sum of |L| over both sides is dominated by a shared
interface-size multiplier and is not evidence of reciprocity.) → leverage_reciprocity.csv.

**But this knowledge is fragile to backbone error — a dose law.** The mixed derivative is read off a backbone,
and it does not survive a large perturbation of one. Jittering the crystal backbone to a target interface RMSD
and re-scoring (the monomer inheriting the *same* jitter, so the partner ablation stays clean; σ=0 reproduces
the crystal value exactly, a positive control), CPI(L | geometry) holds — +0.058 at 0.0 Å, +0.059 at 0.25 Å,
+0.047 at 0.5 Å, +0.032 at 0.75 Å — then collapses to +0.002 by 1.0 Å and −0.001 by 1.5 Å (all on the same
2,949-mutation sample), and Spearman(L, ΔΔG) tracks it down (−0.30, −0.29, −0.29, −0.19, −0.08, −0.06). A
re-drawn noise realization at 1 Å reproduces the collapse, so it is not a single-sample artifact. The
binding signal is robust to *accurate* reconstruction and lost under an inaccurate one. **The decisive test is
whether real predicted backbones fall on the surviving or the collapsed part of this curve, and they fall on
the surviving part** (§6, measured directly): on OpenFold3 and AF2-multimer backbones for 140 shared complexes,
CPI(L | geometry) is **+0.039 [+0.027, +0.050]** and **+0.032 [+0.023, +0.041]** — 69–84% of the matched
crystal value, CI clearing the placebo floor, drop-3 robust — i.e. the ~0.5–0.75 Å rung of this ladder, not the
cliff. So leverage is a design-time-usable readout on the accurate predictors designers actually use, and the
earlier inference that it would collapse on predicted backbones (extrapolated from this jitter ladder before the
direct experiment) was too pessimistic and is corrected by measurement. What the staged backbone→sequence
pipeline still misses at hotspots is not this signal but the *confidence* one it actually reads: on the same
predicted backbones the confidence-type recovery readout degrades into a burial-matched deficit (§6, −0.19/−0.23)
while the mixed derivative survives — the two readouts diverge exactly where the decomposition says they must.
Because the jitter collapse is driven by the
*backbone the derivative is read from* — not by anything specific to ProteinMPNN — the same cliff is predicted
for any method reading this mixed derivative off a generated backbone (BA-Cycle, RedNet, StaB-ddG all do) —
the sensitivity is a property of the *input backbone*, shared by any reader of the derivative. **That class
prediction has now been run empirically under a second model family, and it half-holds — we state which half.**
Repeating the identical ladder with ESM-IF1 (a 142M GVP-transformer with a native-teacher-forced conditional
readout) over all 285 fixture complexes / 2,809 mutations, CPI(L | geometry) is **+0.0362 [+0.0273, +0.0452]**
at 0.0 Å (CI excluding zero), **+0.0350** at 0.25 Å and **+0.0266 [+0.0178, +0.0353]** at 0.5 Å,
so *the sub-Ångström survival replicates cleanly in a second architecture*. It then decays — +0.0115 at 0.75 Å,
+0.0177 at 1.0 Å, +0.0020 [−0.0013, +0.0053] at 1.5 Å, +0.0011 at 2.0 Å — but **the collapse arrives later than
ProteinMPNN's**: at 1.0 Å ProteinMPNN is already at the floor (+0.0024, CI touching zero) while ESM-IF1 decays
more slowly, reaching the floor only by 1.5–2.0 Å (+0.0020 [−0.0013, +0.0053] at 1.5 Å). The 1.0 Å rung itself
is unstable across noise realizations and should be read as *straddling* the floor, not as retained signal:
three independent jitter draws on the 200-complex subsample (σ = 0.99/1.00/1.01, the seed being a function of σ)
give +0.0114, +0.0019 and −0.0002 — a draw-to-draw spread (~0.012) the size of the estimates themselves, so the
per-rung bootstrap CIs understate the tail uncertainty and the 0.75-vs-1.0 non-monotonicity is inside
realization variance. **The honest
class claim is therefore: the fragility to backbone error and the survival of accurate reconstruction are
shared; the *threshold* is model-dependent (≈1.0 Å for ProteinMPNN, ≈1.5 Å for ESM-IF1) and must be quoted per
model, not as a universal ~1 Å cliff.** (ESM-IF1's raw Spearman(L, ΔΔG) is markedly more jitter-robust than its
CPI — −0.252 → −0.169 at 1.0 Å versus ProteinMPNN's −0.301 → −0.077 — plausibly because its readout is
teacher-forced on native sequence context that jitter leaves intact; we flag this as an untested hypothesis.)
→ leverage_noise_ladder.csv, leverage_noise_ladder_075full.csv, leverage_noise_ladder_esmif.csv,
leverage_noise_ladder_esmif_{all285,redraw,tail}.csv, FINDINGS_esmif_dose_law.md.

**And the knowledge extends past single effects to their *couplings* — the second mixed derivative.** If
the single-mutant leverage is the model's first mixed derivative (partner ablation × one mutation), the
natural next object is the second: the partner-ablated pairwise coupling `C_ij(a,b)` — the finite change in
the (mutant-vs-wild-type) conditional log-odds at position *i* when position *j* is set to its mutant, read
from the autoregressive likelihood and symmetrised over decoding orders — the partner-ablated analogue, for an
inverse-folding model and *binding*, of the categorical Jacobian that Zhang et al. (2024) use to read
coevolutionary couplings from a protein language model (that a second sequence-difference of the likelihood
measures epistasis is Nambiar 2025's framing). StaB-ddG's
Appendix&nbsp;B establishes, as a theoretical *expressivity* property, that such a folding-energy predictor
*can* represent binding epistasis (unlike the additivity-enforcing predictors it compares against); whether
the likelihood actually *does* is left untested — we measure it. On the 557 SKEMPI double mutants whose two single mutations are *also* measured (so the
experimental coupling `g = ΔΔG_ab − ΔΔG_a − ΔΔG_b` is defined; 61 complexes, swapped-order duplicates merged),
the model coupling tracks the measured epistasis with the cycle-predicted sign, Spearman(C, g) = −0.14
[−0.23, −0.07]; and — the honest test, since neighbouring residues couple trivially — it *survives* controlling
for inter-residue distance, partial ρ = −0.12 [−0.21, −0.05]. On the clean subset of 383 *cross-interface* pairs
(residues on opposite sides of the interface, coupled only through binding, so no monomer subtraction is needed)
the distance-controlled coupling is −0.13 [−0.25, −0.04], and the method-matched CPI(|C| beyond distance,
|g| > 0.5) = +0.017 [+0.007, +0.029] survives dropping its three most influential complexes
(+0.007 [+0.000, +0.016]). It is not a sign-skew artifact — the distance-controlled partial ρ is negative in
*both* sign strata (g < 0 and g > 0), whereas a magnitude-plus-skew artifact would flip the g > 0 stratum
positive. Partner ablation is what surfaces it: for same-side pairs the *un-ablated* coupling
is null (partial ρ = +0.01 [−0.15, +0.15]) while the ablated one carries a weak same-direction signal (−0.12 [−0.25, +0.02], CI touching zero) — subtracting the
intra-fold coupling exposes the binding coupling, exactly as for single mutations. Three positive controls
hold: the coupling magnitude tracks the measured |g| even after controlling for distance (partial ρ = +0.21),
the two decoding directions — read from *disjoint* order subsets — agree (Spearman = +0.61), and the signal is
strongest *within* contacts (−0.16), not an artifact of the contact boundary. (The 14 complexes >800 residues
dropped for memory are a coverage limit, not a detectable bias: on low power their epistasis distribution is not
distinguishable from the retained set — Kolmogorov–Smirnov p = 0.63 — and the effect is if anything *stronger*
in the larger retained complexes. → p3_coupling_biascheck.csv.) Two honesties bound the claim: the
effect is *modest* — about half the single-site leverage's −0.30, and near SKEMPI's own reproducibility floor
(the same physical pair measured twice differs by up to 1 kcal/mol) — and it is a signal about coupling
*magnitude* more than *sign*. Per-pair sign accuracy is near chance overall (0.54) and unreliable against the majority-class
baseline: it edges the baseline on the high-|C| subset (|C|>p75, 0.65 vs 0.62) but falls below it on
large-|g| (0.63 vs 0.69) and the top decile (|C|>p90, 0.68 vs 0.70). What survives is a small chance-corrected sign channel —
controlling for |g| and distance, C still tracks the *direction* of epistasis at partial ρ = +0.08 [+0.01, +0.17],
a few points of real sign information rather than the fifteen a naïve accuracy would suggest. The direction is
nonetheless unambiguous, and it is the first direct empirical test of that expressivity property: what the model knows about
binding reaches past single-residue effects to their epistatic couplings, and lives — at both orders of the
mixed derivative, the first (single-residue ΔΔG) and the second (epistasis) — in the structure of the
distribution, not in the confidence. → p3_coupling.csv, p3_sign_verify.csv,
FINDINGS_p3_coupling.md.

**De-novo designs corroborate — there, even the scalar distribution shows it.** Where selection is
binding-dominated, the signal is accessible to blunter probes too. On Bennett-2023 de-novo binders with
experimental site-saturation mutagenesis (four targets; a sanity control passes exactly — the SSM-excluded
amino acid equals the native in 4137/4137 positions), the per-substitution complex-conditioned distribution
itself ranks substitutions by binding. → bennett_knows_where.csv.

We pre-registered three tests (P1–P3). **(P1)** The model's complex-conditioned distribution ranks the 19
substitutions by whether they retain binding at an interface AUROC of 0.615 [0.601, 0.628], above chance and
above every sequence baseline — BLOSUM62 (0.589), hydropathy (0.579), and volume similarity (0.539) — so it
is not merely a substitution-similarity matrix in disguise. **(P2)** The same model answers the fold-
*stability* question markedly better than the *binding* question: at buried core positions (a stability
positive control) its AUROC is 0.721, versus 0.615 at the interface, a dissociation of +0.107 with
non-overlapping intervals. This is the control the design demands — a model trained on `p(sequence|structure)`
*should* be better at stability than at binding. **(P3)** Conditioning on the partner adds binding
information specifically at the interface: the complex-conditioned distribution beats the binder-alone
distribution by +0.076 [+0.068, +0.084] at interface positions, and by essentially nothing at core and
surface positions, where the partner is irrelevant. → bennett_knows_where.csv.

**The decisive test: beyond all-atom occlusion.** A skeptic's natural objection is that P3's partner-gain is
mere steric *occlusion* — a bulky substitution at a contacted position clashes with the partner and also
abolishes binding, so the model's "binding knowledge" is just a clash detector. We test this with the
strongest occlusion baseline we can build. For every (interface position, substitution) we construct the
substituted side chain in explicit all-atom detail (rdkit ETKDG rotamers), superpose it on the true backbone,
and compute its minimum van-der-Waals clash against the partner *over all rotamers* — i.e. the best steric fit
achievable by repacking, precisely the operation the objection invokes. A pre-registered validity gate passes
(the builder reconstructs native side chains to a median 0.278 Å). The result inverts the objection: after
repacking, **95.1% of substitutions incur zero clash** — occlusion is nearly absent as a mechanism, and the
all-atom clash predicts binding at 0.519, no better than chance. On a geometry baseline that now includes this
all-atom clash together with contact count, ΔSASA and volume (and is *stronger* than the earlier proxy, 0.619
vs 0.587), the model's per-substitution probability still adds ΔAUROC = **+0.0182 [+0.0145, +0.0220]**,
P(>0)=1.000. The model encodes per-substitution binding *energetics* beyond all-atom steric occlusion. (We
report the process in full: an early run mis-implemented the validity gate as a clash-correlation, which
failed for lack of dynamic range; we corrected it to the pre-registered reconstruction gate and remained
blind to the ΔAUROC until that gate passed.) → bennett_occlusion_allatom.csv.

**Not circularity.** Because the SSM parents are themselves ProteinMPNN outputs, one might worry the model is
scoring substitutions around its own mode. A non-parent model — ESM-IF1, which did not generate the designs —
reproduces every component: interface AUROC 0.625, partner-gain +0.079, and the beyond-occlusion signal
+0.016, all with intervals excluding the null. → bennett_nonparent.csv.

**Feature class, not regime.** An earlier reading of these results proposed a *regime* law — that the model's
binding signal is accessible only where selection is binding-dominated (de-novo), reducing to geometry on
natural complexes. Our own main fixture refutes it: the mixed derivative adds +0.059 on SKEMPI (above). What is
true is the *feature-class* distinction: on natural complexes the binding signal is invisible to *scalar*
summaries — confidence is at chance and adds zero beyond geometry (§3), and the scalar confidence readout
traces a **monotone gradient** set by how fold-coupled the interface is. Within one pipeline, across SKEMPI's
own interface classes, hotspot confidence-AUROC rises in the *pre-registered* order — TCR/pMHC 0.430
[0.355, 0.508] → antibody–antigen 0.457 [0.383, 0.518] → protease–inhibitor 0.554 [0.465, 0.615] — and reaches
0.596 [0.567, 0.624] on de-novo binders (Spearman(transience-rank, AUROC) = +1.0; → threepoint_law.csv,
bennett_conf_fork.csv). It is not a burial effect: the order survives burial-residualization (0.413 → 0.426 →
0.551) and runs *opposite* to the per-class burial-AUROC (antibody–antigen is the most buried class yet
low-confidence). Adjacent classes overlap in CI — a monotone trend with de-novo significantly above the two
transient-recognition classes, not four pairwise-significant steps — but every natural class sits at or below
chance and only de-novo clears it, yet the *mixed derivative* carries the binding signal in all of them: the
leverage-AUROC **clears chance in every natural class and beats confidence in each** — 0.641 [0.539, 0.765] (TCR/pMHC), 0.628
[0.544, 0.716] (AB/AG), 0.701 [0.599, 0.811] (protease–inhibitor), each clearing chance — while confidence
climbs the fold-coupling gradient beneath it. The two feature classes *diverge across a controlled biological
axis*: the scalar is regime-dependent and blind, the derivative clears chance regardless of interface type — exactly the split Proposition 1 forces, resolved
along a controlled biological axis: a scalar of `P` can only track the fold-constraint that varies with interface
class, while the mixed derivative tracks the binding that does not. De-novo designs are simply where the signal is accessible to blunter probes
as well: there, even the scalar complex-conditioned distribution adds +0.018 beyond an all-atom occlusion
baseline — and the *mixed derivative* itself adds in this genuine de-novo regime, CPI(L | geometry) =
**+0.011 [+0.008, +0.014]** on 73 de-novo binders with wet-lab site-saturation labels, where the scalar KL is
at the floor (+0.0003, CI spans zero). Here the one-pass complex readout log p(mut|complex) is itself strong
(+0.023), so the *two-pass-specific* increment is the stricter claim: controlling for it, leverage still adds
**+0.0032 [+0.0015, +0.0048]**, rising to **+0.0041 [+0.0023, +0.0060]** once substitution identity (BLOSUM,
volume, hydropathy) is controlled — where, as on SKEMPI, the reverse also holds (+0.0047 [+0.0033, +0.0061]),
so neither pass subsumes the other. The partner-ablation pass carries design-regime signal on *actual* de-novo
binders, complementing R2's predicted-backbone result (§6) on natural complexes, with the scalar contraction
(KL) at the floor throughout. → leverage_bennett_denovo.csv. A methodological note this forces: an earlier AB-Bind analysis reported the per-mutation
distribution "adds nothing" on natural antibody–antigen ΔΔG (ΔAUROC +0.008 over geometry + substitution), but
under the conditional CPI test on the same baseline it adds **+0.031 [+0.015, +0.045]** (and +0.042 beyond
geometry alone); the ΔAUROC readout's own fitted detection floor (≈−0.002) sits below the effect. The result is
**fixture-fragile**, though — under a leaner control set the CPI is +0.009 and spans zero — and AB-Bind's 27
complexes are too few to decide it either way; SKEMPI is where the question is settled. → abbind_bigidea1.csv,
abbind_cpi.csv, leverage_decomposition.csv.

**The blindness generalises beyond binding — to catalytic residues.** "Confidence is not competence" is not
specific to binding hotspots. (The premise that a *functional*-site signal can be disentangled from a
*stability* signal on this M-CSA benchmark is Cagiada et al.'s, 2023 — with sequence statistics plus a
biophysical stability model, not inverse folding; our distinct contribution is the
inverse-folding-confidence-versus-PLM-conservation dissociation under a within-amino-acid-type control, and the
finding that IF confidence is at chance.) On M-CSA catalytic residues, controlling for amino-acid composition by
stratifying *within* amino-acid type, structure-conditioned confidence is blind (within-type AUROC 0.44–0.50 across strata, chance to weakly anti-predictive,
chance) while a sequence language model's conservation predicts them (0.771 [0.723, 0.822]) — a dissociation
of +0.288 [+0.237, +0.338] that survives on monomers alone (ruling out a partner-chain-truncation artifact:
there MPNN is 0.516 [0.429, 0.604], chance) and under an additional within-amino-acid-type burial control (rSASA
and neighbour-count bins): +0.234 [+0.179, +0.290] on all enzymes, and +0.176 [+0.060, +0.293] with burial and
truncation controlled jointly. (The fully-crossed within-complex cell retains only 39 of 119 catalytic residues
and is not significant, +0.14 [−0.13, +0.43]; we report it rather than rest on it.)
Inverse-folding confidence is thus blind to functional importance across function types; what predicts
function is free geometry (for binding) or sequence conservation (for catalysis). We are deliberate about
mechanism: the model's confidence is *blind* (at chance), not actively *frustrated* — the raw anti-prediction
we first observed was an amino-acid-composition and single-chain-truncation artifact, not a determinacy
signal. → catalytic_audit.csv, FINDINGS_catalytic.md. (Methodological note for the appendix: the effect is
invisible to a ΔAUROC-over-amino-acid-identity control, whose detection floor is a within-type AUROC of
~0.55; the correct readout is the within-type AUROC itself.)

**One attempt did not generalise (reported for the record).** A finer *within*-SKEMPI confidence-decay
gradient, binned by binding affinity, is null on 141 complexes; the natural regime does not furnish an
obligate endpoint (it is defined by measurable dissociation), so a transient→obligate gradient is not
constructible here. → confidence_gradient{,_affinity}.csv.

**The named direction is actionable.** If `L` is the classifier-free-guidance direction, the direct test is to
*guide* with it. Biasing a **frozen, off-the-shelf** ProteinMPNN's interface logits by `+α·L` and sampling
(K=64, 271 SKEMPI complexes, pre-registered grid α ∈ {0,0.25,0.5,1,2}) raises the mean binding-leverage a
**different** model (ESM-IF1) assigns the sampled interface residues, monotonically — **−0.20 at α=0 → +0.27 at
α=2**, CI clearing zero from α=1. A random direction of matched per-position magnitude does the *opposite*
(→ −0.51), so the paired L−random gap is +0.13 / +0.24 / +0.45 / **+0.77** (P(>0)=1.0 at every α>0): it is the
**direction**, not the perturbation. Native interface recovery does not fall but slightly *rises*
(0.276 → 0.297; the random arm degrades it to 0.220) and non-interface recovery is flat — the tilt concentrates
probability on residues that are simultaneously more native-consistent *and* higher binding-leverage. This is
anti-circular (a second, architecturally distinct model scores the sequences) but not an independent binding
oracle: both are inverse-folding models and ESM-IF1 leverage is a model proxy for ΔΔG, so connecting the steered
sequences to a physical or experimental binding readout is the natural next step, not claimed here. So the same
mixed derivative the field's decoders already tilt along (RedNet; §8) works as a training-free knob on a model
that was never trained to bind. → cfg_steer.csv, cfg_steer_summary.csv, FINDINGS_cfg_steer.md; pre-registered in
PREREG_cfg_steer.md.

## 5. On crystal backbones, the hotspot gap is a burial artifact

We now return to the published deficit and show, on crystal backbones, that it is a burial confound. The
confound is visible directly: as hotspot strength increases, both sequence recovery and burial rise in
lockstep (recovery 0.347→0.529, relative SASA 0.218→0.080). Under the pre-registered matched-pair design —
pairing each hotspot to a null residue in the same complex at matched relative SASA, secondary-structure
class, and neighbour count — the deficit attenuates to statistical indistinguishability: the matched estimate
is −0.042 [−0.222, +0.129] and a higher-powered regression estimator is +0.059 [−0.051, +0.167] — the deficit
attenuating sharply from its unmatched value. The matching controls relative SASA, secondary structure and
neighbour count but not partner-contact area (ΔSASA) — the cheap feature that carries the most hotspot
information — so we check the residual directly: the SECONDARY-B pairs do carry a ΔSASA imbalance (+0.072
[+0.045, +0.100]), yet adjusting the deficit for it moves it essentially nowhere (−0.042 → −0.028), so the
attenuation is not a residual-ΔSASA artifact. → dsasa_matched_sens.csv. We do not claim it vanishes everywhere: on the strict
matched-pair PRIMARY tier (47 pairs) MIF and PiFold retain a residual recovery deficit whose CI excludes zero
(MIF +0.277 [+0.098, +0.465]; PiFold +0.191 [+0.022, +0.359]), and a two-one-sided-tests check does not certify
equivalence at the ±0.115-nat margin. The honest claim is **strong attenuation — most of the gap is burial —
not proven absence**, and the residual is architecture-dependent. → FINDINGS.md, panel_summary.csv. Tellingly,
MIF and PiFold — the two architectures that retain this residual *recovery* deficit — are two of the four on
which the *mixed derivative* replicates (§4; CPI(L | geometry) = +0.058 and +0.050): within a single model the
confidence-type readout keeps a deficit exactly where the leverage readout still works — the §4 decomposition
seen *within* an architecture, not only across the feature classes.

The strongest form of this test uses ProBID-Net's own released voxel-CNN. Run on our fixture, its port is
faithful (overall interface recovery 0.472, matching its reported non-hotspot number), and its published
hotspot deficit *does* reproduce — but **only in comprehensively-scanned complexes**: pooled it is null
(+0.014 [−0.052, +0.087]), and it appears at −0.113 [−0.208, −0.022] (p=0.007) only among the 18 complexes with
≥5 measured hotspots (the intermediate scan-depth strata are non-monotone) —
the pattern expected if the deficit is real and sparsely-scanned complexes merely lack the power to show it,
not a cherry-picked subset. But it attenuates under confound-matching: matching residue type
turns it positive (+0.120), matching burial gives −0.038, matching
hydrophobicity −0.051, every interval spanning zero. ProBID-Net's deficit is thus a residue-composition and
burial confound — its voxel-CNN has an unusually extreme amino-acid-type dependence (per-type recall spanning
0.17 to 0.98, a *global* property measured across all positions, not fit at hotspots, so matching residue type
is a control rather than a post-hoc adjustment), and hotspots are enriched in the types it recovers worst — not
evidence of binding-specific blindness. → probid_gap_estimators.csv, composition_confound.csv. (We correct an earlier draft of our own
that mislabeled this as an opposite-sign, fixture-specific null; that reading was a complex-averaging
artifact and is withdrawn.) We offer the matched-pair design itself as a reusable diagnostic protocol.

## 6. The tax lives in the conditioning set

If the deficit were purely a benchmark artifact, it should disappear everywhere once burial is controlled. It
does not — it reappears on the *predicted* backbones that designers actually condition on, and there it
behaves like a real, structured signal. On backbones from two architecturally-independent structure
predictors, OpenFold3 and AlphaFold2-multimer, a burial-matched deficit is present (−0.191 [−0.37, −0.004]
and −0.233 [−0.44, −0.035]; the crystal deficit is ≈0). The claim does not rest on either marginal number —
each attenuates when its three most-*supporting* complexes are dropped, though both survive the more principled
drop of the three highest-magnitude-influence complexes (OF3 −0.173 [−0.337, −0.009], AF2 −0.197 [−0.374,
−0.019]) — but on their *agreement*, which is stronger than a correlation: the two predictors' per-complex
deficits correlate at ρ = 0.565 [0.40, 0.71], **and the three complexes that most support the deficit are the
*same set* under both predictors** (1JRH, 1JTD, 1Z7X). The same complexes are hard under both, and the same
complexes carry the effect — a per-predictor artifact could produce neither. → expD_leverage.csv. A
per-predictor memorisation or architecture artifact would produce disjoint deficits; two independent
reconstructions instead agree, per complex. → FINDINGS_expA.md, FINDINGS_expD.md.

Two controls sharpen this. First, the agreement is not a burial confound one level up: partial correlation of
the two deficits controlling for interface burial is +0.529 [0.354, 0.678], and it survives dropping the
shared top-three complexes (+0.498). The predictors agree on which interfaces are hard *beyond* what burial
predicts. → deficit_burial_residualize.csv. Second, the effect tracks *how* a backbone is non-native, not how
far: on partial-diffusion backbones that are noised crystals at the same interface RMSD, the deficit is
absent. It is a property of *independent reconstruction* — the small, systematic errors a predictor makes at
an interface it must build without seeing the side chains — precisely the regime a de-novo design occupies. →
FINDINGS_expC2.md.

**And the binding signal designers would actually read off these backbones survives them.** The deficit above
is the *confidence*-type readout degrading; the complementary question is whether the *mixed derivative* — the
object this paper says carries binding — still works when computed on a predicted rather than a crystal
backbone. It does. Re-running the identical leverage scorer on the OpenFold3 and AF2-multimer backbones for the
140 complexes shared with the SKEMPI fixture, with geometry recomputed from the predicted structure (the honest
baseline a designer has, not the crystal), CPI(L | burial+nbr+ΔSASA) is **+0.039 [+0.027, +0.050]** on OpenFold3
and **+0.032 [+0.023, +0.041]** on AF2-multimer (pooled +0.036 [+0.026, +0.048]), P(>0)=1.000, each surviving
removal of its three most influential complexes, against a matched crystal-on-140 value of +0.046 — a 69–84%
retention, and L still adds beyond confidence on the predicted backbone (+0.046/+0.036, CI>0). The positive
control gates it: the same pipeline on crystal backbones reproduces the committed leverage to 1×10⁻⁵. Two honest
limits. First, at the position level the signal attenuates more (CPI(L→Ala | geom) +0.007/+0.002, still CI>0),
and the *ranking* gain from adding |L| to geometry, which is significant on crystals (ΔAUROC +0.014 [+0.004,
+0.025]), becomes marginal on predicted backbones (+0.007 [−0.001, +0.015] OpenFold3, +0.005 [−0.002, +0.012]
AF2): the mixed derivative is the right thing to *read binding from* at design time, but as a plug-in ranking
feature its crystal-grade lift does not fully transfer. Second, OpenFold3 retains more than AF2 (84% vs 69%),
the same ordering the deficit gives — the more interface-native predictor loses less signal, as the mechanism
predicts. Pre-registered before any number (PREREG_leverage_predicted.md); → FINDINGS_leverage_predicted.md,
leverage_predicted.csv, leverage_predicted_ranker.csv.

## 7. Ruling out competing mechanisms

Two alternative explanations for a hotspot deficit, both plausible a priori, are separately measured and
refuted. The first is a *sample-budget* effect: at the low sampling temperatures used in design, the joint
probability of recovering a specific multi-residue hotspot constellation is astronomically small (~10⁻¹⁰),
so perhaps hotspots are simply lost to sampling. But that cost is statistically identical at burial-matched
*control* constellations (median difference 0.000, p=0.90): it is a generic property of low-temperature
sampling of any buried residue set, not a hotspot-specific tax, and no amount of oversampling recovers it
because the barrier is the temperature exponent itself. The second is a *commitment-ordering* effect: the
autoregressive schedule might commit non-hotspot context first and paint hotspots into a corner. On
ProteinMPNN, the oracle decoding order is inert (difference-in-differences −0.002, the decisive test); on
MultiFlow, a coupled sequence-structure model, structure commits marginally before sequence and the
unmasking-order knob has only a marginal effect (order-span 0.012, far below the seed-to-seed SD of 0.065 —
the knob is inert relative to seed noise). The schedule mechanism is thus ruled
out decisively on the autoregressive model and shown marginal on the coupled one. → FINDINGS_expB.md. Neither
competitor accounts for the effect; what remains is the conditioning-set signal of §6.

## 8. Related work and positioning

Our sequence-free KL detector is, by construction, a **learned frustratometer** — but the concept is not ours
to claim, and we are careful to credit it. That partner ablation exposes *binding-site frustration* is a
classical statistical-mechanics result: Ferreiro et al. (2007) find highly frustrated interactions clustered
near binding sites, and the active-site case is Freiberger et al. (2019); the *frustratometer* itself is Parra et
al.'s tool, and neural predictors of the classical frustration index already exist (FrustraMPNN, FrustrAI-Seq).
Our narrow, diagnostic point is only that we read the inverse-folding *likelihood itself* — not a physics energy
function — as a per-residue partner-ablation signal, and find it equals ΔSASA, so the neural version does not
beat the physics (which is why the *scalar* KL adds only a small increment). That a partner-conditioned-versus-
masked KL is essentially a geometric quantity was in fact shown concurrently, for *ligand* conditioning, by
UMA-Inverse; what is ours is the *formal* feature-class law and the beyond-geometry control, not the observation
that this one scalar tracks geometry. **The leverage operator L is not ours: it
is BA-Cycle** (Jiao, Mao, Jin et al. 2024, arXiv:2410.09543), whose bound-versus-unbound double-difference
rearranges to exactly our mixed second difference, and which we credit outright for the score (they report a
comparable SKEMPI ΔΔG correlation). Our contribution is orthogonal to theirs: **(i)** the *decomposition* —
identifying their score as the mixed derivative and confidence as the diagonal, with the constructive proof
that confidence is blind to it; **(ii)** the *beyond-geometry control* — to our knowledge the
first for an inverse-folding binding signal (BA-Cycle runs none — no burial/rSASA/ΔSASA/contact anywhere in
their paper, which we verified), built on the conditional predictive impact (Watson & Wright 2021) with a
conditional permutation test (Berrett et al. 2018), so the fact that L survives geometry (and that scalar
summaries do not) is new; and **(iii)** the *feature-class law*. We also differ in construction
— per-position sequence-free marginals (design-time usable, decoding-order-free) versus their whole-sequence
autoregressive likelihoods. **StaB-ddG** parameterises ΔΔG through a folding-energy difference on an overlapping
fixture; a distinct question. **RedNet** independently operationalises exactly this leverage as a *design-time
decoder*: its contrastive decode `logit_bound + α·(logit_bound − logit_apo)` — verified in their released code
(zw2x/rednet_public: the α-tilt in `sampling_utils.py` and the partner-deleted apo contrast in
`infer_pipeline.py`) — is our mixed derivative applied at sampling time. Where RedNet retrains a decoder, we show the tilt is already
actionable on a *frozen, off-the-shelf* model and that an *independent* model scores the steered residues as
higher-binding (§4) — turning the shared direction into a diagnosis-then-intervention arc. (One terminological guard: RedNet's own
framing invokes the *thermodynamic* decomposition of binding free energy — the standard `ΔG_bind = ΔG_complex −
Σ ΔG_partners`; our "decomposition" is a distinct object, a split of the *model's likelihood function* into a
diagonal-confidence and a mixed-leverage derivative, which is what makes ours a diagnostic rather than a decoder
objective.) That an independent design pipeline reintroduces precisely this term is strong corroboration that the
leverage is the *actionable* quantity, and we
credit it as such; our contribution is again orthogonal — the decomposition, the first beyond-geometry control,
and the feature-class law, none of which RedNet reports (it runs no burial/ΔSASA control and no scalar-vs-mixed
split). On the phenomenon itself, **ProBID-Net** reports interface blindness as a recovery deficit; we correct
the attribution — it is neither dynamics nor decoding but conditioning, and a burial confound on the crystal
benchmark. The most telling piece of related practice is **BindCraft**, whose one-shot binder pipeline
hard-codes a 4 Å interface freeze that forbids inverse folding at the interface — the field's implicit
admission of our thesis, to which we give a measurement and a principled improvement. Ranking interface
positions for hotspot triage at a matched budget, IF **confidence is at chance** (position-level AUROC 0.51;
capture@3 0.064 vs 0.084 random, overlapping intervals) — which *justifies* freezing the interface rather than
trusting IF confidence there. But the model's binding knowledge *is* actionable if read from the right place:
on crystal backbones, **adding the mixed derivative to geometry sharpens hotspot ranking**. Leverage |L| on its
own reaches position-level AUROC 0.694 — on par with free ΔSASA (0.664; paired +0.030 [−0.001, +0.061], a trend,
CI touching zero) — but the decisive, actionable statement is the *combination*: adding |L| to the standard
geometric feature set (burial + neighbours + ΔSASA) lifts hotspot AUROC from 0.704 to **0.717 (paired +0.0125
[+0.0007, +0.0246], P(>0)=0.98)**, a CI that excludes zero. So the corrected practical rule is *rank interface
positions by geometry **plus** the mixed derivative, not the confidence* — a training-free readout that adds to
the standard feature set — with the dose-law caveat (§4) that this holds
on accurate backbones and degrades with reconstruction error. → leverage_triage.csv, w4_combined_ranker.csv, bindcraft_triage.csv. Finally, a wave of
conditioning-aware inverse-folding methods (AlphaFold-DB debiasing / DeSAE, target-conditioned inverse
folding, UMA-Inverse) *presupposes* the conditioning-set problem; we *measure* it and show the standard
benchmark hides it. Independent corroboration of the core claim comes from **Janusz et al. (2026)**, who
benchmark antigen-aware antibody inverse folding and report "a very weak effect of antigen on the predictions" —
structure validity acting as a statistical shortcut, with ProteinMPNN *losing* performance when the antigen
chain is included — the same partner-insensitivity we quantify, from a group that did not compute the ablation.
⟨✎ external citations DOI-verified via the reference checker: BA-Cycle, RedNet, StaB-ddG, Frellsen, DeSAE,
UMA-Inverse, Cagiada, Ferreiro/Freiberger, Watson–Wright, Berrett, Janusz, ProteinMPNN; full .bib at submission.⟩

## 9. Limitations

We evaluate on three fixtures — SKEMPI (natural, primary), Bennett de-novo designs, and AB-Bind
(antibody–antigen) — none a full generate→design→wet-lab loop; the de-novo evidence is four targets and
AB-Bind's 27 complexes are indeterminate for the leverage test. The primary claims rest on one natural fixture;
we mitigate this with the two further fixtures, four inverse-folding architectures, and a catalytic-site
replication, but a second large natural binding fixture would strengthen them. The de-novo effect in particular
is small — the two-pass-specific increment is +0.0032 [+0.0015, +0.0048] — and we do not lean on it: its role is
only to show the mixed derivative *reaches* the genuine design regime, and its best support is that predicted
backbones, where the effect is larger, fall in the *surviving* part of the dose law (§4, §6).

**Caveats specific to the decomposition.** (a) *Orthogonal is not independent*: confidence cannot *express*
leverage, but the two are weakly-to-moderately correlated and the correlation is model-dependent
(Spearman(confidence, |L|) = +0.075 for ProteinMPNN, +0.31 for ESM-IF1); we claim blindness by construction
(a flexible learner over the full bound distribution recovers only ~37% of the leverage — ~63% is irreducible
from `P` in both families, → r2_leverage_from_P.csv), not statistical
independence. (b) The leverage operator L *is* BA-Cycle (Jiao et al. 2024); we
credit the score and claim the decomposition, the beyond-geometry control, and the feature-class law. (c) L
estimates −ΔΔG_bind only up to an unknown temperature — no calibrated kcal/mol reading; all our readouts are
scale-invariant. (d) The per-position log-Z argument (that L is better-posed than confidence) is ours; we do
*not* lean on the free-energy interpretation of Frellsen et al. (2025), whose normaliser is global-per-sequence
and whose quantity is ΔΔG_fold, not binding. (e) Rigid backbone: the monomer conditioning is the complex
backbone minus partner. (f) The headline uses one inverse-folding model (ProteinMPNN, sequence-free
marginals); the decomposition *replicates* under a second, architecturally-distinct model — ESM-IF1
(GVP-transformer, native-teacher-forced conditional readout; 337/344 SKEMPI complexes, 7 oversized dropped for
memory — re-running ProteinMPNN on that same 337-subset leaves it unchanged, CPI(L|geom) +0.060 and Spearman
−0.303, so the model gap below is not a sample-selection artifact): confidence stays blind to hotspots (position-level CPI −0.0000, CI spans 0), while leverage adds beyond
geometry (position +0.0042, survives drop-3; mutation Spearman(L,ΔΔG) = −0.26, CPI +0.035 surviving
geometry+substitution+confidence+scalar-KL, and +0.010 fully controlled), with somewhat smaller magnitudes than
ProteinMPNN. So the feature-class law is a property of the inverse-folding class, not one model. →
leverage_esmif.csv. (g) CPI is not formally commensurable across fixtures, so "natural ≫
de-novo" is a suggestive, not a formal, comparison. (h) *Effect sizes are modest in absolute terms* — the
position-level CPI is +0.0048 — because the hotspot label is rare (base rate 2.4%, entropy 0.115 nats). We read
them relatively rather than papering over them: leverage is 4.2% of the label's entropy (an order of magnitude
above any scalar of `P`), ~71% of what the explicit geometric ΔSASA contributes, and, most concretely, it
reaches a standalone training-free hotspot-ranking AUROC of 0.694 (§8). The result is a *dissociation between
feature classes*, not a large-magnitude predictor — and it is that dissociation, replicated across architectures
and backbone regimes, that the paper claims.

**Other limitations.** The all-atom occlusion baseline is a min-over-rotamer repacking proxy, not a force field
(the 95%-zero-clash prevalence bounds what any clash model could recover). De-novo binding labels convolve
display and fold-stability with binding — the core/interface stratification is the control, native excluded.
SKEMPI training leakage makes the predicted-backbone result *conservative*. The strict-control tier is
underpowered by design; the verdict rests on higher-powered tiers declared in advance. One extension did not
survive its control — a within-natural confidence-decay gradient (null) — which we report rather than bury; the
generalisation to catalytic residues, by contrast, *does* survive its composition, burial, and chain-truncation
controls (§4).

## Appendix A. Pre-registered false-positive modes and their controls

Six ways to get a false positive were named in the pre-registration *before* any number was computed. We
list each with the control that addresses it and where the result appears, so a reviewer can check the
armor against the threat it was built for rather than reconstruct the mapping. One (assay heterogeneity) is
controlled only in part; we say so rather than overstate it.

| # | Pre-registered false-positive mode | How it would fake — or hide — the effect | Control | Result / where |
|---|---|---|---|---|
| 1 | **Burial** | Buried positions are where inverse folding is *most* confident, so an uncontrolled hotspot-vs-rest comparison **hides** the effect (the confound cuts against us, not for us) | within-complex matched pairs (rSASA ±0.05, secondary-structure class, neighbour count ±1); and CPI over burial + neighbours + ΔSASA at *every* downstream step | deficit largely attenuates unmatched→matched across 5 architectures + ProBID-Net (residual for 2/5 on the strict tier; §5); leverage CPI **+0.0048** survives full geometry (§4) |
| 2 | **Native amino-acid identity** (Trp/Arg/Tyr are hotspot-enriched with distinctive priors) | the model's per-type prior, not binding, drives the score | per-wt-type breakdown; alanine-only subset; substitution-similarity (BLOSUM, side-chain volume, hydropathy) partialled out | Spearman(L, ΔΔG) negative in **18/19** wt-types and **−0.25 on Ala-only** (n=2,327); survives similarity controls (§4) |
| 3 | **PDB training leakage** | the model has seen these complexes | *none needed* — leakage makes a positive **conservative** (the model is scored on structures it memorised, which can only *help* recovery/confidence, i.e. work against our deficit) | stated as such; every positive here is a lower bound (§1, §9) |
| 4 | **Assay heterogeneity** (SKEMPI pools ITC, SPR, fluorescence) | a hotspot threshold or condition artifact masquerades as signal | strict (>2 kcal/mol, ProBID-Net's threshold) **and** loose (>1) hotspot definitions, both reported | conclusions hold under both thresholds (§2). **Partial:** we do *not* stratify by temperature/pH for the headline — disclosed as a limitation, not claimed as a control (§9) |
| 5 | **Positional independence** (additivity is false) | an additive model of hotspot effects is wrong, so any independence assumption inflates confidence | not assumed — we **measure** the second mixed derivative directly (partner-ablated pairwise coupling) | model couplings track experimental binding epistasis, Spearman(C, g) **−0.14**, cycle-predicted sign (§4, couplings) |
| 6 | **Decoding-order variance** (ProteinMPNN's autoregressive order changes conditional probabilities) | a result could live entirely inside order noise | sequence-free marginals where possible; elsewhere symmetrise over ≥8 orders and read disjoint order subsets | disjoint-order subsets agree Spearman **+0.61**; the oracle decoding order is inert (difference-in-differences −0.002) (§4, §7) |

The pattern worth noting is #1 and #3: the two largest confounds both run *against* the hypothesis, so the
burden of proof is on us to show the effect *despite* them, not because of them — which is why the
matched-pair design, not the raw deficit, is the experiment.

## Reproducibility and LLM-usage disclosure

Every numerical result in this paper traces to a committed script and CSV under `results/`, each carrying the
exact command and a fixed seed; the central result (the Confidence–Leverage Decomposition) was additionally
subjected to an independent adversarial audit of its statistical machinery (the conditional-independence test
was verified not to leak: pure-noise features give CPI ≈ 0, the null floor is ≈16× below the reported effect).
Per the ICLR LLM-usage policy, we disclose that large language models (Anthropic's Claude) were used as an
assistant for code generation, data analysis, prose drafting and editing, and research ideation. All
experiments, numerical results, and citations were verified and validated by the authors, who are solely
responsible for the contents of this submission; no result or citation is included that does not resolve to
committed code and data or to a source we fetched and checked.

---
*Draft status: §1–9 in prose. Pending: fold in the Fable-5 catalytic audit (§4 note); a figure pass;
external-citation URL verification; final length trim to 9 pages.*
