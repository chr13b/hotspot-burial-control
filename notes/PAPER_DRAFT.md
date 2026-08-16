# Confidence is not competence: interface-hotspot "blindness" in inverse folding is a conditioning-set artifact — and what the model knows about binding lives in its distribution, not its confidence

*Working prose draft, 2026-08-14. Expands notes/PAPER_OUTLINE.md (Spine B). Every quantitative claim carries
a `→ file.csv` trace to a committed result. Sections marked ⟨PENDING …⟩ await a running analysis.*

---

## Abstract

Staged binder design — generate a backbone, then inverse-fold a sequence — is believed to stumble at
protein–protein interface *hotspots*, and a prominent report (ProBID-Net) quantifies this as inverse-folding
recovery of 0.334 at hotspots versus 0.472 elsewhere, attributed to dynamics. We show the phenomenon is real
but misread, and give the reason as a theorem. **Confidence and competence are two orthogonal derivatives of
the same inverse-folding likelihood.** A model's per-residue *confidence* is the *diagonal* term — a scalar
summary of one conditioning — and estimates fold-stability constraint; a residue's binding *leverage* is the
*mixed second derivative*, the response to ablating the binding partner, `[log p(a|complex)−log p(wt|complex)]
− [log p(a|monomer)−log p(wt|monomer)]`, which estimates the binding effect. Two positions can share an
identical bound distribution — hence identical confidence — yet differ in leverage, so **confidence is blind
to binding by construction, not by failure** (we verify: confidence-matched positions retain 30% of the
leverage spread). Every readout the field has used — recovery, confidence, the complex-vs-monomer KL — is a
*scalar* summary, and on natural complexes such summaries reduce to cheap geometry: across five architectures
confidence ranks interface hotspots at chance and adds *zero* beyond geometry (conditional predictive impact
0.000), and the KL detector is a *learned frustratometer* recapitulating partner-contact area. **The mixed
derivative does not reduce to geometry.** On our main fixture (SKEMPI, natural complexes) it adds substantial
binding information beyond geometry — mutation-level CPI +0.059, Spearman with experimental ΔΔG −0.30 —
robustly, exactly where every scalar summary (confidence included) adds nothing. This is a *feature-class law*: scalar
summaries are geometry; the mixed derivative is competence. It unifies the field's observations as corollaries
— the published deficit is a burial confound (pre-registered matched design, five architectures plus
ProBID-Net's own model); the blindness generalises from binding to *catalytic* residues (structure-conditioned
confidence blind while sequence conservation predicts, triple-controlled); and the recipe generalises beyond
inverse folding: to read un-trained function from any conditional generative model, ablate the conditioner and
read the mixed derivative, not the confidence. The leverage operator itself is BA-Cycle (Jiao et al. 2024); our
contribution is the decomposition, the blindness theorem, the first beyond-geometry control, and the
feature-class law. Practically: rank interface positions by partner geometry, not confidence (below random),
and read binding from the mixed derivative, not the scalar summaries.

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
verify this is non-vacuous (confidence-matched interface positions retain ≈30% of the leverage spread). This
yields a **feature-class law**: on natural complexes every *scalar* summary of the distribution reduces to cheap
geometry (confidence adds CPI 0.000; the KL detector recapitulates ΔSASA), whereas the *mixed derivative* does
not — on SKEMPI it adds binding information beyond geometry (mutation-level CPI +0.059, Spearman with
experimental ΔΔG −0.30; ~5× the best scalar summary, where confidence adds nothing), robustly. The model knows binding on natural complexes; the knowledge
was invisible to every scalar readout the field has used. → leverage_decomposition.csv, nugget_cpi.csv.

**(ii) Confidence is not competence — a property of inverse folding, with a practical consequence.** The
diagonal is blind across five architectures (interface-hotspot AUROC 0.50–0.54; conditionally independent of
hotspot-ness given geometry, CPI 0.000). This measures the field's implicit BindCraft interface-freeze: ranking
interface positions by confidence captures *fewer* hotspots than random (capture@3 0.064 vs 0.089), while free
ΔSASA captures ~3× more (0.233) — so rank by geometry, not confidence. De-novo designs corroborate the positive
of (i) with even the scalar distribution: it beats substitution baselines (0.615), dissociates stability from
binding, and adds +0.018 beyond an all-atom rotamer-repacked occlusion baseline, reproduced by a non-parent
scorer. → xmodel_confidence.csv, baseline_audit.csv, bindcraft_triage.csv, bennett_occlusion_allatom.csv, bennett_nonparent.csv.

**(iii) The blindness generalises across function types.** On catalytic residues (M-CSA), structure-conditioned
confidence is blind (within-amino-acid-type AUROC ≈ 0.50) while a sequence language model's conservation
predicts them (0.77) — a dissociation surviving composition, burial, and chain-truncation controls.
Inverse-folding confidence is blind to functional importance in general, not only binding. → catalytic_audit.csv.

**(iv) The published deficit is a burial confound.** Under a pre-registered burial-matched matched-pair
design (matching within-complex on relative SASA, secondary-structure class, and neighbour count), the
crystal-backbone hotspot deficit vanishes across five inverse-folding architectures, and ProBID-Net's own
released voxel-CNN reproduces its published deficit and then dissolves it under joint burial-and-composition
matching. We offer the matched-pair design as a reusable diagnostic *protocol*. → probid_gap_estimators.csv,
composition_confound.csv.

**(v) The residual tax lives in the conditioning set.** On the predicted backbones designers actually use —
from two architecturally-independent structure predictors — a burial-matched deficit reappears, and the two
predictors' per-complex deficits agree (ρ = 0.57): the *same* complexes are hard under both. This agreement
survives residualising on interface burial (partial ρ = 0.53), so it is not a recursive burial effect, and it
is absent on noised-crystal backbones at matched distance — it tracks *independent reconstruction*, not
distance-from-native. → deficit_burial_residualize.csv. Two competing mechanisms — a low-temperature
constellation cost and a commitment-ordering schedule — are separately measured and refuted.

## 2. Setup and pre-registration

**Fixtures.** Our primary fixture is SKEMPI 2.0, from which we take single-mutation binding data and define a
hotspot as an alanine-scan ΔΔG_bind > 1 kcal/mol (and a strict variant > 2, ProBID-Net's threshold), with a
null set of |ΔΔG| < 0.25. Because SKEMPI complexes are crystal structures of *natural* complexes, we add two
independent fixtures of different character: **Bennett-2023 de-novo designed binders**, which carry
experimental site-saturation binding measurements over four targets and constitute a true design-regime test;
and **AB-Bind**, antibody–antigen ΔΔG over 32 complexes, a second SKEMPI-class fixture with distinct
biophysics.

**The matched-pair protocol.** The core of the burial analysis is a within-complex optimal 1:1 matching of
hotspots to null residues on relative complex SASA (±0.05), secondary-structure class, and neighbour count
(±1). Effects are aggregated by complex-level bootstrap; every seed is fixed (20260803) and reported with its
bootstrap replicate count. → PREREG.md, PREREG_knows_where.md, PREREG_bennett_hardening.md.

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
computable without any neural network — reaches 0.585; and a cheap-geometry combination reaches 0.734. The
obvious learned alternative is the sequence-free divergence between the model's complex- and
monomer-conditioned distributions (a KL detector), which one might hope captures partner-induced frustration
beyond geometry. It captures a *small* one: under a combiner-free conditional test, KL adds CPI = +0.002
[+0.0006, +0.0034], P=0.998 beyond full geometry, and its within-geometry-stratum AUROC is 0.60 (vs 0.50
leakage) — a genuine learned-frustratometer signal, but ~6× smaller than ΔSASA's contribution and not worth
the network as a standalone ranker. (We are careful here about readout: the unfitted z-sum ΔAUROC we first
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
which by the thermodynamic cycle estimates −ΔΔG_bind (up to an unknown temperature). The scalar KL is one
contraction of this vector: `KL(P‖Q) = E_{a∼P}[L(a)] + const` (verified to 1e-6). A methodological aside that
also motivates L: the per-position softmax normaliser `log Z_i` contaminates confidence but *cancels* in L
(each bracket is within-conditioning), so L is better-posed. **Confidence is blind to leverage by
construction:** two positions with an identical bound distribution have identical confidence yet can differ
arbitrarily in L — and this is not hypothetical, since matching interface positions on the *full* bound
distribution still leaves ≈30% of the leverage spread free.

**The feature-class law (on the main fixture, natural complexes).** On SKEMPI, the mixed derivative adds
binding information beyond cheap geometry where every scalar summary does not. Per interface position
(conditional predictive impact over burial+neighbours+ΔSASA, 13,401 interface positions; confidence here is the
same diagonal §3 finds conditionally independent):

| feature (all functionals of the same distribution) | CPI beyond geometry |
|---|---|
| **confidence** — the diagonal | **+0.0002 [−0.0003, +0.0006]** — conditionally independent (CI spans 0) |
| scalar KL — a contraction of L | +0.0009 [+0.0003, +0.0016] |
| **leverage L** — the mixed derivative | **+0.0048 [+0.0033, +0.0065]** — ~5× the best scalar (KL), where confidence adds nothing; survives dropping the 3 most influential complexes |

On the identical 5,742-position sample §3's confidence test uses, the same ordering holds and sharpens —
leverage +0.0092 [+0.0062, +0.0124], ~5× the scalar KL, while confidence stays conditionally independent (CI
spans 0). → leverage_nugget_match.csv. At the mutation level the effect is large: Spearman(L, experimental ΔΔG_bind) = **−0.30**, and CPI(L | geometry)
= **+0.059 [+0.046, +0.073]**, surviving controls from substitution similarity (BLOSUM, volume, hydropathy) and
from L's own scalar components. → leverage_decomposition.csv, FINDINGS_leverage.md. To calibrate the effect
size: L is a *zero-shot* readout of a model never trained on binding, yet it adds **+0.030 interface AUROC**
(0.700→0.730) beyond a *supervised* geometry+substitution baseline fit directly on the binding labels, and on
its own reaches AUROC 0.647 — near that supervised baseline. → leverage_effect_size.csv. **This is not a
ProteinMPNN artifact:** it replicates under ESM-IF1 — a GVP-transformer with a native-conditioned (not sequence-free) readout
— where confidence is again blind to hotspots and leverage again adds beyond geometry and beyond every scalar
including confidence (337/344 complexes; mutation Spearman −0.26, CPI +0.035; position confidence CPI −0.0000),
so the feature-class law is a property of the inverse-folding *class*. → leverage_esmif.csv. So the model *does*
know binding on natural complexes — the knowledge was invisible to every scalar readout the field used. The law is
not about a *regime* but a *feature class*: on natural complexes scalar summaries reduce to geometry
(confidence exactly, KL nearly); the mixed derivative does not.

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
summaries — confidence is at chance and adds zero beyond geometry (§3), and while a scalar confidence readout
rises to 0.60 on de-novo interfaces it stays at chance on natural SKEMPI (0.538) → bennett_conf_fork.csv — yet
the *mixed derivative* carries it. De-novo designs are simply where the signal is accessible to blunter probes
as well: there, even the scalar complex-conditioned distribution adds +0.018 beyond an all-atom occlusion
baseline. A methodological correction this forces: an earlier AB-Bind analysis reported the per-mutation
distribution "adds nothing" on natural antibody–antigen ΔΔG (ΔAUROC +0.008), but under the correct conditional
test on the same fixture it adds **CPI +0.042 [+0.022, +0.061]** — the ΔAUROC readout is the one with a −0.021
noise floor (§3). AB-Bind's 27 complexes remain too few to be decisive either way; SKEMPI is where the question
is settled. → abbind_bigidea1.csv, leverage_decomposition.csv.

**The blindness generalises beyond binding — to catalytic residues.** "Confidence is not competence" is not
specific to binding hotspots. On M-CSA catalytic residues, controlling for amino-acid composition by
stratifying *within* amino-acid type, structure-conditioned confidence is blind (within-type AUROC 0.48–0.50,
chance) while a sequence language model's conservation predicts them (0.771 [0.723, 0.822]) — a dissociation
of +0.288 [+0.235, +0.336] that survives on monomers alone (ruling out a partner-chain-truncation artifact:
there MPNN is 0.516, chance) and after additionally controlling for burial (+0.174 [+0.062, +0.288]).
Inverse-folding confidence is thus blind to functional importance across function types; what predicts
function is free geometry (for binding) or sequence conservation (for catalysis). We are deliberate about
mechanism: the model's confidence is *blind* (at chance), not actively *frustrated* — the raw anti-prediction
we first observed was an amino-acid-composition and single-chain-truncation artifact, not a determinacy
signal. → FINDINGS_catalytic.md, catalytic_audit.py. (Methodological note for the appendix: the effect is
invisible to a ΔAUROC-over-amino-acid-identity control, whose detection floor is a within-type AUROC of
~0.55; the correct readout is the within-type AUROC itself.)

**One attempt did not generalise (reported for the record).** A finer *within*-SKEMPI confidence-decay
gradient, binned by binding affinity, is null on 141 complexes; the natural regime does not furnish an
obligate endpoint (it is defined by measurable dissociation), so a transient→obligate gradient is not
constructible here. → confidence_gradient{,_affinity}.csv.

## 5. On crystal backbones, the hotspot gap is a burial artifact

We now return to the published deficit and show, on crystal backbones, that it is a burial confound. The
confound is visible directly: as hotspot strength increases, both sequence recovery and burial rise in
lockstep (recovery 0.347→0.529, relative SASA 0.218→0.080). Under the pre-registered matched-pair design —
pairing each hotspot to a null residue in the same complex at matched relative SASA, secondary-structure
class, and neighbour count — the deficit vanishes: the matched estimate is −0.042 [−0.222, +0.129] and a
higher-powered regression estimator is +0.059 [−0.051, +0.167], with every architecture's primary interval
containing zero across all five models. → FINDINGS.md.

The strongest form of this test uses ProBID-Net's own released voxel-CNN. Run on our fixture, its port is
faithful (overall interface recovery 0.472, matching its reported non-hotspot number), and its published
hotspot deficit *does* reproduce — concentrated, as one would expect, in comprehensively alanine-scanned
complexes (five or more measured hotspots: −0.113 [−0.208, −0.022], p=0.007). But it dissolves under
confound-matching: matching residue type turns it positive (+0.120), matching burial gives −0.038, matching
hydrophobicity −0.051, every interval spanning zero. ProBID-Net's deficit is thus a residue-composition and
burial confound — its voxel-CNN has an unusually extreme amino-acid-type dependence (per-type recall spanning
0.17 to 0.98), and hotspots are enriched in the types it recovers worst — not evidence of binding-specific
blindness. → probid_gap_estimators.csv, composition_confound.csv. (We correct an earlier draft of our own
that mislabeled this as an opposite-sign, fixture-specific null; that reading was a complex-averaging
artifact and is withdrawn.) We offer the matched-pair design itself as a reusable diagnostic protocol.

## 6. The tax lives in the conditioning set

If the deficit were purely a benchmark artifact, it should disappear everywhere once burial is controlled. It
does not — it reappears on the *predicted* backbones that designers actually condition on, and there it
behaves like a real, structured signal. On backbones from two architecturally-independent structure
predictors, OpenFold3 and AlphaFold2-multimer, a burial-matched deficit is present (−0.191 [−0.37, −0.004]
and −0.233 [−0.44, −0.035]; the crystal deficit is ≈0). The claim does not rest on either marginal number —
neither survives dropping its three most influential complexes — but on their *agreement*: the two predictors'
per-complex deficits correlate at ρ = 0.565 [0.40, 0.71], so the same complexes are hard under both. A
per-predictor memorisation or architecture artifact would produce disjoint deficits; two independent
reconstructions instead agree, per complex. → FINDINGS_expA.md, FINDINGS_expD.md.

Two controls sharpen this. First, the agreement is not a burial confound one level up: partial correlation of
the two deficits controlling for interface burial is +0.529 [0.354, 0.678], and it survives dropping the
shared top-three complexes (+0.533). The predictors agree on which interfaces are hard *beyond* what burial
predicts. → deficit_burial_residualize.csv. Second, the effect tracks *how* a backbone is non-native, not how
far: on partial-diffusion backbones that are noised crystals at the same interface RMSD, the deficit is
absent. It is a property of *independent reconstruction* — the small, systematic errors a predictor makes at
an interface it must build without seeing the side chains — precisely the regime a de-novo design occupies. →
FINDINGS_expC2.md.

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
unmasking-order knob has only a marginal effect (order-span 0.012, comparable to the seed-to-seed SD 0.011 —
not the clean null an earlier miscomputed variance suggested; corrected). The schedule mechanism is thus ruled
out decisively on the autoregressive model and shown marginal on the coupled one. → FINDINGS_expB.md. Neither
competitor accounts for the effect; what remains is the conditioning-set signal of §6.

## 8. Related work and positioning

Our sequence-free detector is, by construction, a **learned frustratometer**: the partner-induced change in
local frustration at interfaces is a classical statistical-mechanics quantity (Ferreiro, Parra and
colleagues), and our finding that the KL detector equals ΔSASA says the neural version does not beat the
physics — which is why the *scalar* KL adds only a small increment. **The leverage operator L is not ours: it
is BA-Cycle** (Jiao, Mao, Jin et al. 2024, arXiv:2410.09543), whose bound-versus-unbound double-difference
rearranges to exactly our mixed second difference, and which we credit outright for the score (they report a
comparable SKEMPI ΔΔG correlation). Our contribution is orthogonal to theirs: **(i)** the *decomposition* —
identifying their score as the mixed derivative and confidence as the diagonal, with the constructive proof
that confidence is blind to it; **(ii)** the *first beyond-geometry control* — BA-Cycle runs none (no
burial/rSASA/ΔSASA/contact anywhere in their paper, which we verified), so the fact that L survives geometry
(and that scalar summaries do not) is new; and **(iii)** the *feature-class law*. We also differ in construction
— per-position sequence-free marginals (design-time usable, decoding-order-free) versus their whole-sequence
autoregressive likelihoods. **StaB-ddG** parameterises ΔΔG through a folding-energy difference on an overlapping
fixture; a distinct question. **RedNet** independently operationalises exactly this leverage as a *design-time
decoder*: its contrastive decode `logit_bound + α·(logit_bound − logit_apo)` — verified in their released code
(zw2x/rednet_public: the α-tilt in `sampling_utils.py` and the partner-deleted apo contrast in
`infer_pipeline.py`) — is our mixed derivative applied at sampling time. That an independent design pipeline
reintroduces precisely this term is strong corroboration that the leverage is the *actionable* quantity, and we
credit it as such; our contribution is again orthogonal — the decomposition, the first beyond-geometry control,
and the feature-class law, none of which RedNet reports (it runs no burial/ΔSASA control and no scalar-vs-mixed
split). On the phenomenon itself, **ProBID-Net** reports interface blindness as a recovery deficit; we correct
the attribution — it is neither dynamics nor decoding but conditioning, and a burial confound on the crystal
benchmark. The most telling piece of related practice is **BindCraft**, whose one-shot binder pipeline
hard-codes a 4 Å interface freeze that forbids inverse folding at the interface — the field's implicit
admission of our thesis, to which we give a measurement and a principled improvement. Ranking interface
positions for hotspot triage at a matched budget, IF **confidence captures fewer hotspots than random**
(capture@3 0.064 vs 0.089; @5 0.125 vs 0.138) — which *justifies* freezing the interface rather than trusting
IF there — while free **ΔSASA captures ~3× more** (0.233 @3), well above the uniform freeze. So the field's
hack is right about confidence and improvable with free geometry: freeze-then-prioritise-by-ΔSASA beats both
trusting confidence and the uniform freeze. → bindcraft_triage.csv, FINDINGS_bindcraft.md. Finally, a wave of
conditioning-aware inverse-folding methods (AlphaFold-DB debiasing / DeSAE, target-conditioned inverse
folding, UMA-Inverse) *presupposes* the conditioning-set problem; we *measure* it and show the standard
benchmark hides it. ⟨✎ verify all external citation URLs before submission — BAIF/DeSAE/CPI/free-energy-interp
fetched; the remainder search-only.⟩

## 9. Limitations

We evaluate on three fixtures — SKEMPI (natural, primary), Bennett de-novo designs, and AB-Bind
(antibody–antigen) — none a full generate→design→wet-lab loop; the de-novo evidence is four targets and
AB-Bind's 27 complexes are indeterminate for the leverage test.

**Caveats specific to the decomposition.** (a) *Orthogonal is not independent*: confidence cannot *express*
leverage, but the two are weakly-to-moderately correlated and the correlation is model-dependent
(Spearman(confidence, |L|) = +0.075 for ProteinMPNN, +0.31 for ESM-IF1); we claim blindness by construction
(a confidence-matched pair still spans most of the leverage range — 73% of SD for ESM-IF1), not statistical
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
de-novo" is a suggestive, not a formal, comparison.

**Other limitations.** The all-atom occlusion baseline is a min-over-rotamer repacking proxy, not a force field
(the 95%-zero-clash prevalence bounds what any clash model could recover). De-novo binding labels convolve
display and fold-stability with binding — the core/interface stratification is the control, native excluded.
SKEMPI training leakage makes the predicted-backbone result *conservative*. The strict-control tier is
underpowered by design; the verdict rests on higher-powered tiers declared in advance. One extension did not
survive its control — a within-natural confidence-decay gradient (null) — which we report rather than bury; the
generalisation to catalytic residues, by contrast, *does* survive its composition, burial, and chain-truncation
controls (§4).

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
