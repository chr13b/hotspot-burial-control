# Confidence is not competence: interface-hotspot "blindness" in inverse folding is a conditioning-set artifact — and what the model knows about binding lives in its distribution, not its confidence

*Working prose draft, 2026-08-14. Expands notes/PAPER_OUTLINE.md (Spine B). Every quantitative claim carries
a `→ file.csv` trace to a committed result. Sections marked ⟨PENDING …⟩ await a running analysis.*

---

## Abstract

Staged binder design — generate a backbone, then inverse-fold a sequence onto it — is widely believed to
stumble at protein–protein interface *hotspots*, the few residues that dominate binding free energy. A
prominent report (ProBID-Net) quantifies this as an inverse-folding sequence-recovery of 0.334 at hotspots
against 0.472 elsewhere, and attributes it to protein dynamics. We show the phenomenon is real but almost
entirely misread, and in the process characterise what inverse-folding models do and do not know about
binding. **First, confidence is not competence.** Across five inverse-folding architectures, a model's own
per-residue confidence ranks interface hotspots no better than chance (AUROC 0.50–0.54), and a combiner-free
conditional-independence test shows it carries *zero* information about hotspot-ness beyond cheap structural
geometry (conditional predictive impact 0.000); a free geometric feature — the buried surface area a residue
loses on binding — predicts hotspots instead. The obvious learned detector, the divergence between a model's
complex- and monomer-conditioned distributions, largely recapitulates that geometry (adding only a small real
increment beyond it): a learned frustratometer, not a new method. This pattern replicates on a second,
biophysically distinct fixture
(antibody–antigen ΔΔG). **Second, and in tension with the first, the model does know binding — in its
distribution, not its confidence.** On genuinely de-novo designed binders with experimental site-saturation
binding measurements, a model's per-substitution complex-conditioned distribution ranks substitutions by
measured binding, does so better for the fold-stability question than the binding question (a built-in
control), gains specifically from conditioning on the partner, and — the decisive test — still adds signal
beyond an all-atom, rotamer-repacked steric-occlusion baseline: after repacking, 95% of substitutions incur
no clash, so occlusion cannot explain the effect. A non-parent model reproduces every part of this, ruling
out circularity. **Third, the published deficit is a burial confound**, and the residual signal lives in the
*conditioning set* the field benchmarks on: under a pre-registered burial-matched design the crystal-backbone
deficit vanishes (five architectures plus ProBID-Net's own released model), yet it reappears on the predicted
backbones designers actually use, where two independent structure predictors agree, per complex, on which
interfaces are hard — agreement that survives controlling for burial. The upshot is practical: read hotspots
from partner geometry, not model confidence, and read the model's binding knowledge from its full
distribution, not its scalar summaries.

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

We make four contributions.

**(i) Confidence is not competence.** At interface hotspots, a model's own per-residue confidence is at
chance (AUROC 0.538 for ProteinMPNN; 0.50–0.54 across five architectures) and ranks hotspots *below* random
for fixed-budget triage. A conditional-predictive-impact test — cross-fitted and immune to any choice of
combiner — shows confidence is conditionally independent of hotspot-ness given cheap geometry (CPI 0.000
[−0.0003, +0.0003]). What predicts hotspots is a free geometric feature, the change in solvent-accessible
surface area on binding (ΔSASA, i.e. partner-contact area). The natural learned detector — the KL divergence
between a model's complex-conditioned and monomer-conditioned sequence distributions — adds only a small real
increment over that geometry (CPI +0.002, P=0.998; ~6× below ΔSASA): a *learned frustratometer* that largely
recapitulates the classical geometry, which we treat as a diagnostic rather than a method. The entire pattern
replicates on a second, independent fixture (antibody–antigen ΔΔG).
→ nugget_cpi.csv, xmodel_confidence.csv, baseline_audit.csv, kl_geometry_control{,_predicted}.csv, abbind_nugget.csv.

**(ii) The model knows binding — in its distribution, not its confidence.** On de-novo designed binders with
experimental site-saturation binding data, the model's per-substitution complex-conditioned distribution
ranks the 19 non-native substitutions by measured binding (AUROC 0.615, beating substitution-similarity
baselines), answers the fold-*stability* question better than the *binding* question (a dissociation that
serves as a positive control), and gains specifically from conditioning on the partner (+0.076,
interface-specific). Crucially, this signal survives an *all-atom, rotamer-repacked* occlusion baseline
(+0.018 [+0.015, +0.022]); since 95% of substitutions incur no steric clash after repacking, occlusion cannot
account for it — the model encodes per-substitution binding *energetics* beyond geometry. A non-parent scorer
reproduces the interface ranking, the partner-gain, and the beyond-occlusion signal, ruling out that this is
a model scoring around its own mode. → bennett_knows_where.csv, bennett_occlusion_allatom.csv, bennett_nonparent.csv.

**(iii) The published deficit is a burial confound.** Under a pre-registered burial-matched matched-pair
design (matching within-complex on relative SASA, secondary-structure class, and neighbour count), the
crystal-backbone hotspot deficit vanishes across five inverse-folding architectures, and ProBID-Net's own
released voxel-CNN reproduces its published deficit and then dissolves it under joint burial-and-composition
matching. We offer the matched-pair design as a reusable diagnostic *protocol*. → probid_gap_estimators.csv,
composition_confound.csv.

**(iv) The residual tax lives in the conditioning set.** On the predicted backbones designers actually use —
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

## 4. The model knows binding — in its distribution, not its confidence

The results so far are corrective: the model's confidence is uninformative about hotspots, and the natural
learned detector reduces to geometry. Taken alone they would make a purely negative paper. But they concern
only *scalar summaries* of the model — its confidence, and a one-number divergence. The model's full
per-substitution distribution is a richer object, and it turns out to carry genuine binding information that
those scalars discard. Establishing this requires a fixture with *per-substitution* binding measurements, so
we turn to de-novo designed binders (Bennett-2023), which come with experimental site-saturation mutagenesis
over four targets: for each interface position, the binding phenotype of all 19 non-native substitutions. A
sanity control passes exactly — the single amino acid absent from each SSM library equals the design's native
residue in 4137/4137 positions. → bennett_knows_where.csv.

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

**Why de-novo, and only de-novo.** The theory that organises these results is a distinction between
*constraint* and *leverage*. Inverse-folding confidence estimates how *constrained* a position is by the fold
it conditions on; hotspot-ness is *leverage*, how much binding free energy depends on the residue. These
coincide only when selection on a position is binding-dominated. De-novo binders are the extreme of that
regime — they exist only to bind — so the prediction is that the model's binding signal should be *most*
accessible there. Two observations bear it out. First, scalar confidence, which is at chance for hotspots on
natural SKEMPI complexes (0.538), rises to 0.60 on de-novo interfaces. → bennett_conf_fork.csv. Whether the
*beyond-geometry* positive itself is de-novo-specific is the open question: on natural antibody–antigen
mutations (AB-Bind) the model's distribution correlates with ΔΔG in the right direction (Spearman −0.17) and
beats chance standalone (0.578), but that fixture is **underpowered** — it cannot certify even BLOSUM62 as
adding beyond geometry in the conditional test — so it can neither confirm nor refute a natural-complex
positive (n=420 mutations, 27 complexes). The pivotal test is on SKEMPI with the full per-mutation leverage
operator (§4a, the decomposition); we report it there. → abbind_bigidea1.csv. The picture the decomposition
predicts is *graded*, not binary: on *natural* complexes the model's binding signal is **mostly** geometry
(the scalar KL adds only a small increment beyond ΔSASA), whereas on *de-novo* designs the distribution
carries binding energetics that geometry does not (+0.018 beyond all-atom occlusion). The model's binding
knowledge is real, latent in its distribution, and increasingly accessible as selection becomes
binding-dominated.

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
physics — which is why we demote it. The closest machine-learning prior art is **BAIF** (Boltzmann-aligned
inverse folding), which scores mutations by inverse-folding log-likelihood over bound-versus-unbound states —
the same two conditioning sets our KL uses — for ΔΔG prediction. We differ in question and in claim: we run a
per-*substitution* test of the *full* conditional distribution against experimental binding, stratified by a
stability positive control, and beyond an all-atom occlusion baseline (§4); and we keep the scalar detector
demoted rather than proposing it as a method. **StaB-ddG** parameterises ΔΔG through a folding-energy
difference on an overlapping fixture; a distinct question. On the phenomenon itself, **ProBID-Net** and
**RedNet** report interface blindness (as recovery deficit and as a decoding problem respectively); we correct
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

We evaluate on three fixtures — SKEMPI (natural complexes, primary), Bennett de-novo designs (the
design-regime positive), and AB-Bind (antibody–antigen) — but none is a full generate→design→wet-lab loop;
the de-novo evidence is four targets. Effect sizes are modest (ΔAUROC ≈ 0.016–0.018), though their intervals
exclude zero and the central nugget (CPI 0.000) is a clean rather than a small result. The all-atom occlusion
baseline is a min-over-rotamer repacking proxy, not a full molecular-mechanics force field; the 95%-zero-clash
prevalence bounds how much any clash model could recover, but a physics force field could shift the baseline.
The de-novo binding labels convolve display and fold-stability with binding — the core/interface
stratification is the control, and the native is excluded so the parent-is-model-output bias is uniform across
strata. SKEMPI training leakage makes the predicted-backbone result *conservative* (the predictor nearly
reconstructs complexes it has seen and the deficit appears anyway). The pre-registered strict-control tier is
underpowered by design; the verdict rests on the higher-powered tiers declared in advance. And two extensions
did not survive their controls — a within-natural confidence gradient, and the generalisation to catalytic
sites — which we report rather than bury.

---
*Draft status: §1–9 in prose. Pending: fold in the Fable-5 catalytic audit (§4 note); a figure pass;
external-citation URL verification; final length trim to 9 pages.*
