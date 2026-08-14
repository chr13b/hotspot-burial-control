# Paper outline — Spine B (the distribution-knows-binding spine)

Reframed 2026-08-14 after the fail-fast Bennett-hardening battery (5/5 landed; results/PREREG_bennett_hardening.md,
FINDINGS_bennett_hardening.md). Every claim maps to a committed CSV; the `→` tags name the source. Target:
**ICLR main track** (fallback TMLR). ~9 pages + appendix. Venue decided ICLR/Spine B by the battery.

## Working title
**"Confidence is not competence: interface-hotspot 'blindness' in inverse folding is a conditioning-set
artifact — and what the model knows about binding lives in its distribution, not its confidence."**

## One-paragraph abstract
Staged binder design — generate a backbone, then inverse-fold a sequence — is believed to struggle at
interface *hotspots*; a published result (ProBID-Net) reports inverse-folding recovery 0.334 at hotspots vs
0.472 elsewhere. We report a diagnosis, a positive, and a mechanism. **(1) Confidence is not competence.**
Across **five inverse-folding architectures** (ProteinMPNN, ESM-IF1, PiFold, MIF, ProBID-Net) the model's
per-residue confidence ranks interface hotspots at chance (AUROC 0.50–0.54) and is **conditionally
independent** of hotspot-ness given cheap geometry — a combiner-free conditional-predictive-impact test gives
**CPI 0.000 [−0.0003,+0.0003]** — while a *free* geometric partner-contact feature (ΔSASA) predicts them; the
obvious learned detector (KL of the complex-vs-monomer distribution) merely **recapitulates ΔSASA** on all
four backbone classes (a *learned frustratometer*, demoted). **(2) But the model *does* know binding — in its
distribution, not its confidence.** On genuinely de-novo designed binders with **experimental site-saturation
binding data**, the per-substitution complex-conditioned distribution ranks the 19 substitutions by measured
binding (AUROC 0.615 > BLOSUM/volume/hydropathy), dissociates the fold-*stability* question (core 0.721) from
the *binding* question (a built-in positive control), gains specifically from conditioning on the partner
(P−Q +0.076, interface-specific), and — the decisive test — **adds beyond an all-atom, repacked occlusion
baseline** (+0.018 [+0.015,+0.022]; after rotamer repacking 95% of substitutions have *zero* steric clash, so
occlusion cannot explain it). A **non-parent** scorer (ESM-IF1, which did not generate the designs) reproduces
all three signals, and confidence tracks binding-leverage only in the binding-dominated de-novo regime (0.60
vs 0.54 on natural complexes) — the *constraint-vs-leverage* signature. **(3) The published deficit is a
burial confound**, and **the residual tax lives in the conditioning set** the field benchmarks on: under a
pre-registered burial-matched design the crystal deficit vanishes (five architectures + ProBID-Net's own
model, reproduce-and-dissolve), yet on backbones from two independent structure predictors (OpenFold3,
AF2-multimer) a matched deficit reappears and the two predictors' per-complex deficits agree (ρ=+0.57) —
agreement that *survives* burial control (partial ρ +0.53). **The mechanism:** crystal backbones, carved by
the side chains being predicted, hide the deficit; predicted backbones expose it — so *read hotspots from
partner geometry, not model confidence, and the model's binding knowledge lives in its full distribution, not
its scalar summaries.* Two competing mechanisms — a low-temperature constellation cost and a commitment-
ordering schedule — are each measured and refuted.

> **Reframe history (audit trail).** The paper began as a KL-detector method; KL was found to recapitulate
> ΔSASA on all four backbone classes (crystal +0.007, OF3 +0.008, AF2 +0.005, Bennett; R1 = c59688e) and the
> KL-triage claims (capture@k, kcal/mol) were baseline artifacts — **KL-as-method WITHDRAWN**. Reframed
> nugget-forward + Big-Idea-1-forward (167867d). The **fail-fast Bennett battery (2026-08-14)** then hardened
> the positive against a hostile review: T1 all-atom occlusion (P adds +0.018 beyond a *stronger* baseline;
> the old +0.025-over-weak-baseline is superseded), T2 non-parent ESM-IF1 (not circular), T4/R2 5-model
> (a property of inverse folding), T5 burial-residualized cross-predictor agreement, T3 constraint-vs-leverage.
> The within-SKEMPI confidence-decay *gradient* was attempted and DROPPED (underpowered categories; null on a
> powered continuous-affinity axis) — T3's de-novo-vs-natural contrast is the robust form.

## Section map

**1. Introduction.**
- The staged pipeline; the frustration hypothesis (hotspots are buried polars / strained rotamers that buy
  affinity, not stability); the intuition that inverse folding, lacking a binding-energy term, misses them.
  ProBID-Net's 0.334/0.472 gap and its dynamics attribution; the confound nobody controlled: hotspots are
  buried, and burial is where inverse folding is *most* confident. → BRIEF §2.
- Contributions:
  **(i) Confidence is not competence (the nugget).** Per-residue confidence ranks interface hotspots at
  chance across **five architectures** (0.50–0.54) and is **conditionally independent** of hotspot-ness given
  geometry (CPI 0.000); a *free* ΔSASA feature predicts them; KL = a demoted *learned frustratometer*.
  → nugget_cpi.csv, xmodel_confidence.csv, baseline_audit.csv.
  **(ii) The model knows binding, in its distribution not its confidence (the positive).** On de-novo designs
  with experimental SSM labels, the per-substitution distribution ranks binding (0.615), dissociates
  stability-vs-binding (core 0.721), gains from the partner (+0.076), and **adds beyond all-atom repacked
  occlusion** (+0.018); reproduced by a **non-parent** model (ESM-IF1); confidence tracks binding only in the
  de-novo regime (constraint-vs-leverage). → bennett_knows_where.csv, bennett_occlusion_allatom.csv,
  bennett_nonparent.csv, bennett_conf_fork.csv.
  **(iii) The published deficit is a burial confound.** Five architectures + ProBID-Net's own model; a
  pre-registered burial-matched matched-pair design (offered as a reusable diagnostic **protocol**).
  → probid_gap_estimators.csv, FINDINGS.md.
  **(iv) The residual tax lives in the conditioning set.** A burial-matched deficit appears on two independent
  predictors' backbones (cross-predictor ρ=+0.57, **survives burial residualization** partial ρ +0.53), absent
  on noised-crystal generative backbones (independent-reconstruction, not distance). → deficit_burial_residualize.csv.
  (Two competing mechanisms — constellation cost, commitment ordering — measured and refuted.)

**2. Setup and pre-registration.**
- SKEMPI 2.0; hotspot = Ala-scan ΔΔG>1 (strict >2 = ProBID-Net's threshold); null |ΔΔG|<0.25. Bennett-2023
  de-novo SSM (4 targets, experimental per-substitution binding) as the independent design-regime fixture.
- The matched-pair design (rSASA ±0.05 / pydssp SS / neighbour ±1) — the diagnostic protocol; complex-level
  bootstrap; seed 20260803. → PREREG.md, PREREG_knows_where.md, PREREG_bennett_hardening.md.
- Five architectures + validation gate (positive control on every path). → validate.py.

**3. Confidence is not competence.** *(the nugget)*
- The quantity a designer would trust is useless at hotspots: confidence AUROC 0.538, capture@3 0.064 <
  random 0.084. → baseline_audit.csv, confidence_antipredicts.csv.
- **Combiner-free:** CPI(confidence | burial+nbr+ΔSASA) = **0.000 [−0.0003,+0.0003]** — conditionally
  independent of hotspot-ness given structure; ΔSASA CPI +0.013 (adds), KL CPI +0.002 (tiny). → nugget_cpi.csv.
- **A property of inverse folding, not one network:** interface-hotspot confidence-AUROC is 0.50–0.54 across
  ProteinMPNN 0.538 / ESM-IF1 0.517 / PiFold 0.499 / MIF 0.509 / ProBID 0.536, all ~0.15–0.19 below burial.
  → xmodel_confidence.csv (T4/R2).
- **Free geometry is what predicts:** ΔSASA (partner-contact area) AUROC 0.585; the full cheap-geometry
  baseline (burial+nbr+ΔSASA) 0.734. The learned KL detector **recapitulates ΔSASA** on all four backbone
  classes (ΔAUROC over full geometry ≈ 0: crystal +0.007, OF3 +0.008, AF2 +0.005) — a *learned frustratometer*
  (ONE paragraph; cite Frustratometer + BAIF). → kl_geometry_control{,_predicted}.csv.
- **Designer table** (baseline_audit.csv): single-signal AUROC / capture@3 — random, nbr, burial, ΔSASA, KL,
  confidence (below random), full geometry — the "what a designer actually needs" table.
- **Figure 1 (the nugget):** designer table + confidence-below-random bar, with the 5-model spread inset.

**4. The model knows binding — in its distribution, not its confidence.** *(the positive; NEW dedicated section)*
- Pre-registered (PREREG_knows_where.md, 287f884). Bennett de-novo SSM, 60,971 (position,substitution) pairs,
  73 designs; positive control: SSM-excluded aa == PDB-native = 1.000.
- **P1** interface AUROC(P) = 0.615 [0.601,0.628] > 0.5, and **beats every sequence baseline** (BLOSUM 0.589,
  hydropathy 0.579, volume 0.539) — not a substitution-similarity matrix.
- **P2 (dissociation / positive control)** core AUROC(P) 0.721 vs interface 0.615 (Δ +0.107, non-overlapping):
  the model answers the *stability* question better than the *binding* question, as it must.
- **P3 (partner-conditioning)** interface AUROC(P) − AUROC(Q) = +0.076 [+0.068,+0.084], interface-specific
  (P≈Q at core/surface). → bennett_knows_where.csv.
- **The decisive test — beyond ALL-ATOM occlusion (T1, the venue decider).** Rebuilt occlusion as a real
  all-atom min-over-rotamer vdW clash (rdkit rotamers, Kabsch onto backbone, vs partner heavy atoms;
  pre-registered RMSD validity gate passed at 0.278 Å). After repacking, **95.1% of substitutions have zero
  clash** (occlusion near-vacuous; clash-standalone AUROC 0.519 = chance); on a *stronger* all-atom geometry
  baseline (0.619 > the old 0.587) **P still adds ΔAUROC +0.0182 [+0.0145,+0.0220], P(>0)=1.000**. The model
  encodes per-substitution binding **energetics beyond all-atom steric occlusion.** (Supersedes the old
  +0.025-over-weak-baseline.) → bennett_occlusion_allatom.csv, FINDINGS_occlusion_allatom.md.
- **Not circular (T2).** ESM-IF1 — which did NOT generate the ProteinMPNN parents — reproduces all three:
  interface AUROC(P) 0.625, P−Q +0.079, ΔAUROC over all-atom occlusion +0.016 (all P=1.000). → bennett_nonparent.csv.
- **Constraint-vs-leverage (T3).** Confidence predicts binding-hotspots above chance in the binding-dominated
  de-novo regime (logp 0.596 / negentropy 0.627) but at chance on natural SKEMPI (0.538): confidence estimates
  positional *constraint*, which coincides with binding *leverage* only when selection is binding-dominated.
  (The finer within-SKEMPI gradient was tested and dropped — underpowered/null; reported honestly.)
  → bennett_conf_fork.csv, confidence_gradient{,_affinity}.csv.
- **Figure 2 (the positive):** LEFT core-vs-interface dissociation (0.721 vs 0.615) with sequence baselines;
  RIGHT P adds +0.018 over the all-atom occlusion baseline, with the 95%-zero-clash inset.

**5. On crystal backbones, the hotspot gap is a burial artifact.** *(the correction)*
- The confound drawn: recovery rises with hotspot strength AND burial rises in lockstep. → FINDINGS.md §2.1.
- Matched: SECONDARY-B −0.042 [−0.222,+0.129]; regression estimator +0.059 [−0.051,+0.167]; **five
  architectures agree** (every PRIMARY CI contains zero); junction control on single-chain models.
- **ProBID-Net's own released voxel-CNN — reproduce-and-dissolve** (audited + re-verified). Port faithful
  (0.472 = their non-hotspot number); their deficit reproduces on comprehensively-scanned complexes (≥5
  hotspots −0.113, p=0.007) and **dissolves under composition+burial matching** (AA-matched +0.120,
  burial-matched −0.038, every CI spans zero). Composition is stereotyped across models (ρ 0.87–0.90) but
  burial is the dominant confound here. → probid_gap_estimators.csv, composition_confound.csv, §4.2e.
- Matched-pair design offered as a reusable **diagnostic protocol** (not a benchmark; ProtDBench exists).
- **Figure 3:** the burial confound (recovery vs burial by hotspot class) + the matched-pair forest across 5 models.

**6. The tax appears on independently-reconstructed backbones — and it is predictor-general.** *(conditioning-set localization)*
- **Cross-predictor reproducibility is the headline.** OpenFold3 (Exp A) and AF2-multimer (Exp D) each show a
  burial-matched deficit (−0.191 [−0.37,−0.004]; −0.233 [−0.44,−0.035]; crystal ≈ 0), and the two predictors'
  per-complex deficits **agree ρ=+0.565 [+0.40,+0.71], n=127** — the same complexes are hard under both. Not
  an ensemble; two independently-trained predictors agreeing. → FINDINGS_expA.md, FINDINGS_expD.md.
- **Survives burial (T5, kill-shot #5).** The agreement is not recursive burial: partial ρ(d_of3,d_af2 |
  interface burial) = +0.529 [0.354,0.678], and +0.533 after dropping the shared top-3 leverage complexes.
  → deficit_burial_residualize.csv.
- **Honest framing up front:** each single-predictor deficit is marginal (neither survives dropping its top-3);
  the claim rests on cross-predictor agreement, symmetric leverage jackknife applied identically. → FINDINGS_expD.md §5.
- **Independent-reconstruction, not distance-from-native:** on partial-diffusion noised crystals at equal iRMSD
  the deficit is ABSENT. → FINDINGS_expC2.md.
- **Figure 4:** the AF2-vs-OF3 per-complex deficit scatter (ρ=+0.57), the transferable signal.

**7. It is the conditioning geometry, not the schedule or the sample budget.** *(ruling out competitors — condensed)*
- N_hot: the T=0.1 constellation cost (~10^10) is identical at burial-matched control constellations (median
  Δ 0.000, p 0.90) — generic low-temperature sampling, not a hotspot tax.
- Commitment ordering: ProteinMPNN oracle order inert (DiD −0.002); MultiFlow ordering knob inert (F4). Ruled
  out on an autoregressive AND a coupled model. → FINDINGS_expB.md. (Moves toward appendix if space is tight.)

**8. Related work and positioning.**
- ProBID-Net (phenomenon; we correct the attribution). RedNet (blindness is conditioning, not decoding).
  MultiFlow (ordering knob inert). *Frustratometer* (Ferreiro/Parra) — KL is a learned frustratometer, and
  KL≈ΔSASA means the neural version does not beat classical geometry. *BAIF* (arXiv 2410.09543) — bound-vs-
  unbound IF log-likelihood for ΔΔG; **closest prior art** — we differ: a per-*substitution* experimental-
  binding test of the full distribution beyond an all-atom occlusion baseline (Big Idea 1), not a ΔΔG cycle;
  and we keep KL demoted, not a method. *StaB-ddG* (2507.05502) — folding-energy ΔΔG, distinct question.
- **BindCraft hook:** the leading one-shot pipeline hard-codes a **4 Å interface freeze forbidding inverse
  folding at the interface** — the field's implicit admission of our thesis; we give it its measurement and a
  principled reading (read partner geometry / the full distribution, not confidence).
- **Conditioning-aware IF (concurrent):** DeSAE (2506.08365), target-conditioned IF, UMA-Inverse — methods
  that *presuppose* the conditioning-set problem; we *measure* it and show the benchmark hides it. ✎ verify
  all search-only URLs before submission (BAIF/DeSAE/CPI/free-energy-interp 2506.05596 fetched; rest ✎).

**9. Limitations (stated up front).**
- Two fixtures (SKEMPI main + Bennett de-novo) of different character; still no full generate→design→wet-lab
  loop (a second SKEMPI-class fixture, AB-Bind/AbBiBench, is the obvious floor-raiser — queued).
- Effect sizes are modest (ΔAUROC ~0.016–0.018) though tight (CIs exclude 0); the CPI 0.000 is a clean result.
- All-atom occlusion is a min-over-rotamer repack proxy, not a full force field (95%-zero prevalence bounds
  what any clash model could capture). Bennett labels convolve display/fold-stability with binding — the
  core/interface stratification is the control; native excluded so the parent-is-model-output bias is uniform.
- SKEMPI training leakage makes the predicted-backbone result *conservative*; strict-control PRIMARY tier
  underpowered, verdict rests on higher-powered tiers (declared in advance).

## Figure inventory (4 main + appendix)
1. **The nugget:** designer table (confidence below random, geometry best, CPI 0.000) + 5-model spread. *(have — baseline_audit, xmodel_confidence, nugget_cpi)*
2. **The positive:** Big-Idea-1 core-vs-interface dissociation + P adds +0.018 over all-atom occlusion (95%-zero-clash inset). *(have — bennett_knows_where, bennett_occlusion_allatom)*
3. **The correction:** burial confound + 5-model matched-pair forest. *(have)*
4. **Conditioning-set:** AF2-vs-OF3 per-complex deficit scatter ρ=+0.57 (survives burial). *(have — deficit_burial_residualize)*
Appendix: per-model panel, decoding-order spread, TOST/Holm, external ΔΔG validation, T2 non-parent full table,
constraint-vs-leverage gradient (exploratory/null), N_hot + commitment-ordering nulls, junction sensitivity.
**RETIRED:** the old "money figure" (deficit-vs-backbone-distance), all KL-triage capture@k / kcal-mol panels.

## What must land before submission
- **Prose pass:** turn this section map into 9 pages of prose (elements-of-style). Highest priority.
- Verify all ✎ search-only citation URLs (§8). Zenodo DOI at submission (task #11).
- Optional floor-raiser: **AB-Bind/AbBiBench** second SKEMPI-class fixture (kills "single fixture"; CPU).
- Optional ceiling: external **obligate/transient** dataset for the constraint-vs-leverage gradient (higher-
  risk after the within-SKEMPI affinity null; research agent scoping).
- Optional field-level: generalize "confidence≠competence" to **catalytic residues** (M-CSA) — upgrades the
  nugget from binding-hotspots to functional-sites-in-general (novelty check pending).

## Honest self-assessment of the ICLR case
FOR: a rigorous, pre-registered **burial correction** (the benchmark hides the effect); an **airtight
combiner-free nugget** (CPI 0.000, a property across 5 architectures); a **genuine positive on de-novo designs
with experimental labels** that survives an all-atom repacked occlusion baseline (+0.018), a non-parent scorer,
and carries a confirmed **constraint-vs-leverage** endpoint; cross-predictor reproducibility that survives
burial; competing mechanisms adjudicated; unusually disciplined pre-registration and self-correction (KL-as-
method withdrawn; the within-SKEMPI gradient dropped as null — both reported straight). AGAINST: one *main*
fixture (SKEMPI) + a small de-novo one (Bennett, 4 targets); modest effect sizes; prose + figures not yet
written. **NET (post-battery, 2026-08-14): TMLR strong (~0.85); ICLR ~0.55–0.58** — earned by a decisive
pre-registered test passing, not assumed. Ceiling levers remaining: AB-Bind (floor), catalytic-residue
generalization (field-level), external obligate data (gradient) — all optional; the binding constraint is now
the prose.
