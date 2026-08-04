# Innovation lineage — High-impact paper: flow matching / diffusion for molecular design

> Directed-evolution ideation run (LITE mode, research profile).

**Goal:** A genuinely novel ML-method contribution at the intersection of flow matching / diffusion generative models and molecular design (small-molecule SBDD and/or protein-peptide binder design), publishable at NeurIPS / ICML / ICLR.

**Constraints (from intake):**
- **Compute: 1-2 GPUs.** Cannot win by scale, so the paper must win on insight. Biases toward inference-time methods, diagnosed-phenomenon-plus-cheap-fix, path/objective changes visible at small scale, and theory validated on controlled systems.
- **Data: public benchmarks only** (GEOM-Drugs, QM9, CrossDocked2020, PDBBind, PoseBusters, PDB, PepBDB, binder benchmarks). **No wet lab**, so every claim must be checkable in silico.
- Solo author, blank slate, no preexisting results or hunch.
- Target: open to both small-molecule SBDD and protein/peptide binder design.
- Paper type: ML method paper (not a benchmark, not a domain application).

**Profile:** research · **Mode:** lite
**Started:** 2026-06-20

**Saturation warning recorded up front:** this is one of the most crowded areas in ML (2023-2026). The run's own history says that in saturated fields the loop delivers *vetting* rather than greenfield novelty. Breadth at generation plus a hard two-angle prior-art sweep is the defence. An honest null is an acceptable outcome.

---

## L1 — Candidate population (8 mutators, one per lens)

Lenses assigned, chosen to span the space rather than cluster. Each mutator ran toolless
(generation is cheap and ungrounded by design, L2 does the grounding) with the frontier
anchored negatively: an explicit do-not-restate list of DiffDock, TargetDiff, DiffSBDD,
Pocket2Mol, DecompDiff, EquiFM, SemlaFlow, RFdiffusion, FrameDiff, FrameFlow, FoldFlow,
MultiFlow, Chroma, ProteinMPNN, AlphaFold3, Boltz, Chai, discrete flow matching,
Riemannian flow matching, stochastic interpolants, classifier-free guidance, minibatch-OT
couplings, consistency and distillation models.

1. **Probability-path / interpolant mechanism** — attack the flow-matching machinery itself: source distribution, linear interpolant, coupling, per-modality noise schedules, conditional-expectation averaging.
2. **Inversion / contrarian** — invert a field consensus (equivariance necessity, distribution-matching as the objective, generate-then-filter, expressivity as the binding constraint, joint all-atom generation).
3. **Distant-domain mechanism transfer** — mechanical transfer from statistical physics, control theory, coding theory, developmental biology, motion planning, metallurgy, compilers, or evolution.
4. **Test-time compute allocation** — where inference compute goes: early abandonment, adaptive steps, trajectory search, particle filtering, test-time scaling laws. Best structural fit for the 1-2 GPU constraint.
5. **Evaluation-derived method** — derive a method (not a benchmark) from known metric pathologies: Vina exploitation, PoseBusters failures, CrossDocked/PDBBind leakage, self-consistency proxies sharing failure modes with the generator.
6. **Trajectory timing** — when each design property is committed during generation, measured by branch-and-resample; rewind rather than restart; guidance concentrated where it has leverage.
7. **Constraint inversion (small-scale advantage)** — what method is only possible at small scale and invisible to a 1000-GPU lab: exact physics priors over learned capacity, per-target inference optimization, bending the data-efficiency curve.
8. **Wild lane** — ignore all of the above framings.

### L1 results — the eight candidates

**1. Resolution-time mismatch (probability path).** The isotropic source `Σ=I` is not neutral, it is a *scheduler*. A decision separating two hypotheses by Euclidean distance δ becomes identifiable at `t*/(1-t*) ≈ √λ/δ`, so the path releases information in order of atomic displacement magnitude. Chemically decisive but geometrically small decisions (enantiomers, ring pucker, amide cis/trans) therefore become identifiable only as t→1, where the field is stiffest, fewest steps land, and no transport budget remains. Fix: graph-conditioned anisotropic source covariance under a fixed log-det budget, redistributing prior entropy rather than shrinking it. Distinctive predictions: error collapse in `N_steps·(1−t_id)`; solver-dependent ring-pucker branch ratios.

**2. Conditioning-capacity inverted-U (inversion).** Generalization to unseen pockets is non-monotonic in the pocket→ligand channel rate, optimum around 10 bits, and the field is pushing the wrong way. CrossDocked fits `p(ligand|pocket)` at roughly one sample per condition, so the loss-minimizing solution for a high-capacity map is recognition (retrieve nearest training pocket, emit its ligand), not complementarity. Method: VIB bottleneck reading the rate in bits, swept; measure pocket-swap AUROC (re-dock the same ligand into volume-matched decoy pockets, which cancels size and lipophilicity bias exactly). Accepts that the best model must be trained worse and that validation loss is a misleading selection signal.

**3. Saddle-ridge straddle (distant-domain: motion planning).** The marginal velocity is a posterior mean, so under a multimodal posterior the trajectory tracks the medial axis between valid modes, and valid molecular geometry is non-convex, so between-modes is invalid. Detect mid-trajectory from the denoiser's own second difference (3 evals, no oracle, no reward model); the first difference simultaneously hands you the escape direction. Repair by one kick along it, random sign. Transferred from the bridge test for narrow passages in probabilistic-roadmap planning. Has a P0 premise check needing no model at all.

**4. Option-value compute allocation (test-time compute).** SBDD is scored by a *max* over samples, not an expectation, so reward-tilted SMC targets the wrong objective and discards exactly the high-variance particles carrying the upper tail. Allocate steps by a Weitzman reservation index `z = μ − kσ`, with μ a cheap reward on the free endpoint prediction and σ from the realized quadratic variation of that reward (justified by the endpoint being a Doob martingale). Best-of-N and greedy verifier selection are the two limits; the claim is the optimum is interior. Fork crossover located by intraclass correlation.

**5. Basin width, not depth (evaluation-derived).** Docking-score hacking is one omitted physical term. `F ≈ U(x*) + (kT/2)·log det H`: the score reports basin *depth* and discards basin *width*, so the proxy's null space is exactly "needle-like minima", predicting the observed large-greasy-molecule pathology. Gaussian smoothing at thermal scale recovers the missing term to second order. Guidance uses `σ_eff(t)² = max(0, σ_thermal² − Var[x₀|x_t])`, scheduled against the model's own posterior variance. Carries a dimension-free bound `‖∇F_σ‖ ≤ R/(σ√(2π))` — a certified ceiling on judge exploitation. Phase A is checkable on already-released public samples with no training.

**6. Chirality as an uninformed coin flip (trajectory timing).** Parity (signed chiral volume) is smooth and low-frequency so it commits early; substituent atom identities are discrete and high-entropy so they resolve late — meaning handedness is decided before the information determining it exists, and is then irreversible because flipping requires passing through the planar degenerate state. Instrument: distance-only E(3) nets are exactly mirror-symmetric on the ligand marginal, so reflect-and-realign builds a free exact parity counterfactual. Intervention: mirror-fork at the measured commitment time, ~1.4x cost versus 2x for independent restarts. Has a cheap kill-first pilot (do enantiomer pairs even differ in docking score?).

**7. Kill the protein encoder (small-scale advantage).** A rigid pocket acts on a ligand as a fixed external potential that is a property of physics, not of the dataset, and is exactly computable in seconds as AutoGrid-style field channels. Delete the learned encoder; condition a ligand-only flow on ~6 field channels plus exact gradients read pointwise at the *current noisy* coordinates. Sharply distinguished from energy guidance, which must approximate an expectation over the denoising posterior and degrades at high noise, whereas field readout is exact at every t. Predicts an encoder advantage only above ~30-40% pocket similarity, i.e. only where it is effectively retrieving.

**8. Non-conservative velocity fields (wild lane).** For Gaussian paths the true marginal velocity is exactly `a(t)x + b(t)∇log p_t` — linear plus gradient — so its Jacobian must be symmetric and its circulation around any closed loop exactly zero. Nothing enforces this in a trained net, and attention is structurally non-reciprocal (Newton's third law violated inside the model, an active-matter system with circulating currents). Any measured curl is therefore *certified error with no ground truth required*. Probe: small closed loops in two named dihedrals, with an analytic zero-curl noise floor control. Repair: project the frozen field onto gradient fields (weighted Helmholtz decomposition). Free byproduct: a differentiable pocket-conditioned energy with zero affinity labels.

**Observed convergence:** ideas 1, 3 and 6 arrived from three unrelated lenses (path mechanics, motion-planning transfer, trajectory timing) at overlapping territory — multimodal posteriors and *when* a decision becomes irreversible. Ideas 2 and 7 converge independently on the claim that the learned pocket encoder is doing less useful work than assumed. Convergence from independent lenses is weak evidence the territory is real, but it also raises the duplication risk that triage must resolve.

## L2 — Adversarial prior-art sweep (3 clustered sweeps, two angles each, 217 fetches total)

| # | Idea | Verdict | Closest prior art (fetched) | Genuinely unshipped | Cost | Most likely failure |
|---|------|---------|------------------------------|---------------------|------|---------------------|
| 6 | Chirality commitment + mirror-fork | **PARTIALLY-DONE on components, NOVEL as the composed claim** | Raya & Ambrogioni SSB; Sclocchi forward-backward; Synchronization Gap (Mar 2026); "E(3) models cannot learn chirality"; ET-Flow post-hoc mirroring; Antithetic Noise (t=0 only) | t_parity < t_identity measured in a molecular generative model; ligand-only reflection with pocket fixed as a free exact counterfactual; fork at measured t* | **Low** (measurement only, frozen checkpoints) | competitive generators are deliberately chirality-aware (SE(3), cross products), so reflection stops being exact |
| 5 | Basin width not depth | PARTIALLY-DONE → escalate | **Chang/Chen/Gilson PNAS 2007** ("narrower energy wells", scoring functions neglect it); OpenEye Chemgauss (Gaussian-convolved potentials, shipped); Reward Sharpness-Aware FT (CVPR 2026); PoseCheck | the generated-vs-crystal differential under thermal smoothing; σ_eff(t) scheduled against posterior variance; the unification claim PoseCheck declines to make | **Very low (CPU-only)** | Vina terms already piecewise-smooth and crystal ligands sit in Vina's own sharp minima → differential vanishes, reduces to known MW/logP bias |
| 2 | Conditioning-capacity inverted-U | PARTIALLY-DONE → escalate | Delta Score (pocket-swap metric shipped; only 2/20 scored best on true target); Gu et al. memorization-from-conditioning; ligand-centric baselines beating 3D SBDD | generalization non-monotonic in a *measured bit rate*; validation loss anti-correlated with specificity beyond the optimum | Medium | measured KL is the chosen encoder's rate, not the true channel's → "10 bits" is architecture-dependent, U flattens below AUROC noise |
| 1 | Resolution-time mismatch | PARTIALLY-DONE → escalate | EigenFold ("cascading resolution along eigenmodes"); HarmonicFlow; ET-Flow; **FLOWR.root v6** (anisotropic priors with *trace* normalization); Carré du champ FM; critical windows | fixed **log-det** (vs FLOWR's trace) = entropy-neutral reallocation; per-decision-class identifiability as an explicit objective; the N_steps·(1−t_id) collapse | High (retraining) | **FlowMol's published ablation says structured priors underperform Gaussian on this exact task family** — points the wrong way |
| 4 | Option-value compute allocation | PARTIALLY-DONE → escalate | Optimal Stopping vs Best-of-N (Oct 2025, Weitzman index); **SVDD-PM** (free reward on x̂₀, on QuickVina-scored molecules; its α already interpolates greedy↔uniform and greedy is their default); Diffusion Tree Sampling DTS*; VASR-MAX | quadratic-variation σ as the allocation signal; step-level reservation index; ICC(t) breadth-vs-depth crossover | High | reward on x̂₁ correlates too weakly at high noise; index collapses to greedy-or-uniform inside seed variance |
| 3 | Saddle-ridge straddle | **ALREADY-DONE** → thin slice | **RODS** (NeurIPS 2025: finite-difference curvature probe every step, detect-then-perturb, 70%/25%); Mode Interpolation (oracle-free trajectory-variance detector, >95%); Laplacian Score Sharpening; molecular uncertainty estimation (Jun 2026) | vector-valued 2nd difference of x̂₁ (all priors are scalar/trace); Kabsch alignment making it well-posed under equivariance; NFE-matched; molecular | Medium | 2nd difference dominated by ordinary curvature everywhere — exactly why RODS needed adversarial max-over-ball instead of a plain difference |
| 7 | Kill the protein encoder | PARTIALLY-DONE → escalate | **GRID/Goodford 1985** (74-probe classical interaction fields driving de novo design); LiGAN (voxel but occupancy-only + learned encoder); VoxBind | only the ablation *with a signed prediction*: the 30-40% homology crossover | **Highest** (from-scratch training) | at high noise the ligand sits outside the pocket where all fields are flat → no conditioning exactly when the scaffold is decided |
| 8 | Non-conservative velocity fields | **ALREADY-DONE + headline interpretation contradicted** | Chao et al. ICML 2023 (Asym/NAsym via Hutchinson, verbatim); **Bigi et al. ICML 2025 "dark side of the forces"** (closed-path work + λ=‖J_anti‖/‖J‖); Thornton et al. (the Helmholtz-projection repair, named as such); **Khelifa 2026** (solenoidal error structurally invisible to marginals); **Horvat & Pfister** (conservativity neither necessary nor sufficient) | 3 thin residuals: non-contractible torsion-torus loop; conditional-vs-unconditional curl contrast; φ as label-free pose ranker | Low | you measure real curl, project it out, and nothing changes — the literature predicts this outcome |

### Structural findings from the sweep

- **Ideas 3 and 6 are near-duplicates in mechanism.** Idea 6 is the *controlled special case* of idea 3 where the two modes are known a priori to be exact reflections, which is what makes the direction free and the counterfactual exact. Presenting both would read as padding.
- **Idea 6's measurement is diagnostic for idea 1's premise too.** If parity does not commit before identity, idea 6 has no target and idea 1 loses its motivating example. So idea 6's probe is the cheap test of a premise two ideas share.
- **The convergence noticed at L1 was real but pre-existing.** Speciation ordering and critical windows are established literature; the run rediscovered a known frontier from three lenses rather than finding new ground. What survived is narrower and more specific than the L1 write-ups claimed.

## L3 — Survivors

**KEEP (in priority order):**
1. **Idea 6 — chirality commitment time + mirror-fork.** The only NOVEL-as-composed verdict. Cheapest to run, no retraining, produces a publishable object (the t_parity vs t_identity entropy curves) *even if the fork intervention fails*, and de-risks idea 1 for free.
2. **Idea 5 — basin width, not depth.** Narrowest surviving claim but the best insight-per-franc: Phase A is a CPU-only day on already-released public samples and returns a signed, pre-registerable number.
3. **Idea 2 — conditioning-capacity inverted-U.** Most conceptually ambitious survivor; the direction-of-effect inversion survived every search. **Open scoop risk:** the sweep could not decode the MSCoD and IBEX PDFs, so whether either already bottlenecks the pocket channel needs a manual check before committing.

**DROP:** idea 8 (already done *and* the interpretation is refuted by fetched work), idea 3 (subsumed by idea 6, and RODS ships the general version), idea 7 (1985 ancestor, most expensive), idea 4 (three of five components already published, one of them in this exact domain), idea 1 (published ablation points the wrong way; its premise rides on idea 6's measurement anyway).

**Meta-finding:** 8/8 came back partially- or already-done, consistent with the saturation warning recorded up front. Unlike the Swiss product run, however, two ideas carry *precisely stated* unshipped sub-claims and one earned a NOVEL-as-composed verdict, so this is not a null.

---

# ESCALATION RUN — Survivor 1 into the full loop + COURT MODE

Seeded per the escalation mechanics: the lite survivor is round-0 lineage, recon is pointed at its load-bearing claims rather than the whole field, round 1 is exploit-heavy, and court mode is enabled because the make-or-break question is factual.

## Step 0.5 — Targeted recon (65 fetches, model source code read)

### The reflection-symmetry table (the make-or-break answer)

| Model | Verdict | Deciding feature (read in source) |
|---|---|---|
| TargetDiff | **E(3) reflection-blind** | `uni_transformer.py`: scalar `dist` → GaussianSmearing only; zero cross/det/frame/spherical-harmonic ops |
| DecompDiff | **E(3) reflection-blind** | its `torch.cross` is immediately `.norm()`'d → `atan2` = *unsigned* bond angle, reflection-INVARIANT (a red herring) |
| MolCRAFT | **E(3) reflection-blind** | reuses the identical TargetDiff UniTransformer |
| **DiffSBDD** | **SE(3) CHIRALITY-AWARE as released** | `coord2cross()` pseudovector added to the coord update; all 8 released configs set `reflection_equivariant: False`, run_names `SE3-*` |
| Pocket2Mol | O(3) MP, head not exactly invariant; autoregressive so **no noise level t** | Vector Neurons, no cross products |
| FLOWR | **E(3) reflection-blind** | zero cross products in 6409 lines; spherical-harmonic parity selection rule |
| FlowMol / FlowMol3 | SE(3) chirality-aware | explicit `torch.linalg.cross` in `gvp.py`; paper states it outright |
| SemlaFlow | E(3) reflection-blind | normalized difference vectors + Gram matrix, all O(3)-invariant |
| ET-Flow | **both variants shipped** | `so3_equivariant` flag; `o3` configs use `parity_switch: post_hoc` |

**The instrument is exact and free on TargetDiff, DecompDiff, MolCRAFT and FLOWR — precisely the CrossDocked SOTA line.** Sharper than the seed stated: reflecting the ligand and Kabsch-realigning leaves every ligand-INTERNAL input bit-identical; only ligand-pocket cross-distances change.

### Findings that changed the idea
1. **The seed's assumption about DiffSBDD was flatly wrong** — it is the one pocket-conditioned model that is deliberately chirality-aware. Caught only by reading the 8 config files. Converts it from target to **matched control**.
2. **The premise is already measured for unconditional generation.** GeoDiff chirality inconsistency **0.500** (15,763 molecules × 10 conformers), 0.488 on experimental CSD; ETKDGv3 **0.000**, GeoMol 0.032. Corroborated by Torsional Diffusion's parity ablation and ChIRo (SchNet 54.5%, DimeNet++ 65.7%, ChIRo 98.5%). **So the phenomenon can no longer BE the contribution.**
3. **The field's evaluation is structurally blind to this.** PoseBusters' `identity` module (sole source of `stereo_tetrahedral`/`stereo_dbond`) requires `mol_true`, appears only in `redock.yml`/`regen.yml`, and is absent from `mol.yml`/`gen.yml`/`dock.yml`; `cli.py` routes de novo generation to `mode="dock"`. Zero chirality metrics across 11 flagship papers/repos; `DiffSBDD/analysis/metrics.py` calls `Chem.RemoveStereochemistry`. **Every headline "PoseBusters-valid" number in SBDD contains zero stereochemistry information.**
4. **Feasibility better than assumed:** released samplers already dump `pred_ligand_pos_traj` with one immutable index tensor across all 1000 steps. No code change needed.
5. **FMG Proposition 1 does not kill the pocket-conditioned case** — it governs *unconditional* E(3)-invariant distributions. Inside a fixed chiral pocket the cross-distance set differs between hands, so these models CAN express a chiral preference. Must be stated in paragraph one or a reviewer kills the paper on Prop 1 alone.
6. **Commitment-time work has never touched a chemical invariant**, and every branching method picks its fork time by fixed schedule or grid search — forking at a *measured* commitment time is unclaimed.

### The three dead-ends
- **D1 — the selector, not the instrument, is the weakest link.** Vina separates enantiomers above *seed* noise (~17% of ChIRo's 200K pairs exceed 0.3 kcal/mol; DOCKSTRING floor ≤0.1) but ranks them **below chance against experiment**: 141 pairs, 7 targets, Glide 22-25%, **Vina 13.8% vs a 33.3% random baseline**, "Our test failed for all modes and targets". Vina's accuracy floor is 2.85 kcal/mol ≈ 10× a typical enantiomer delta. Live lead: **gnina's CNN is chirality-sensitive by construction and untested on this.**
- **D2 — the measurement may come back trivially early.** Parity may be set at initialization by the handedness of the Gaussian draw. Pre-register the null: require DiffSBDD (aware) and TargetDiff (blind) to differ.
- **D3 — chirality-aware ≠ chirality-good.** FMG Fig 3: a reflection-variant model generated "tetrahedrons of approximately 0 volume" — collapsing onto the very degenerate state the barrier argument relies on.

## Round 1 — exploit-heavy mutation (5 lanes, cards drawn with real entropy)

| Lane | Card | Result | Layer | Verdict on seed |
|---|---|---|---|---|
| **M1** selector | Cross-breed · "lighthouse" · sommelier tasting | **Triangle-test reframe.** Published enantiomer failures measure *affinity vs assay truth*; instead measure *discrimination vs deposited hand*, where chance is exactly 0.5 because the negative is constructed. Paired ELBO gap Δ with **mirrored common random numbers** makes the ligand-internal contribution **exactly zero by symmetry**. Free calibration: mirror ligand AND pocket → Δ must be 0 to float precision. | Scoring | sharpens instrument, **replaces selector**, demotes fork |
| **M2** does the pocket steer | Adapt · "ferry" · monastic scriptoria | **A(t) dependence test with a DETERMINISTIC null.** Not an accuracy question (no ground truth exists) but a dependence question. No pocket influence → sampler commutes with reflection → mirrored run is bit-exactly the mirror → A(t)=0 every pair. Decoy-pocket arm separates pocket complementarity from a learned ligand-intrinsic prior (conjunctive vs polygenetic agreement). Corrector: sampler *draws* parity, user wants the *mode*. Pre-registered **abort** if the aware control also returns A≈0. | Generator | **REPLACE** |
| **M3** evaluation blind spot | Make-it-strange · "crucible" · pidgin trade languages | **"The field evaluates the mirror quotient."** Predicates partition by arity; de novo generation admits only reference-free ones, so stereo/bond-order/tautomer sit at the model's prior. "Just turn it on" is unavailable — the check is *inapplicable*, not disabled. Constructive exploit: adversarially flip stereocenters within |ΔVina|<0.3 and reproduce the published table. Oracle: epimeric contrastive learning gives 2^k−1 **free** negatives per complex. | Evaluation | COMPLEMENT, and **adversarial** — it is the cheapest competitor the fork must beat |
| **M4** fresh explore | Steal-mechanism · "ferry" · beekeeping | **Parity as a conserved charge.** {ν=0} is codimension-1 and contains **no real molecules** — a data void where drift is pure extrapolation. Unifies two facts: blind models never cross (crisp tetrahedra, 0.500 = *correct by abstention*); aware models must cross and strand mass mid-crossing (*wrong by traffic jam*). Gate ν̇ = ⟨∇ν,v⟩ (closed form) + discrete branch-transposition jumps that flip parity exactly while never occupying ν=0, reaching all 2^K diastereomers. | Sampler dynamics | NOT the seed |
| **M5** wild | Make-it-strange · "marginalia" · beekeeper smoke | **Delete the anti-informative subspace.** S₋ = ½[S(x)−S(x̄)] IS the reward's entire opinion on handedness, and it is *inverted*, not weak — sharp-and-inverted is worse than noisy because optimization confidently descends the wrong way and more compute amplifies it. Orbit-average it to exactly zero, defer the bit to ETKDGv3 (~100%). Detector: Vina is exactly invariant under whole-complex improper transform, so **mirror inconsistency is a certified-zero channel** — the field benchmarked against seed noise, which is the wrong null. | Objective | merge-candidate with M3 |

**Triage:** finalists **M2 · M3 · M4** (spanning generator / evaluation / sampler). M1 folds into M2 as its external-validity arm (shared paired-counterfactual instrument). M5 folds into M3 — M3 says the evaluation cannot *see* parity, M5 says the one metric that can see it points *backwards*; together they indict the evaluation layer more completely than either alone.

## Court-mode debates, rebuttals, consolidated sweep, and neutral fact-check

### Neutral fact-checker adjudication (primary sources; highest evidentiary weight)

| # | Claim | Verdict |
|---|---|---|
| 1 | Vina/Glide fail on enantiomers | **TRUE.** Ramírez & Caballero, *IJMS* 2016 17(4):525. 141 pairs, 7 targets. Glide HTVS 25.19 / SP 22.14 / XP 25.53, **Vina 13.81%**, vs a **33.33%** baseline. Quote verbatim: *"Our test failed for all modes and targets…"*. **Two agents had wrongly declared this unsourceable** — an artifact of the exhausted search budget, not absence. |
| 2 | One paper supplies both the 0.500 benchmark and an in-denoiser fix | **TRUE.** arXiv 2403.07925 (Williams & Inala, method = **PIDM**; "Nobias" is their employer). GeoDiff 0.500, GeoMol 0.032, ETKDGv3 0.000, CSD 0.488; PIDM's own rate 0.013–0.057. **Kills any "nobody fixes this inside the denoiser" framing.** |
| 3 | FMG "0-volume tetrahedra" | **Quote true, evidence overstated.** ICLR 2025; Prop 1 scoped to the *joint/unconditional* distribution. Evidence = **48 synthetic toy molecules**, KDE plot, **no rate**. |
| 4 | arXiv 2204.02513 pre-empts | **FALSE.** Measures stereocenter *counts* on a model that "yields contact pairs, not coordinates"; enumerates enantiomers post hoc. A *confirming instance*. |
| 5 | ET-Flow post-hoc beats SO(3) | **TRUE.** Global whole-molecule z-flip vs RDKit tags; precision coverage 74.38 vs 67.27. **Trick credited to GeoMol (2021)** — not novel to ET-Flow. |
| 6 | PoseBusters configs | **TRUE.** `dock.yml` has protein-contact checks but **no** stereo checks; only `redock.yml` (needs `mol_true`) has them; **all three** set `assign_stereo: False`. |

### Judge ruling

| Finalist | Novelty | Testability | Mechanism | Significance | Verdict |
|---|:-:|:-:|:-:|:-:|---|
| **F2 — the mirror quotient** | 4 | 5 | 4 | 4 | **KEEP** |
| **F1 — pocket-steering test** | 3 | 4 | 3 | 3 | **IMPROVE** → drop A(t) and the corrector; promote the deposited-hand likelihood gap to be the whole idea |
| **F3 — conserved-charge gate** | 2 | 4 | 2 | 2 | **DROP** (salvage only the void histogram + the transposition jump) |

**Judge overruled the critics on three points:** the decoy objection is *incorrect as stated* (matched-nuisance controls are exactly how a shared confound is subtracted; the real defect is that the decoys were unmatched); "A(t)=0 is a theorem" is *refuted by the brief's own source reading* (cross-distances are not reflection-invariant); and on circularity it **ruled for the advocate** — F2's discriminator trains on *experimental* complexes, which satisfies the critic's own escape condition, so the live risk is shortcut learning (embedding strain), not circularity.

**Survivors:** F2 (primary, narrowed) + F1-mutated (secondary, conditional, shares F2's fixture so it costs ~a week not a second project). F3 not kept — *"I am not keeping it to fill a slot."*

**Most promising next mutation:** replace every Vina-derived label in the population with the **crystallographically deposited configuration**, and make the label set itself the first deliverable — positives = deposited hand, negatives = the 2^k−1 re-embedded epimers under an identical embed-and-minimize protocol. Then three zero-training probes on one fixture: Vina + gnina discrimination; released checkpoints' likelihood gap; PB-dock pass rates on positives vs negatives. This converts the record's most damaging fact (Vina 13.81% vs 33.33%) from a threat into the motivating result and supplies the non-circular oracle all three finalists lacked.

**Plateau:** novelty has plateaued **on chirality**, but not on the goal. The binding constraint turned out to be *oracle availability*, not idea quality — every route to a method hit "and then what scores the hand?" with Vina disqualified by primary source and no wet lab. A fresh round seeded deliberately **away from** chirality is explicitly *not* covered by this plateau call.

**Honest bottom line:** after every concession, **no method contribution survived intact.** F1's corrector is prior art and breaks on multi-stereocenter ligands; F3's gate is self-defeating by its own advocate's admission; F2's method is a discriminator — a modest ML idea carried by a strong empirical audit. Plan one week, not one month. Kill condition first: **if off-the-shelf gnina already ranks the deposited hand above its epimers, stop at day five.**

---

# EMPIRICAL FITNESS PASS — lite Survivor 2 ("basin width, not depth")

Run on the lite run's second survivor, independently of the escalation. Artifacts: `.innovator/experiments/basin-width-docking/` (`results.txt`, `PROVENANCE.txt`, `raw_scores.csv.gz` = 2.87M rows).

**MEASURED: REFUTED.**

**Setup.** Real `smina` static binary (*"Smina Oct 15 2019. Based on AutoDock Vina 1.1.2"*, sha256 verified) — nothing hand-rolled. Public data only: generated molecules from Zenodo 10205723 (`targetdiff.zip`, `pocket2mol.zip`), pockets and references from the official TargetDiff CrossDocked2020 100-pocket test split. **151 real ligands vs 11,700 generated** across 100 pockets; heavy atoms well overlapped (crystal 22.3 mean, generated 21.9). **2,867,942 scorings**, ~35 min on 12 cores. Perturbation was rigid-body only, calibrated so E[RMSD] = σ *independent of molecule size* (verified: realised RMSD 0.0957/0.1909/0.2875/0.4763 Å for σ = 0.1/0.2/0.3/0.5, identical across 10-, 25- and 45-atom molecules). Two independent matched estimators (post-stratification + ANCOVA) agreed throughout; CIs are cluster bootstraps over pockets, 4000 reps.

**Pipeline validation** against TargetDiff's published Table 1 on the same pockets: measured TargetDiff score-only −5.39 vs published −5.47; Pocket2Mol minimized −6.47 vs published −6.42.

**Result (Analysis A, as-given poses, matched heavy-atom count).** Δ(σ) is flat: +2.402 at σ=0 → +2.368 at σ=0.3. **Shrinkage +1.4% [−0.5%, +4.5%]** against a pre-registered ≥20% threshold. The second SUPPORTED clause also fails — the **crystal group degrades *more*** (0.677 vs 0.646, ratio 0.954). Bootstrap SE on the shrinkage ≈1.3 points, so 20% sits **>13 SE away**: a refutation, not an underpowered null.

**Why the premise fails.** At matched heavy-atom count on released poses, generated molecules score **2.4 kcal/mol worse** than real ligands. *There is no advantage to collapse.* The idea assumed generated molecules enjoy an inflated docking advantage; in this regime they do not.

**What the data says the omitted term actually is: SIZE, not basin width.** −0.19 (crystal) to −0.22 (generated) kcal/mol per heavy atom. In the one regime where TargetDiff does look better (both groups minimized), its −0.25 kcal/mol advantage **reverses to +0.30 once size is matched** — so ≈0.55 kcal/mol of apparent advantage is size against ≈0.06 kcal/mol from width. **An order of magnitude apart.**

**The small effect that does exist.** Analysis B (both groups minimized first, removing the off-minimum confound) finds the predicted direction but tiny: generated lose +0.055 [+0.007, +0.097] kcal/mol more at σ=0.3 and +0.174 [+0.046, +0.284] at σ=0.5 — basins ~7–9% narrower. **Carried entirely by TargetDiff** (1.153 [1.062, 1.237]); **Pocket2Mol shows none** (0.933 [0.837, 1.023]).

**Honest caveats.** Only 151 real ligands (median 1.5/pocket) limits crystal-side precision. Receptors were scored as raw `_rec.pdb` rather than `prepare_receptor4` PDBQT, making references ~0.8 kcal/mol stronger than published — but the same receptor and protocol apply to both groups within each pocket, so the *contrast* is unaffected. Rigid-body smoothing only; a torsional Hessian could carry width the rigid modes miss. **Unexplored regime:** the advantage the idea was built on lives in the *redocked* (Vina Dock) setting, which this test did not run.

---

# PASS 2 — fresh lite run, seeded away from chirality

Triggered by the judge's plateau call: *novelty plateaued on chirality, not on the goal; the binding constraint was **oracle availability**, not idea quality.* Three changes to the brief versus pass 1:

1. **Hard exclusion** of chirality / stereochemistry / parity / reflection symmetry / E(3)-vs-SE(3) — and of the non-conservative-velocity-field line, which died on its own evidence.
2. **Oracle trustworthiness as a first-class output field.** Every mutator must name which oracle adjudicates its central claim and why that oracle is trustworthy *for that claim*. In pass 1 this question only surfaced during debate, after ideas were already built on sand.
3. **The empirical result fed back as a standing constraint:** *any claim adjudicated by an unmatched docking score is measuring size* (−0.19 to −0.22 kcal/mol per heavy atom; TargetDiff's −0.25 advantage reversing to +0.30 under matching).

Lenses were chosen for oracle availability rather than topic appeal: physics (xTB), compute (wall-clock/NFE), held-out distribution, experimental ground truth, theory (no oracle), data structure, distant-domain transfer, wild.

## L1 — the eight candidates

| # | Lane | Idea | Oracle | Why the oracle is trustworthy |
|---|---|---|---|---|
| **B1** | Physics | **Spectral curvature-gain audit.** g(λ) = model's implied curvature ÷ true curvature; a Boltzmann model has g≡1. Predicts g rises 1–2 orders with λ — "rigid caricature along bonds, mush along torsions". Fix: forward covariance kT·H⁻¹ along a metric-fidelity ladder. | Within-molecule GFN2-xTB strain ΔE | **Size-immune by construction** — same molecule both sides, no cross-molecule comparison anywhere |
| **C1** | Compute | **Per-molecule commitment time.** Required NFE varies >5×, predicted by lever-arm-weighted torsional flexibility not size; detectable free from endpoint drift; retire-and-compact the ragged batch. | Same-seed paired RMSD + NFE | **Dissolves the oracle problem** — identical molecules means every metric must agree |
| **A1** | Distribution | **Envelope-only conditioning.** Rank pockets by the model's own likelihood; molecule-only terms cancel *algebraically*. Δ(σ) spectrum localizes where pocket info lives; predicts it collapses below σ≈0.5 Å and that CFG structurally cannot fix it. | Pocket retrieval by own likelihood | Algebraic cancellation of all molecule-only terms |
| **C2** | Experimental | **B-factors are per-atom noise widths.** B=8π²⟨u²⟩ is a definition, so deposited B is already a Gaussian width in Å — the same axis as the flow schedule. Truncate per atom at t_i = 1−u_i/σ_max; the stopping time becomes a predicted diffraction observable. | COD / sub-1.0 Å geometry, deposited ADPs, RSCC | Experimental quantities, not predictions |
| **B2** | Theory | **Stiffness DELOCALIZES the denoiser.** ξ_t = Θ(σ_t√λ_max/α_t) *grows* with stiffness → ~50 bonds at σ=1 Å, exceeding drug-like diameters for ~80% of the schedule. Any receptive field R<ξ_t is information-theoretically insufficient (excess risk Θ(e^{−2R/ξ_t}), carried by soft modes). Corollary: **idealizing bond lengths makes fixed-cutoff architectures worse.** | Theorem + closed-form empirical denoiser | No oracle needed at all |
| **A2** | Data | **Rate-matched conditioning.** N_eff ≈10³ not 10⁵ (cross-docking inflates pairs, not environments); at ~1 sample/condition memorization *is* the MLE. Mechanism test is a scaling prediction: R* slides +1 bit per doubling of N_eff. | Molecule→pocket classifier | Label is ground truth by construction |
| **B3** | Numerics | **The sampling ODE is stiff in a closed-form-known subspace.** Contraction rate ≈1/(2ε): bonds |r|≈25, torsions |r|≤1 — a 25–50× spread inside one molecule. Explicit Euler at 20 steps sits *on* the stability boundary. Exponential integrator on the bond-stretch subspace; zero learned parameters. | xTB strain (paired) + NFE | Physics + compute, no scoring function |
| **A3** | Wild | **Condition-contrastive guidance.** p(P\|x) ∝ exp(−βs) is **exactly invariant** to any molecule-only g(x), gradient included — so size and every unnamed pocket-agnostic pathology cancel *without needing to know what they were*. Turns a broken oracle into a usable one by changing the question. | Chance baseline + provable invariance | Cannot manufacture MI about the pocket unless the scorer responds to the pocket |

### Structure of the population

- **Conditioning cluster (A1, A2, A3):** three unrelated lenses independently concluded the pocket signal may be near-absent, and each found a different chance-baselined, size-immune way to measure it — the model's own likelihood, a capacity/rate argument tied to dataset statistics, and an exact softmax invariance.
- **Stiffness cluster (B1, B2, B3):** three consequences of one physical fact. B1 says the **noise geometry** is wrong, B2 says the **architecture** is, B3 says the **integrator** is. **B2 and B3 independently explain why torsional diffusion works** (it deletes the stiff coordinates, collapsing B2's correlation length and removing B3's stiff subspace) — two unrelated derivations landing on the same explanation of a known empirical fact.
- **Standalone (C1, C2).**

**Known overlap risks flagged to the sweeps rather than left to be discovered:** B1's fix versus pass 1's anisotropic-prior finding (EigenFold, HarmonicFlow, ET-Flow, FLOWR.root, and **FlowMol's published ablation reporting structured priors underperform Gaussian**); A2 versus pass 1's conditioning-capacity idea (Delta Score, Gu et al., plus the unresolved MSCoD/IBEX scoop risk).

## L2 — three clustered sweeps (233 tool calls)

**All three sweeps hit WebSearch 200/200 before their first query, and all three rebuilt search from scratch** — curl against the arXiv Atom API, OpenAlex, Europe PMC full-text, Hugging Face semantic search, Crossref, plus full PDF downloads and grep. One ran a control query to confirm DuckDuckGo had started serving CAPTCHAs so no blank would be miscounted as a negative. Every negative below is graded strong or weak. This is the pass-1 lesson holding.

### Corrections to the brief I gave them

1. **FlowMol does NOT refute a positional prior.** I asserted twice that FlowMol's ablation shows structured priors underperform Gaussian on this task family. Sweep B downloaded the PDF and grepped it: the ablation is **categorical-only** (atom types, formal charges, bond orders on the simplex). *"Atom positions use a standard Gaussian throughout."* This also retroactively weakens **pass 1's kill of the resolution-time-mismatch idea**, which cited the same ablation.
2. **The pass-1 scoop risks are closed — both negative.** MSCoD's information bottleneck is a multi-scale feature-extraction module; IBEX's is a *PAC-Bayesian analysis tool* (*"IBEX retains the original TargetDiff architecture and hyperparameters"*). Neither bottlenecks the pocket channel. Corroborated by an OpenAlex exact-phrase search returning **exactly 2 works** for `"information bottleneck" AND "drug design"`.
3. **Pass 1 under-read its own best citation.** Delta Score (2403.12987) was fetched in pass 1 but only for its pocket-swap metric. Its §4.1 and Eq. 14 contain far more, and they damage three pass-2 ideas.

### Verdicts

| # | Idea | Verdict | Closest prior art | Genuinely unshipped |
|---|---|---|---|---|
| **B2** | Stiffness delocalizes the denoiser | PARTIALLY-DONE — **ranked #1 overall** | `2508.06614` Local Diffusion Models (time-dependent correlation length, local-denoiser error bound, buffer width ≳ ξ·ln(NK/ε), "global nets only near phase transitions"); Kamb & Ganguli (closed-form optimal score under locality+equivariance, r²≈0.95); banded-inverse decay is classical | **The sign of the effect.** Nobody derives ξ from a *stiffness* matrix; nobody claims stiffer data needs a larger receptive field. *"Idealizing bond lengths degrades a fixed-cutoff architecture — cleaning your data makes your model worse"* is **the single freshest claim across all eight ideas; nothing resembling it found** |
| **A1** | Envelope-only conditioning | PARTIALLY-DONE — #1 of its cluster | **MMG `2509.20609`: "MI corresponds to half the gap in MMSE between conditional and unconditional diffusion, integrated over all SNRs"** — literally Δ(σ). Its **Eq. 11 makes the MI integrand the squared norm of the CFG difference vector**, so the "CFG can't fix this" corollary is one line from a published identity. **Delta Score §4.1 states A1's thesis outright:** progress *"is primarily attributable to improvements in the unconditional components p_θ(x)"* | Running MMSE-gap machinery on a **pocket-conditioned 3D molecular** model and plotting Δ against a **physical length scale in Å** to locate where pocket information dies relative to the bond scale |
| **B1** | Curvature-gain audit | PARTIALLY-DONE — fix taken, diagnostic survives | **HI-FM `2410.11433`** integrates an energy Hessian into conditional flows (MNIST + Lennard-Jones only, no drug-like molecules, no xTB oracle); Whitened Score Diffusion (anisotropic Σ breaks DSM — live numerical hazard); GUD (per-mode SNR crossing); GenBench3D already splits stiff from soft modes | *"HI-FM owns the principle; nobody owns the audit."* The mode-resolved g(λ) on frozen checkpoints; the ladder to real force fields on drug-like molecules; the "total strain is a stiff-mode/hydrogen readout" critique |
| **C2** | B-factors as per-atom noise | PARTIALLY-DONE; one sub-claim contradicted | **Ambient Diffusion Omni** assigns per-sample minimum diffusion time and trains only above it; Soft Truncation. **Counterexample:** PDB ligand strain vs resolution has **r² = −0.0025, "no discernible correlation"** | STRONG negative across 12 queries: the unit identification B=8π²⟨u²⟩ ⟹ t_i with **zero fitting** (Ambient Omni needs a trained classifier), **per-atom** not per-sample, and the stopping time as a predicted diffraction observable |
| **C1** | Per-molecule commitment time | PARTIALLY-DONE — **cheapest decisive test** | **AdaptiveDiffusion** (NeurIPS'24) already has the zero-cost drift signal *and* the identical-output protocol; AdaDiff owns "steps should be sample-specific" | **Nobody retires samples** — all prior work walks every sample to t=1, skipping or caching; none removes one from the batch. Plus the lever-arm predictor and >5× spread (STRONG negative) |
| **A2** | Rate-matched conditioning | PARTIALLY-DONE | **Bajorath pair (PMID 41472830, 41504623)** already demonstrates the thesis for sequence-conditioned design: *"the transformer models did not learn target sequence information relevant for ligand binding"*; CrossDocked's redundancy is a **stated design property** and the authors ship clustered splits | The inverse-Simpson N_eff over pocket conditions, and the **scaling prediction (R\* +1 bit per doubling)**. **WEAK negative** — arXiv 503'd, OpenAlex budget gone |
| **B3** | Structured exponential integrator | PARTIALLY-DONE, closest to already-done | **gDDIM already exponentiates a full matrix operator**; **STORK** is already a stiff solver for diffusion/flow sampling, sold as *structure-independent* — the opposite pitch; operator-splitting samplers exist | Only the operator's *source* (the molecule's own Wilson B-matrix, post-hoc on an isotropically-trained checkpoint) — *"an engineering delta on published machinery, not a mechanism"* |
| **A3** | Condition-contrastive guidance | **ALREADY-DONE** | **Delta Score §4.1 IS the algebra** (Boltzmann posterior over a pocket bank), and its difference form **already cancels molecule-only g(x) exactly**. **Eq. 14 is an InfoNCE softmax over pockets** used as sampling-time guidance. **CASF-2016 reverse screening power** is a decade-old chance-baselined version of the task | Thin — a sampling-time variant for frozen external scorers |

### The killer objection to A3, and why it matters generally

Reverse docking has a documented dominant bias — "interference proteins" — correlating with **pocket contact area and hydrophobicity**, standardly corrected by z-scoring *per protein*. A3's softmax cancels the **molecule-only** nuisance g(x) and **leaves the pocket-only nuisance h(P) entirely intact** — and h(P) is the empirically dominant one in exactly this task. **A3 cancels the wrong nuisance.**

### Structural calls

- **Merge B2 and B3.** They are the off-diagonal and on-diagonal readouts of the *same* resolvent (α²I + σ²K)⁻¹ — B2's correlation length is its decay rate, B3's stiffness ratio its eigenvalue spread. *"Presenting them as independent contributions invites a referee to notice they are one derivation, which is worse than saying so yourself."* Use B1's audit as the opening empirical section; discard B1's Hessian-prior fix (HI-FM has it).
- **A1 and A3 are ~80% the same experiment** (K-pocket bank, top-1 retrieval, chance baseline, volume-matched decoys). *"Do not treat them as two bets."* A2 is the genuinely distinct one — the only *intervention* with a quantitative prediction, and the only claim about the dataset rather than the model.
- **Torsional diffusion is the most dangerous citation in the set**, precisely because B2 and B3 both explain why it works: *"a referee reads that as — the field already solved this in 2022 by removing stiff DOFs, and your theory is a post-hoc rationalization of folk knowledge."*

### The wedge nobody else has

Delta Score damages all three conditioning ideas — **but contains no discussion of molecule size, heavy-atom count, or ligand efficiency as a confound.** That gap is real, and this project has already measured it: 2,867,942 smina scorings showing apparent advantage is dominated by size, with TargetDiff's −0.25 kcal/mol reversing to +0.30 under matching. **That is an owned asset no competing group has.**

### Open items requiring a run with WebSearch available
- **GeomFlow**, "Geometry-aware adaptive diffusion model via Hessian information" (Neurocomputing 2026) — Elsevier returned a redirect stub. Alarming for B1.
- WEAK negatives to re-check: "nobody sets a conditioning bottleneck's capacity from a measured dataset statistic" (A2's mechanism); "nobody baselines a policy gradient across conditions" (A3); B-factor loss weighting in ML code (GitHub code search was unavailable).
