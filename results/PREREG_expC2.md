# Pre-registration — Experiment C2: interface-pinned partial-diffusion dose-response

Written 2026-08-10 on Sherlock, **after** reading BRIEF.md / CLAUDE.md / results/FINDINGS.md /
results/FINDINGS_expA.md / results/FINDINGS_expC.md / results/PREREG_expC.md, but **before generating a
single C2 RFdiffusion backbone**. C2 **inherits PREREG_expC.md wholesale** except the two changes marked
`[C2]` below (the fix, and the ladder) plus the two new controls (the leakage split KILL C2b and the
continuous-slope primary statistic). No reading is moved after seeing a number.

## 0. The question, and why C2 exists

Experiment C tested the burial-matched hotspot log-prob deficit on RFdiffusion partial-diffusion
backbones as a dose–response vs interface-RMSD. Result (FINDINGS_expC.md): the sequence-free **KL
detector transferred** (C-KL fired), but the **burial-matched log-prob gap (C-PRIMARY) was only
suggestive** — significant only at non-physical drift (`partial_T=40`, iRMSD≈18 Å), non-monotone, and
not surviving the iRMSD<3 Å confound. The cause was mechanical, not scientific: diffusing the binder
against a held target **without hotspot conditioning** used the `Complex_base` checkpoint outside its
training regime, so the binder floated off and **62–75 % of designs diverged** (binder coords 10³–10⁷ Å).
The physical regime that decides the question — **interface FORMED, iRMSD 2–8 Å** — was never densely
populated (only 12–13 interface-formed complexes per intermediate level).

**C2 fixes exactly that**: pin the binder to the interface so it stays physical while the backbone
drifts, then sample the deciding regime densely. Everything else (generator, label-preserving trick,
55-complex set, scoring, bootstrap) is inherited from PREREG_expC verbatim.

## 1. [C2] The fix — keep the interface formed

Two independent ways to stop the diffused binder floating off. **Option A is PRIMARY; Option B is the
pre-declared fallback**, triggered by an objective rule fixed here before any number:

- **(A) PRIMARY — `ppi.hotspot_res` conditioning.** Pass RFdiffusion the **TARGET-chain residues that the
  labelled binder hotspots contact in the crystal**: a target residue is a hotspot-contact if its virtual
  Cβ is within **10.0 Å** of the Cβ of any *conditioned-arm* labelled binder hotspot (arm defined in §3).
  *[Executed at 10.0 Å; originally pre-registered at 5.0 Å — see the §8 pre-data correction.]*
  These are supplied via `ppi.hotspot_res=[<chain><resnum>,...]` in RFdiffusion **input-PDB chain space**
  (the held target's output letter + its 1..Lt renumbering, mapped from crystal via
  `results/expC_outmap.json`), telling the `Complex_base` checkpoint to keep the binder docked to that
  patch during denoising.
  - **Validity note (why recovery is still measurable):** `hotspot_res` are **TARGET residues** — a
    geometry / targeting constraint on *where the interface sits* — **not** the binder's own residue
    identities and **not** the binder hotspot positions whose recovery we score. That is what keeps the
    log-prob recovery measurement valid; it is checked empirically by **KILL C2b** (§5), which is the
    reason the arm split exists.
- **(B) FALLBACK — the `Base` checkpoint** (`Base_ckpt.pt`) instead of `Complex_base`, same contig, **no**
  `hotspot_res`. Used only if Option A **still diverges on >40 % of designs** (interface-formed fraction
  <60 %) assessed on the actual C2 run pooled over the physical levels {5,10,15,20}. Because Option B
  passes no conditioning, KILL C2b is moot under it (nothing was conditioned → no leakage possible), and
  that is reported as such.

**Interface-formed fraction is reported per noise level** (same QC as Exp C: ≥5 inter-chain Cβ contacts
within 10 Å at labelled hotspots). If the fix works it should be **>> Exp C's 25–38 %**; the pre-registered
success target is **≥30 interface-formed complexes per physical level** (Exp C had 12–13).

## 2. [C2] The ladder — sample the physical regime densely

- **Noise ladder (FIXED):** `partial_T ∈ {0, 5, 10, 15, 20, 30}` of 50 (drops Exp C's non-physical
  `40`, adds `15` and `30`). **N ≥ 6** backbones per (complex, noised level); seeds fixed and recorded
  (`inference.seed` per complex, designs 0..N−1).
- **`partial_T=0` = the crystal backbone** (0 noising = identity), taken directly from the prepared input
  PDB and pushed through the identical convert+score path — the mandatory positive control (KILL C2a).
  T0 is deterministic, so **N=1** fully characterises it (6 identical copies would be pure waste); this
  single declared deviation from "N≥6 per condition" is stated here up front.
- **Same 55 complexes** as Exp C (`results/expC_complexes.csv`, L≤400), same binder/target assignment,
  same prepared inputs (`$SCRATCH/expC/inputs/`), same contigs (`results/expC_inputs.csv`).
- **Dose variable = interface Cα-RMSD to crystal** (superpose on the held target, RMSD over labelled
  binder interface Cα), identical to Exp C §2.
- Backbones → `$SCRATCH/expC2/backbones/<cid>_T<T>_<s>.pdb`. Total design: 55 × (5×6 + 1) = **1 705**
  backbones. Resumable; RFdiffusion `nan`-coordinate divergences and interface-dissolved backbones are
  **listed and excluded, never silently dropped** (CLAUDE.md rule 6).

## 3. [C2] The leakage split (pre-registered; drives both hotspot_res and KILL C2b)

Fixed here, before any backbone, seed **20260803**:

- **Labelled binder hotspots** per complex = distinct (chain,resnum) on the binder that appear as a `hot`
  position in any committed `results/p0_dssp_pairs_*.csv` (the same union `src/expC_setup.py` used to
  assign binder vs target). Written to `results/expC2_hotspot_split.csv`.
- **50/50 split per complex:** deterministically shuffle each complex's binder hotspots with a
  complex-seeded RNG (`seed = 20260803 + stable_hash(cid)`, recorded) and assign the first ⌈n/2⌉ to the
  **conditioned** arm, the rest to **held-out**. Odd counts give the extra to *conditioned* so
  single-hotspot complexes are still pinned. The split is a property of positions, fixed once.
- **hotspot_res uses ONLY the conditioned arm's contacts** (§1A). The held-out arm's hotspots are never
  passed to RFdiffusion. Both arms' positions are scored identically afterwards.
- Trade-off, declared: passing only the conditioned half (not all hotspots) slightly weakens pinning
  versus passing all, but is required to create a held-out set for KILL C2b. Target-contact patches of
  different hotspots overlap heavily (one interface), so the conditioned half is expected to pin the
  binder adequately; if not, the >40 % rule routes to Option B.

## 4. Scoring — identical to Exp C (reuse `src/expC_score.py`, `src/expC_analyze.py`)

- Re-key each backbone to crystal via `results/expC_outmap.json`; ProteinMPNN `v_48_020.pt`,
  `augment_eps=0`, 8-order teacher-forced native log-prob + unconditional KL pass; burial = Cβ neighbour
  count. Gap tier = **SECONDARY_B within-binder** (the power tier), plus the **EXPC_within_binder
  re-match** robustness variant. Complex-level bootstrap, **10 000** gap / **2 000** KL reps, seed
  **20260803**. Interface QC and iRMSD exactly as Exp C.
- **[C2] KL-bootstrap hardening:** `paired_dauroc` computes `p_gt0` **nan-aware** (mean over finite
  bootstrap replicates only) and reports the **fraction of degenerate resamples dropped** per level
  (`frac_degen`). This is a strictly-more-correct hardening applied to `src/expC_analyze.py`; re-running
  Exp C through it must leave the committed Exp C KL numbers unchanged (verified before use).
- **[C2] Continuous-slope primary statistic** (`src/expC_slope_check.py`, new): the per-backbone gaps
  `d` written by `expC_analyze.py` (`*_gap_perbackbone.csv`) are regressed on `log10(interface-RMSD)`.
  Two slopes are computed and sharply distinguished:
  - **physical-generated slope** (the honest C2-PRIMARY statistic): over backbones with
    `interface_ok==1` **AND** `iRMSD ≤ 8 Å` **AND** `partial_T ≠ 0`. Slope, 95 % complex-level bootstrap
    CI, and two-sided bootstrap P. Units: **gap per log10 Å**.
  - **naive all-backbone slope** (the artifact NOT to be reported as evidence): over all backbones incl.
    the T0 crystal anchor and the dissolved >10 Å tail. Reported only to expose the crystal-vs-dissolved
    confound it manufactures.
  - **Positive control (gate):** the same script is first run on Exp C's committed
    `results/expC_gap_perbackbone.csv`; Exp C reported the physical-generated slope **flat** and the
    naive slope apparently significant. The C2 slope machinery is trusted only after it reproduces that
    qualitative Exp C split (flat physical, steep naive); the reproduced values are written to
    `results/expC_slope_check.csv` and reported verbatim (whatever they are).

## 5. Pre-registered readings (fixed before running; **both primaries are publishable**)

- **C2-PRIMARY (positive).** The **physical-generated slope** (§4, interface-formed, iRMSD≤8 Å, excluding
  T0) is **negative with 95 % complex-bootstrap CI excluding zero** → the tax scales with backbone drift
  *within the physical generative-design regime*; the conditioning-set claim is complete. The binned gap
  and the iRMSD 3–8 Å band CI are reported alongside, but **the pre-registered verdict statistic is this
  continuous physical-generated slope** (the binned all-backbone or crystal-anchored slope is not evidence
  for C2).
- **C2-NULL (equally reportable; a-priori the more likely outcome given Exp C).** If that
  physical-generated slope is **TOST-equivalent to zero** — 90 % CI entirely inside **±0.10 gap-per-log10
  Å** (bounded below the Exp A predicted-backbone effect) — the log-prob tax is **prediction-scale only**
  and does not appear in the physical generative-drift regime → an honest bound; KL remains the
  transferable signal. This is a pre-registered negative that sharpens the paper's scope, **not** a failed
  experiment; power is not chased past the pre-declared N to move it.
- **C2-KL.** KL ΔAUROC-over-burial CI **excludes zero at each physical level** {5,10,15,20} (replicating
  Exp C's interface-formed C-KL).
- **KILL C2a (mandatory control).** `partial_T=0` reproduces the ~zero crystal within-binder deficit
  (Exp C T0: SECONDARY_B within-binder +0.303 [−0.19,+0.80], CI ∋ 0). If not, the pipeline is broken —
  **stop and debug**.
- **KILL C2b (leakage control, NEW).** The burial-matched gap at hotspot positions **whose target
  contacts WERE passed** to `ppi.hotspot_res` (conditioned arm) must be **statistically indistinguishable**
  from the gap at the **held-out** arm (paired per complex, complex-bootstrap CI of the arm difference
  **contains zero**). If conditioning leaked interface information into recovery, conditioned positions
  would show a *smaller* deficit than held-out; if the arms **differ**, C2-PRIMARY is confounded by the
  conditioning and is reported as such. (Under Option B fallback: no conditioning → reported moot.)
- **CONFOUND (as Exp C).** The gap restricted to well-formed-interface backbones (contacts preserved AND
  iRMSD<3 Å) is reported; the signal must **not** live only in the marginal-interface tail.

## 6. SECONDARY — binding-relevant readout (CPU, free)

Recompute `partial-ρ(experimental SKEMPI ΔΔG_bind, ProteinMPNN log-odds ℓ(mut)−ℓ(wt) | burial)` per
level on the C2 backbones (`src/expC_secondary.py`). Prediction unchanged (Exp C: collapses −0.236→≈−0.06
off the native manifold). New question on the denser physical ladder: does it collapse **gradually** with
iRMSD or (as in Exp C) is it already gone by `partial_T=5`? Exploratory; no falsifier attaches.

## 7. Optional (only if C2-PRIMARY lands clean; ~15–25 GPU-h)

AF2-multimer ipTM on a ~20-complex subset: ProteinMPNN-design K=8 on the crystal vs the highest physical
C2 backbone, fold, compare interface ipTM/pAE at hotspots. Not run unless C2-PRIMARY is clean.

## 8. Deviations / operational items, declared up front

- **[Pre-data correction, 2026-08-10, PI-approved]** The `ppi.hotspot_res` target-contact cutoff was
  pre-registered at **5.0 Å Cβ** but is executed at **10.0 Å Cβ**. A positive-control geometry diagnostic
  (`src/expC2_contact_diag.py` → `results/expC2_contact_diag.csv`) confirmed the geometry is correct
  (median hotspot→target Cβ contact **5.44 Å**, IQR 4.57–6.28, i.e. interface-scale — not a chain/coord
  bug) and that 5.0 Å is simply too strict: it left `hotspot_res` empty for **35/55** complexes (only
  **20/36** hotspot-bearing complexes pinnable, median 1 token), so Option A could not be tested at all.
  **10.0 Å is the project's own interface-formed contact definition** (`src/expC_score.py`, inter-chain
  Cβ<10 Å) — an internally-consistent value, not one tuned to an outcome — and it pins **35/36**
  hotspot-bearing complexes (median 11 tokens). The cutoff is a **generation-time pinning** parameter that
  does **not** enter the gap/slope/KL/leakage measurements, and **no C2 outcome had been computed** when it
  was changed. Both cutoffs' `hotspot_res` counts are recorded; the 50/50 arm split is cutoff-independent
  and unchanged. `src/expC2_hotspot_res.py` takes `--cutoff` (run with `--cutoff 10.0`).
- `src/expC_slope_check.py` and `results/expC_slope_check.csv` **did not exist in Exp C**; the slope
  statistic is *defined and implemented here* and validated by reproducing Exp C's committed
  `expC_gap_perbackbone.csv` split as the §4 positive control. Whatever the reproduced Exp C slope values
  are, they are reported verbatim — the pre-registered C2 verdict does not depend on their matching any
  previously-quoted figure.
- `hotspot_res` is specified in RFdiffusion input-PDB chain space (held target's output letter + 1..Lt),
  derived from `results/expC_outmap.json` (registration-preserving, positional).
- T0 uses N=1 (deterministic crystal identity); all noised levels use N≥6.
- All seeds fixed (RFdiffusion `inference.seed` per complex; bootstrap 20260803; split 20260803).
  Bootstrap replicate counts reported beside every CI; every excluded/failed backbone listed with reason.
- Raw outputs → `results/expC2_*.csv` with the exact command in a `command` column; verdict written to
  `results/FINDINGS_expC2.md` strictly from the readings above.
