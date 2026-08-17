# FINDINGS — catalytic dissociation: REAL (survives composition + truncation); only the *frustration mechanism* is dead

**CORRECTED 2026-08-14 after an independent Fable-5 audit (src/catalytic_audit.py) + my own re-verification.**
My first write-up called this a composition null. **That was a methodological error** (see §Correction). The
dissociation is real.

**Scripts:** `src/catalytic_dissociation.py` (scoring), `src/catalytic_audit.py` (corrected analysis).
**Outputs:** `results/catalytic_positions.csv`, `results/catalytic_audit_{mpnn_entropy,burial,nchains}.csv`.
M-CSA (114 enzymes, 40,951 positions, 391 catalytic = role type 'reactant'). ProteinMPNN vs ESM-2 (150M);
enzyme-clustered bootstrap (2000), seed 20260803.

## Question
Does "confidence is not competence" generalise beyond binding hotspots — is structure-conditioned IF
confidence blind to CATALYTIC residues even though a sequence PLM (ESM-2) finds them — after controlling for
the two confounds this project knows to check (amino-acid COMPOSITION and BURIAL)?

## Result (the RIGHT analysis: within-amino-acid-type stratified AUROC)
Composition is removed exactly by stratifying within each amino-acid type (not by ΔAUROC over a one-hot
baseline, which is low-power — see §Correction).

| within amino-acid type | AUROC (is_catalytic) |
|---|---|
| MPNN log p(native) | 0.482 [0.436, 0.531] — **chance / BLIND** |
| MPNN negentropy | 0.432 [0.386, 0.481] |
| **ESM-2 negentropy** | **0.771 [0.723, 0.822] — PREDICTS** |
| **DISSOCIATION (ESM − MPNN)** | **+0.288 [+0.235, +0.336], P=1.000 — SURVIVES composition** |

**Truncation control (monomers only — no partner chains deleted):** ESM-2 0.764 [0.676, 0.844] predicts;
MPNN negentropy 0.516 [0.433, 0.601] = **chance** (the raw anti-prediction was a chain-truncation artifact,
not frustration). My independent check: monomers ESM 0.740, MPNN 0.505, dissociation +0.235.
**Strictest test (monomers AND within (aa, burial)):** ESM 0.648 [0.535, 0.757] predicts; MPNN 0.470/0.430 =
chance; **dissociation +0.176 [+0.060, +0.293], P=0.999 — still holds** (committed catalytic_audit.csv,
monomers_only×aa_burial). (CORRECTED 2026-08-17: burial partly *contributes* to the raw gap — catalytic
residues are more buried and burial raises MPNN's apparent determinacy — so controlling for it SHRINKS the
dissociation, +0.288→+0.234(all)→+0.176; it SURVIVES but is smaller. This is the OPPOSITE of the binding case,
where burial MASKS the effect — do not conflate the two.)

## Reading
**The dissociation is real and robust.** Structure-conditioned inverse-folding confidence is **blind** to
catalytic residues (within-type AUROC ≈ 0.50, chance) while a sequence PLM's conservation **predicts** them
(0.77), a gap of +0.17 to +0.29 that survives amino-acid composition, burial, and chain-truncation. This
**generalises "confidence is not competence" from binding hotspots to catalytic sites**: IF confidence is
blind to functional importance across function types, and what predicts function is either free geometry (for
binding, §3–4) or sequence conservation (for catalysis). ESM is the built-in negative control for structural
artifacts — it never sees structure and is invariant to truncation (0.764 vs 0.774), exactly as it must be.

**What is dead: the frustration MECHANISM.** The raw MPNN *anti*-prediction (0.398) is composition +
single-chain truncation, NOT frustration. State the mechanism as tested-and-negative; the finding is the
dissociation (blindness), not a frustration story.

## Correction (why my first pass was wrong — recorded deliberately)
1. **Wrong quantity.** I scored MPNN by log p(native), which is confounded with amino-acid identity *by
   construction* (p(His|backbone) is low wherever His appears), so it can NEVER survive an identity control
   regardless of truth. The determinacy/frustration question is about ENTROPY (negentropy), independent of the
   native token; I never computed MPNN negentropy for the control.
2. **Wrong readout.** ΔAUROC over an 0.853 aa-identity baseline is compressive and low-power: a synthetic
   positive control shows the detection floor is a within-type AUROC of ~0.55–0.57, so "ΔAUROC≈0 → vanishes"
   cannot distinguish no-effect from a moderate effect. ESM's "+0.032 small" corresponds to a within-type
   AUROC of 0.77 — not small. **The ΔAUROC-over-one-hot estimator is retired; report within-type AUROC.**
3. My earlier line "~10× smaller" compared a ΔAUROC to an AUROC difference — apples to oranges; withdrawn.

## Caveats
- ESM-2 runs unmasked (sees the native token); neutralised for the within-type comparison (token constant in
  a stratum). Masked-marginal ESM-2 would clean the *raw* magnitudes; optional.
- The chain-truncation artifact is specifically in the MPNN *entropy* measure (monomer−multimer gap +0.119
  [+0.023, +0.221], significant), not logp(native) (+0.046, n.s.); on monomers both MPNN measures are at
  chance. ESM-2 is invariant to truncation (0.764 vs 0.774), as it must be (never sees structure).
- The fully-crossed strictest cell (monomers × complex × amino-acid × burial) is underpowered (~39 catalytic
  residues) and n.s. (+0.143 [−0.134, +0.426]); the powered strict cell (monomers × aa × burial) holds
  (+0.174 [+0.062, +0.288], P=0.999). Report the powered cell; do not over-read the fully-crossed one.
- Contrast with the binding nugget: that survives its geometry control cleanly (CPI 0.000). Both now stand.
- Provenance: `src/mcsa_build_labels.py` reproduces `mcsa_labels.csv`; DATA.md has the M-CSA entry.
