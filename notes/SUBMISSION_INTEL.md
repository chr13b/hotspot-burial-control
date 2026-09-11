# Submission intel — ICLR 2027 policy, workshops, OpenReview recon

*Living note for maximizing acceptance odds. Verified items cite the URL + date fetched (CLAUDE.md rule 5);
"TO-VERIFY" = not yet confirmed against the primary source. Updated 2026-09-11.*

## ICLR 2027 — dates & policy
- **Deadlines (project-known; verify on the CFP):** abstract **18 Sep 2026**, full paper **25 Sep 2026** (AoE).
  Today 2026-09-11 → ~14 days to the paper deadline.
- **Dual-submission / concurrent workshops — VERIFIED (blog.iclr.cc/2026/09/02/submission-policies-for-iclr-2027,
  fetched 2026-09-11):** papers presented at **non-archival** workshops (venues with **no proceedings**) do **not**
  violate the dual-submission policy; posting to **arXiv during review is allowed**. → We may submit to ICLR 2027
  **and** a non-archival workshop concurrently, as long as it is not also under review at another *peer-reviewed,
  with-proceedings* venue. MLSB / GEM / LMRL are all non-archival — safe.
- **New ICLR 2027 rules (TO-VERIFY against the CFP; from a secondary source, aiweekly.co):** author cap ~20
  papers/author; a **mandatory AI-use statement**. We already have a "Reproducibility and LLM-usage disclosure"
  section — **confirm it matches the mandated format** when the CFP template lands. See [[submission-hygiene]].

## Workshops — top fits (fit-ranked, deadlines TO-VERIFY on each site)
1. **MLSB — Machine Learning in Structural Biology** (NeurIPS 2026, Dec; 5th edition). *The* fit: protein
   structure/design, inverse folding, ΔΔG. Non-archival. 2025 deadline was Oct 1; 2026 ~early Oct (TBD).
   mlsb.io/call-for-papers.
2. **GEM — Integrating Generative & Experimental Platforms for Biomolecular Design** (NeurIPS 2026, Dec 13,
   Atlanta). Deadline **Sep 30 2026** (AoE). Dry-lab + wet-lab tracks — our steering/design angle fits the dry-lab
   track squarely. gembio.ai.
3. **LMRL — Learning Meaningful Representations of Life** (NeurIPS 2026). Our leverage-as-representation /
   "what does the model represent about binding" framing fits; less design-specific than MLSB/GEM.
4. **AI for Drug Discovery** (NeurIPS 2026). Moderate fit (binding affinity / ΔΔG relevance).
5. **Simbiochem — ML for Simulations in Biology & Chemistry** (NeurIPS 2026). Moderate fit via the FoldX/physics
   cross-modality angle.
- **Recommendation:** MLSB (primary) and/or GEM (Sep 30, earliest) as a *non-archival* companion to the ICLR
  submission — free visibility, no dual-submission conflict, and an early expert-reviewer signal. A 4-page workshop
  version can reuse Figs. 0/P/L.

## OpenReview recon — what to learn from close papers (TO FILL during the recon)
*Target: 3–5 recent ICLR/NeurIPS papers nearest our topic (inverse-folding conditioning, binding/ΔΔG from
sequence models, training-free steering, protein design diagnostics — e.g. BA-Cycle, RedNet, StaB-ddG,
target-conditioned IF, ProteinMPNN-adjacent). For each, read the OpenReview thread and record:*
- accept/reject + scores; the reviewers' **top praise** and **top complaint**;
- recurring reviewer asks in this subfield (baselines? wet-lab? generalization? novelty vs prior score?);
- what rebuttals worked; what sank borderline papers.
*Then translate into concrete edits here.* Hypotheses to test against the threads (from our own weaknesses):
in-silico-only validation, single primary fixture, "is this just BA-Cycle", modest effect sizes — do reviewers in
this area accept honest-null + triangulation, or demand wet-lab? Calibrate scope accordingly.

## Pre-submission hygiene checklist (see [[submission-hygiene]], [[data-archival]])
- De-identify: no author/AI names, no `chris`/home paths, no session URLs in the submission repo or PDF.
- Every number → committed CSV; INDEX.md mapping claims→CSVs; verify-refs + .bib; anonymized mirror; Zenodo
  (before ~2026-10-09, per [[data-archival]]).
- Ask the operator for their previous **audit-prompt** template before running the final numbers/de-id audit.
