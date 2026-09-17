# SUBMISSION RECON — persistent notes for a successful ICLR 2027 submission

*Living file. Verified facts carry a source URL + date checked; unverified items are marked. Updated 2026-09-11.*

## ICLR 2027 — key dates & the dual-submission / workshop policy
- Deadlines (from CLAUDE.md project state): abstract **18 Sept 2026**, full paper **25 Sept 2026 (AoE)**. Today
  2026-09-11 → ~14 days to the paper deadline. Double-blind review.
- **Dual-submission policy (verified 2026-09-11):** a paper on arXiv, or **presented at a workshop with NO formal
  proceedings (non-archival), does NOT violate** ICLR's dual-submission policy. So we MAY put this work in a
  non-archival workshop concurrently with the ICLR 2027 main submission. Rejected ICLR papers become non-archival
  and may be submitted elsewhere. *(sources: iclr.cc/Conferences/2027/CallForPapers, /AuthorGuidelines, /2025/FAQ)*
- **New ICLR 2027 rules to honor (TO-VERIFY on the final CfP; secondary source aiweekly.co):** a **mandatory
  AI-use statement** and an author cap (~20 papers/author). We already have a "Reproducibility and LLM-usage
  disclosure" section — confirm it matches the mandated format when the template lands. See [[submission-hygiene]].
- **Caveats to respect:** (1) the workshop must be **non-archival** (MLSB, GEM, ICBINB all are/have been). (2)
  Double-blind: a non-anonymous workshop version can de-anonymize; ICLR tolerates public arXiv/workshop versions,
  but do NOT cite our own workshop paper non-anonymously in the ICLR submission. (3) Re-confirm on the final CfP
  before acting.

## Workshop options — ranked by FIT (timing secondary, per the user)
NeurIPS 2026 workshops announced 2026-08-10 (Sydney/Paris/Atlanta hubs); **suggested** submission date **Aug 29
2026 has already passed**, individual deadlines vary — many NeurIPS 2026 workshop deadlines are likely closed, so
the practical routes are ICLR 2027 workshops (deadlines ~Feb 2027, natural soft-landing) or a late-deadline NeurIPS
one. Verify each deadline before committing.
1. **GEM — Integrating Generative and Experimental Platforms for Biomolecular Design** (NeurIPS 2026, Dec 13,
   Atlanta). **Best fit:** generative (inverse-folding) models + experimental/physics validation for binder design
   — exactly our steering + ΔΔG/FoldX story; **non-archival** (OpenReview, opt-out), **dry-lab track, ≤5 pages**
   (we fit dry-lab). **DEADLINE AMBIGUOUS — likely already closed:** gembio.ai (fetched 2026-09-11) shows a
   **final deadline of 30 August 2026 AoE** with crossed-out early-September dates — NOT the "Sep 30" an earlier
   pass claimed. Treat GEM-2026 as probably closed; **verify directly** before relying on it. *(gembio.ai; neurips.cc 2026 workshops blog)*
2. **MLSB — Machine Learning for Structural Biology.** The canonical home for ProteinMPNN/ESM-IF-class work; 2025
   was NeurIPS-co-located, non-archival, deadline ~26 Sept/1 Oct 2025. **2026 edition NOT yet confirmed on
   mlsb.io** (site still shows 2025) — verify; historically runs yearly. *(mlsb.io)*
3. **AI for Drug Discovery: Bridging the Translation Gap** (NeurIPS 2026, Sydney). Good fit — binding/affinity/ΔΔG,
   design-to-application; our "does the model know binding + a training-free knob" lands well.
4. **ICBINB — Failure Modes of AI in Biology** (NeurIPS 2026, Sydney). Strong *thematic* fit for the diagnostic
   half: "confidence is not competence," inverse folding is blind to binding, the field reads the wrong quantity —
   a distinctive surprising-failure story that would stand out here.
5. **Simbiochem — ML for Simulations in Biology and Chemistry** (NeurIPS 2026, Sydney). Decent — the physics/FoldX
   energy-function angle; more simulation-centric than our ML-diagnostic core. (Runner-up: ML4Molecules, weaker.)
- Also: **LMRL — Learning Meaningful Representations of Life** (NeurIPS 2026) — fits the representation-learning
  framing ("what does the model represent about binding") but is less design-specific than MLSB/GEM.
- **Net on NeurIPS-2026 timing:** GEM's deadline is ambiguous/likely-closed (30 Aug 2026 final per gembio.ai), and
  the suggested NeurIPS-2026 date (29 Aug) has passed — so a NeurIPS-2026 non-archival companion is probably NOT
  actionable this cycle. The clean practical route is an **ICLR 2027 workshop** (~Feb 2027 deadline, natural
  soft-landing) or **MLSB 2026** if it runs (date TBD, verify mlsb.io). Verify any deadline before committing.

Recommendation: prioritize the **ICLR 2027 main** submission (25 Sept). A non-archival workshop is a low-risk
bonus (feedback + visibility) and policy-clean; if we do one, **GEM** or **MLSB** are the natural homes, **ICBINB**
the distinctive angle. Realistically an **ICLR 2027 workshop** (Feb 2027 deadline) is the clean target and a soft
landing if the main is rejected.

## DE-IDENTIFICATION — findings that MUST be fixed before the anon submission repo ships
- **The name/username leaks in ~65 tracked files** (`git grep -il bertsch` = 65), notably `environment/*.sbatch`
  (Sherlock `/scratch/users/cbertsch/...` paths). These MUST be scrubbed or excluded from the anonymized mirror.
- **Git history itself** carries `Christian Bertsch <chris.bertsch01@gmail.com>` on every commit → the anonymized
  submission repo must be a **fresh `git init`** (no history), not a clone.
- **269 committed `results/*.csv`**; most are intermediate/orphan (not cited by name in the paper). The submission
  repo needs an `INDEX.md` mapping every paper number → its source CSV, and a decision on which CSVs ship.
- Session/attribution strings (Claude-Session URL, Co-Authored-By) are in commit messages → fresh init also
  removes these from the shipped history.

## Submission-repo checklist (packaging phase)  → run `notes/AUDIT_PROMPT.md` for the full audit
- [ ] Fresh `git init` (no identifying history) for the anon mirror. **66 tracked files carry `bertsch`/`cbertsch`
      + git-history author = real name/email → history CANNOT ship; fresh init only.**
- [ ] Scrub `bertsch`/`cbertsch`/email/scratch paths from all files (env scripts, latex, any hardcoded paths).
- [ ] **Load-bearing-ONLY (explicit requirement): ship only files actually used for the paper.** Walk every
      `results/*.csv` (269) + `src/*.py`; keep only what an `INDEX.md` row or a reproduce step needs; exclude
      pilot/scratch (`_*.csv`, `*_audit2`), superseded shards where a consolidated file exists, and orphan
      intermediates. No unused overhead in the anon repo.
- [ ] **Directory structure** for the anon repo: `paper/` (draft + built PDF), `src/`, `results/` (or `data/`,
      load-bearing CSVs only), `figures/`, `README.md`, `INDEX.md`, `reproduce.sh`.
- [ ] `INDEX.md`: paper claim → CSV → script that produced it (reproducibility map). Seed from `FOLDED_STATUS.md`.
- [ ] Strong `README.md`: what/why, install, one command per figure/table, data provenance, license.
- [ ] Numbers-vs-CSV full audit (every bolded number in the paper traces to a committed CSV) — `AUDIT_PROMPT.md` Ph.1.
- [ ] verify-references on the .bib; ensure no self-identifying citations.
- [ ] **Delete the working marker `⟨✎ external citations DOI-verified …⟩` in §8 (PAPER_DRAFT.md ~line 1038)** — it
      is a to-self note, must NOT appear in the built PDF.
- [ ] Remaining readability (audit-flagged): §4 subsection headers (done), dose-law σ-ladders → table, Tier-2/3
      sentence splits.
- [ ] Zenodo archive before ~Oct 9 (data-archival memory deadline).

## OpenReview recon — FINDINGS (deep pass, 2026-09-17)
**Provenance + caveat (citation discipline).** OpenReview's API + forum pages sat behind a live domain-wide
bot-check on 2026-09-17 (confirmed via a positive control on the ViT/ICLR-2021 forum — identical block), so review
TEXT below was recovered from third-party Hugging Face dataset mirrors that crawl OpenReview
(`smallari/openreview-iclr-peer-reviews`, `insomnia7/iclr2026_stats`), each record cross-verified against the
forum id (WebSearch) + exact title + reviewer-id format before trusting. Accept/venue for each paper was confirmed
FIRST-PARTY via iclr.cc/icml.cc virtual pages. **Gap:** the rebuttal exchange (author responses, mid-review score
changes) lives only on the blocked notes API — "which rebuttal moves moved scores" is unrecoverable this pass.
Treat review quotes as mirror-sourced (not first-party-OpenReview) and NEVER quote them in the paper.

**Threads extracted (ratings):**
- **BA-Cycle** (Jiao et al.) — ICLR 2025 Accept/Poster (iclr.cc/virtual/2025/poster/28490); ratings [6,10,6,8].
- **Complexa / Proteína-Complexa** (Didi et al., NVIDIA/Oxford) — ICLR 2026 **Oral + Poster**
  (iclr.cc/virtual/2026/oral/10007212); ratings [10,8,4,6]. **Most relevant to us: in-silico-only, oracle-based.**
- **PRISM** (retrieval-augmented IF) — ICLR 2026; ratings [6,4,8,6].
- **MoMPNN** (multi-objective preference-aligned IF) — ICLR 2026; ratings [8,6,4].
- **StaB-ddG** — ICML 2025 Poster (icml.cc/virtual/2025/poster/45926); review text NOT recovered (no mirror) — honest gap.
- **ProBID-Net** — *Chemical Science* (RSC), closed review, no OpenReview forum (matches [[submission-hygiene]]).
- **RedNet** (bioRxiv 2026-05-09) — no forum indexed yet (plausibly in an unopened ICLR-2027-style pipeline).

**The load-bearing finding — Complexa.** An in-silico-only binder-design paper whose evaluation leans entirely on
structure-predictor oracles got an **ORAL**, even though multiple reviewers named oracle-reliance / in-silico-only
as weaknesses. The 10/10 review said the risk aloud ("massively overrelying on AlphaFold… biases… would not be
picked up until wet-lab") and it did NOT cost the score. Reviewer-credited survival recipe: ≥2 independent oracles
(AF2 **and** RF3), rich compute-normalized ablations, no easy-task/weak-baseline corner-cutting, code-release
commitment, a plain up-front pipeline overview, limitations owned explicitly. → strong external evidence our
in-silico-only design is venue-viable IF framed like Complexa; the "in-silico ⇒ auto-reject" fear is not borne out.

**Recurring critiques to pre-empt (each independently raised by ≥2 reviewers where noted):**
1. **"Incremental over the nearest prior method"** — the single most repeated complaint (PRISM vs AIDO.Protein-IF;
   Complexa vs La-Proteína), 2 reviewers/paper. → OUR named #1 risk (vs BA-Cycle/RedNet). Answer at mechanism level
   on p.1–2, not in rebuttal.
2. **Single-fixture / limited-dataset** — named even in an 8/10 BA-Cycle review. → state the SKEMPI-only scope ourselves.
3. **Theory without a paired empirical payoff** — PRISM's harshest reviewer called a derivation "trivial" once the
   empirical link felt thin. → keep the no-go proposition welded to its measured non-vacuity (~63% irreducible).
4. **Vague data-splitting / leakage** — 2 BA-Cycle reviewers independently pressed fold-splitting leakage. → state our
   complex-level, burial-matched splitting plainly and early.
5. **Unfair baseline settings** (mismatched temperature/sampling) — named in PRISM + MoMPNN. → confirm every comparator
   (FoldX/geometry/substitution/log-odds) runs under identical conditions; foreground the decoding-order-averaging rule.
6. **Undefended static calibration/threshold** — MoMPNN. → defend fixed α / post-hoc-affine L→kcal/mol as principled
   zero-shot (already our "calibration is not fitting" + "don't tune L" stance).

**Rewarded (emulate):** owning limitations (BA-Cycle 10/10, Complexa 10/10); dual-oracle anti-circular eval;
front-loaded comprehensive + compute-normalized ablations; explicit code-release commitment; a one-paragraph
plain-language pipeline overview for a multi-component method.

**Apply-to-us checklist (✓ = already in draft; → = action, best done in the full LaTeX assembly):**
- ✓ ≥2 independent anti-circular oracles (AF2/Boltz-2 ipTM + FoldX ΔΔG_bind) — → **foreground earlier**, not as an afterthought.
- → **"Why not just BA-Cycle/RedNet," mechanism-level, in §1 and top of §8** (our #1 risk; make it unmissable).
- ✓ single-fixture + in-silico-only owned in §9 — add the one-paragraph "why the interim signal is informative" justification.
- ✓ leakage stance (complex-level, burial-matched) — → **state plainly and early**, not only in the appendix table.
- → **methods line asserting identical baseline settings** (temperature/decoding-order/sampling) for every comparator.
- ✓ code-release commitment — the prune guarantees the anon repo is genuinely complete at submission.
- → **one-paragraph pipeline overview** up front (decomposition → no-go → beyond-X → CFG-steering → anti-circular eval).
- ✓ LLM-usage disclosure + resolvable-DOI citation hygiene (run verify-references before submission; [[submission-hygiene]]).
