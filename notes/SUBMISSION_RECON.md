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
   Atlanta). **Best fit AND still open: deadline Sep 30 2026 (AoE)** — the earliest actionable route, five days
   after the ICLR paper deadline; dry-lab + wet-lab tracks (we fit the dry-lab track). Generative (inverse-folding)
   models + experimental/physics validation for binder design — exactly our steering + ΔΔG/FoldX story.
   Non-archival. *(gembio.ai; neurips.cc 2026 workshops blog, fetched 2026-09-11)*
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
- **Correction to the pessimism above:** GEM is *open* (Sep 30). So a NeurIPS-2026 non-archival companion IS still
  actionable this cycle via GEM; MLSB 2026 date TBD (verify mlsb.io).

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

## Submission-repo checklist (packaging phase)
- [ ] Fresh `git init` (no identifying history) for the anon mirror.
- [ ] Scrub `bertsch`/`cbertsch`/email/scratch paths from all files (env scripts, latex, any hardcoded paths).
- [ ] `INDEX.md`: paper claim → CSV → script that produced it (reproducibility map).
- [ ] Strong `README.md`: what/why, install, one command per figure/table, data provenance, license.
- [ ] Numbers-vs-CSV full audit (every bolded number in the paper traces to a committed CSV).
- [ ] verify-references on the .bib; ensure no self-identifying citations.
- [ ] Remove pilot/scratch CSVs not needed for reproduction (e.g. `_pilot_*`, `*_audit2`, raw shards if a
      consolidated file exists).
- [ ] Zenodo archive before ~Oct 9 (data-archival memory deadline).

## OpenReview recon — PLAN (deep pass queued)
Goal: read the OpenReview threads of 4–6 closely-related recent papers (inverse folding / binding / protein design
at ICLR/NeurIPS) and extract what reviewers rewarded and what they punished, then apply it. Candidates to pull:
BA-Cycle, RedNet, StaB-ddG, ProteinMPNN-adjacent ICLR/NeurIPS papers, ProBID-Net, any ESM-IF binding papers.
For each: the review scores, the top reviewer complaints (baselines? novelty vs prior? single-fixture? in-silico?),
the rebuttal moves that worked, and any desk-reject/ethics/format issues. Record concrete "do/avoid" items here.
*(To fill next turn.)*
