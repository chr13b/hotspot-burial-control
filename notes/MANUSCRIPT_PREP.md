# Manuscript preparation roadmap (revisit when ready to package)

Master checklist for taking the draft to an ICLR submission. Nothing here is lost; work top-down.
Odds context: ~0.85+ *once R1/R2 closed and the scope trim + figures land*. Current: R1 CLOSED (R²(L|P)).

## PHASE A — remaining Tier-1 fixes (do before restructure)
**Experiments (CPU, inputs in hand):**
- [x] **R1** — R²(L|P_full) flexible learner → ~70% irreducible (`r2_leverage_from_P.csv`). DONE, threaded in.
- [ ] **R2** — run `CPI(L | geometry)` + geometry+|L| ranker on the OpenFold3 / AF2-multimer predicted
      backbones we already have for the 127 shared complexes (`expA_gap_summary.csv`, `expD_gap_summary.csv`).
      Closes "L never computed on a real predicted backbone". Either outcome strengthens the paper.
- [ ] **Model symmetry** — leverage on **PiFold + MIF** (scorers exist: `src/models/ftax_{pifold,mif}.py`
      expose `*_conditional_logprobs`; monomer construction is in `leverage_esmif.py`). Makes the feature-class
      law rest on the same 5-architecture panel for both halves. Bonus: MIF+PiFold are the two with residual
      §5 deficits → connects §4 and §5.

**Free writing fixes:**
- [ ] **CFG reframe (highest-leverage, free).** Name L as the classifier-free-guidance / contrastive-decoding
      direction (`logit_bound + α(logit_bound − logit_apo)`), conditioner = binding partner; confidence = the
      conditional marginal. State the general principle in the abstract + intro. Cite CFG (Ho & Salimans),
      contrastive decoding (Li et al.), expert-minus-amateur decoding. One line: `L_i(a)` is a difference of
      pointwise mutual informations between residue identity and partner presence.
- [ ] **Entropy-normalized effect sizes.** Position-level hotspot base rate 2.44% → label entropy 0.1147 nats.
      Report the normalized column (leverage +0.0048 = **4.2%** of label entropy, ~40% of ΔSASA's 11.2%).
      State the base rate in §2 (currently absent). Stop juxtaposing position-level CPI (2.4% base rate) with
      mutation-level CPI (~50%) in one abstract sentence without flagging the scale difference.
- [ ] **Enrichment@budget** alongside AUROC for triage (capture@3/@5 already in `leverage_triage.csv`):
      "geometry finds 3.4/10, geometry+L finds 4.1/10".
- [ ] **Unify the geometry baseline.** §3 full-geometry AUROC 0.734 (5,742 pos, `baseline_audit.csv`) vs §8
      0.704 (13,401 pos, `w4_combined_ranker.csv`) — same features, different samples, lift measured against the
      weaker. One table, one sample, all marginals + the paired lift with CI.
- [ ] **Report both scalarizations everywhere** (L→Ala and |L|_rms), pre-declare one primary. The alanine
      triage lift CI spans zero (`w4_combined_ranker.csv`); capture@3 ordering flips. Both are in the CSVs.
- [ ] **ProBID §5 full estimator ladder** as a table: uncontrolled hot-minus-nonhot **+0.098 [+0.019, +0.176]**
      (reverse sign!), like-4-like pooled +0.014, deep-scan ≥5 −0.113. Show all three, not two. Validate the
      port on the *matching* quantity (our non-hotspot 0.445 vs their 0.472; our hotspot vs their 0.334), not
      our-overall-vs-their-non-hotspot.
- [ ] **Paired CI** for the "+0.030 AUROC beyond supervised" abstract claim (`leverage_effect_size.py` never
      computes the difference).
- [ ] **LOW hygiene:** move `.md`-only numbers into CSVs (masked drop-3 +0.0041, +0.71 estimator corr, −0.08/
      −0.14 Spearmans, coupling +0.610, the 27% concentration → attach to the *unmasked* run); add n/seed/command
      to `w_placebo_ladder.csv`, `w4_combined_ranker.csv`, `nugget_cpi.csv`; fix trace pointers (0.704→0.717
      cites triage CSVs but lives in `w4_combined_ranker.csv`; masked numbers cite `skempi_conservation.csv` but
      live in `skempi_conservation_masked_cpi.csv`; §4 reverse partial prints [−0.145,−0.054] vs csv
      [−0.142,−0.047]); coupling sign-accuracy overclaim (`p3_sign_verify.csv`: |C|>p75 model 0.65 > majority
      0.621 — it *does* beat on that subset).

## PHASE B — figures (start now, before cutting text)
Fable-5 figure-design plan pending (see when it lands). Target: 4–5 main-text figures, ICLR-quality
(print-safe + colorblind-safe palette, publication typography, no chartjunk). Render in matplotlib → PDF.
Candidate set: (F1) CFG-direction schematic — confidence=conditional marginal vs leverage=derivative wrt
conditioner; (F2) feature-class law — placebo floor + scalars-at-floor + leverage clears, beyond geometry AND
conservation; (F3) dose law, per-model decay + half-life; (F4) second derivative / coupling; possibly the
constraint gradient. Load the `dataviz` skill before rendering.

## PHASE C — restructure to ~9 pages (~6,000 words; currently 9,562, §4=4,609; ZERO figures rendered)
Reviewer's recommended arc (main text): (1) the no-go, armed — identifiability + R²(L|P) irreducibility +
vector-P control; say "we prove", drop "theorem". (2) feature-class law — one table/sample/baseline, both
scalarizations, ≥4 architectures, lead with **conservation** (the real feature set). (3) dose law with §6
predicted-backbones **folded in** (NOT appendix — it's the answer to "does this matter in the design regime").
(4) triage with enrichment@k. (5) coupling, ONE paragraph.
**Appendix:** ProBID §5 (compressed to a corollary + full ladder table); §7 mechanisms; catalytic; Bennett
occlusion; reciprocity; effect-size-vs-supervised.
- **RECONSIDER the constraint-gradient demotion.** Adding per-class **leverage**-AUROC (now in
  `threepoint_law.csv`: 0.641/0.628/0.701, flat+high, all clear chance) upgrades it from a caveated 4-point
  confidence trend to a *divergence of the two feature classes across a controlled axis* — the designer flags
  this as the highest-ROI number in the set and "roughly level with Fig 4". Recommendation: **KEEP** it (as
  upgraded Fig 5 + a short main-text paragraph), rather than demote. Await user confirmation.
- [ ] Correct `FRAMING_PLAN.md`: it sends §6 to appendix — DON'T; merge §6 into the dose law.

## PHASE D — release hygiene / de-identification (reviewer sees a clean repo)
- [ ] **Absolute → relative/config paths.** Leaks found: `/home/chris/ftax/...` in
      `src/p_confidence_gradient{,_affinity}.py`, `src/models/ftax_{esmif,pifold,mif}.py`; `/mnt/c/Users/chris/...`
      sys.path in `src/decoding/test_steer.py`, `src/models/ftax_panel.py`. Introduce one `ftax_paths.py` (or a
      `FTAX_DATA`/`FTAX_MODELS` env with repo-relative defaults); 29 scripts use `~/ftax`.
- [ ] **Sherlock / $SCRATCH** references in `expA_*`, `expB_commitment.py`, `expC2_*`, `dsasa_matched_sens.py`
      comments, and several CSV `command` columns → genericize or move to an env var.
- [ ] **Curate `notes/SHERLOCK_*.md`** (7 files) out of the release tree.
- [ ] **Code comments** — scrub anything identifying (machine names, personal layout, dated banter).
- [ ] **`results/INDEX.md`** — every `→ file.csv` trace in the paper mapped to script + claim (doubles as the
      reproducibility appendix). Low-risk, zero path changes.
- [ ] Full directory reorg — POST-submission only (103 scripts, 96 hardcode `results/`, 57 sys.path imports).

## PHASE E — pre-submission (NO ROOM FOR ERROR — user's standing instruction)
- [ ] **Full-manuscript numbers-vs-CSV audit pass** over the *entire* final draft (same method as the two audit
      passes already run) BEFORE it goes live. Every number re-verified against its committed CSV.
- [ ] `verify-references` skill on the bibliography (phantom/wrong/​truncated citations).
- [ ] LaTeX conversion; formal `.bib` from `REFERENCES_verified.bib`.
- [ ] Zenodo archival (TIME-SENSITIVE: Sherlock SCRATCH purge ~2026-10-09).
- [ ] Anonymize for double-blind (no author/identity in the PDF or the linked repo).
