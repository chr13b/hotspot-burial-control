# Pre-submission audit prompt — factorization-tax / hotspot-burial-control

*Adapted for THIS repo from a template that caught real problems in a prior project (a false per-model claim,
point-estimate overstatements, mis-attributed citations, a codename leak, a stale section compiling into the PDF).
Paste everything below the line to a capable agent with repo + web access before submitting. Run it even if
"nothing changed." Give it repo write access only for mechanically-safe fixes; anything that changes a claim's
meaning is proposed, not applied.*

---

You are auditing an ICLR 2027 paper for submission. Be adversarial. Find, in priority order: (1) claims not
supported by the underlying data, (2) hallucinated or mischaracterized citations, (3) internal contradictions,
(4) anonymity leaks, (5) reproducibility gaps and non-load-bearing overhead, (6) presentation defects. Assume a
skeptical reviewer who has the released code and can recompute every number.

**Inputs.** Paper (single source of truth): `notes/PAPER_DRAFT.md`. LaTeX (when built): `latex/main.tex`,
bibliography `latex/iclr2027_conference.bib`. Raw data: `results/*.csv` (~269 files) and `results/FINDINGS_*.md`,
`results/PREREG_*.md`. Analysis + figure scripts: `src/*.py` (esp. `src/fig_*.py`, `src/leverage_ladder.py`,
`src/foldx_*`, `src/cfg_steer*`, `src/analyse_*`). What-is-folded index: `notes/FOLDED_STATUS.md`. Project rules:
`CLAUDE.md` (falsifiers pre-registered; honest null valid; never fabricate; every number → committed CSV; cite
only fetched URLs; positive controls before a zero).

**Ground rules.** Recompute from the raw CSVs, never trust the prose or a FINDINGS file. Fetch URLs, never cite
from memory. When you verify a number, quote your recomputed value next to the paper's. Fix mechanically-safe
issues directly (typos, a wrong §-ref, a stale CSV pointer); propose wording for anything that changes a claim.
Never weaken honesty to save a headline.

## Phase 1 — Numbers audit (data vs paper)
For every bolded number and every number-bearing sentence in `PAPER_DRAFT.md`, recompute it from the source CSV in
`FOLDED_STATUS.md` (complex-clustered bootstrap where the paper uses one; SEED=20260803, NBOOT as in the script).
Flag any mismatch beyond rounding. Pay special attention to hand-edited numbers (abstract, intro, table cells,
§9). Confirm every figure regenerates (`python3 src/fig_overview.py`, `fig_foldx.py`, `fig_ladder.py`, `fig1..5`,
`fig_robustness.py`) and matches what the paper shows; confirm each passes `S.save` (≤5.5in). Confirm the
best-of-k / mean-over-k pair, the partial-correlation ladder rungs, the predicted-backbone judge+ipTM numbers,
and the AB-Bind / ATLAS / calibration numbers all trace to their CSVs.

## Phase 2 — Honesty audit (claims vs evidence)
List every superlative/comparative: "beats", "clears", "beyond", "not a crystal artifact", "barely attenuated",
"competitive", "generalizes", "first", "only", "decisive". For each check: (a) supported at the stated strength
(difference → paired CI excludes 0; equality → CI includes 0); (b) holds on EVERY model/fixture claimed or is it
silently one-model/one-fixture (state per-model/per-fixture); (c) a point estimate narrated as a real difference
(esp. the AB-Bind partial that grazes zero — must read "underpowered/indeterminate", not "confirmed"). Hunt
contradictions between sections: abstract vs §4; intro vs §9 limitations; the title "confidence is not competence"
vs the naive-strong-baseline insight (§4) — confirm the anti-contradiction guard is intact (confidence-as-readout
at chance; L still beats the confidence *direction*). Read the conclusion against every section it summarizes.
Confirm best-of-k is labeled a selection/upper read with mean-over-k visible.

## Phase 3 — Citation audit (no hallucinations, no drift)
Run the `verify-references` skill on `latex/iclr2027_conference.bib`. For every cited work: confirm it resolves
(Crossref/arXiv/DBLP), the title/authors/year match, and the sentence citing it matches what it actually does
(flag mischaracterization). Confirm the FoldX triplet (Guerois 2002 / Schymkowitz 2005 / Delgado 2019), BA-Cycle,
RedNet, StaB-ddG, ProBID-Net, UMA-Inverse, Frellsen, Janusz, ProteinMPNN, ESM-IF1. Fresh prior-art sweep (last
4–6 weeks) for each core claim (mixed-derivative decomposition, beyond-geometry control, training-free steering,
L-beyond-physics); return GO/SOFTEN per claim with fetched URLs.

## Phase 4 — Anonymity sweep (double-blind)
Grep the paper, the built PDF text, and the ENTIRE release tree for: `bertsch`, `cbertsch`, the email, ETH/any
institution, `/scratch/users/...` cluster paths, `Claude-Session`, `Co-Authored-By`, and any private-repo
identifier. (Known state: ~66 tracked files carry the name/username and git history's author is a real name+email
→ the anon submission repo MUST be a **fresh `git init`** with no history, paths/name scrubbed.) Confirm the built
PDF prints "Anonymous authors", no acknowledgements, no self-identifying self-citation ("our prior work"), and the
anonymized-repo link resolves.

## Phase 5 — Reproducibility + load-bearing-only
Build the reproducibility map `INDEX.md`: every paper claim/figure/table → its CSV → the script that produced it.
Then **the load-bearing decision (explicit user requirement): ship ONLY files actually used for the paper** — walk
every `results/*.csv` and `src/*.py`; keep those an `INDEX.md` row depends on (or a reproduce step needs), and
list the rest (pilot/scratch `_*.csv`, `*_audit2`, superseded shards where a consolidated file exists, orphan
intermediates) for exclusion. Confirm every script compiles, a top-level `reproduce.sh` regenerates every reported
condition, and a reviewer could reproduce each number. Propose the submission-repo directory layout
(`paper/`, `src/`, `results/` or `data/`, `figures/`, `README.md`, `INDEX.md`).

## Phase 6 — Presentation & hygiene
No `⟨PENDING⟩`/TODO/internal-note text in the body; every figure referenced from the text; every table captioned;
the PDF builds with zero undefined refs/citations; the LLM-usage disclosure matches ICLR 2027's mandated format;
page count as expected. Check §-cross-references resolve to the right sections.

## Output
Severity-ranked findings table: BLOCKER (false/unsupported claim, hallucinated cite, anonymity leak), MAJOR
(mischaracterized cite, unreproducible number, contradiction), MINOR (wording, hygiene), NOTE (camera-ready debt).
Each: location, what's wrong, evidence (recomputed number or fetched quote), fix (applied or proposed). Finish
with an explicit GO / NO-GO and the list of fixes already applied.
