# IDEATION — living scratchpad for the confidence-vs-competence / leverage line of work

*Not paper prose. A place to let the framings sink in and to hold next-paper ideas. Updated as of
2026-09-09, after the FoldX physics results landed and the "L beyond physics" conditional result was recovered.*

---

## Naive mechanism (what the confidence tilt actually is)

`naive` = steer frozen ProteinMPNN toward its **own confidence**: at every interface position (same set as `L`),
take the model's per-residue log-odds vector, mean-centre it (`conf = logP − mean(logP)`), normalize, rescale to
`L`'s per-position magnitude, add as a logit bias, sample. It points at the residues the model already prefers
(its argmax), applied at **all** interface positions, not selectively at unconfident ones. `naive` = "be more
foldable/confident"; `L` = "be more binding-favorable" (often *frustrated* residues) — **identical but for
direction, at matched dose.** That is what makes it the honest specificity control: not "L vs nothing" but "L vs
the obvious foldable alternative."

## The three framing points (updated for FoldX landing)

1. *"Pre-empts the reviewer's best attack."* A meta-argument for us; the paper earns it simply by disclosing the
   result honestly. (Done.)
2. **The judge-wins / ipTM-wins divergence IS the frustration thesis** — a foldability metric under-credits
   frustrated binding residues, so ipTM's "failure" is *evidence for* the thesis, not against it. (In §4.)
3. **`L` is binding-specific beyond confidence — and now BOTH the inverse-folding judges AND the physics FoldX
   model agree on it.** Upgraded from "per inverse-folding judges" to "per inverse-folding judges *and* a physics
   energy function." Strongest form; in §4, the abstract, and §8.

## RedNet β-mask + naive+L — corrected take

I do **not** have precise published details of RedNet's β-mask; I inferred "balance foldability vs binding" from
code-positioning and should not have stated it as fact — **retracted.** And the point stands: RedNet already does
the L-direction, so "naive+L re-derives RedNet" is **not** a real objection — **retracted.** The genuine reasons
`naive+L` is a *follow-up*, not core to the diagnostic paper: (a) it's *optimization*, not the *diagnosis* this
paper makes; (b) it adds a mixing weight (a small tune). But it is promising and nearly-free (reuse both tilts):
`L` alone costs a little foldability, `naive` alone misses binding, `naive+L` could get **both** — *"a
training-free recipe for binders that fold and bind."* Strong next-paper; kept out of the diagnostic paper to keep
"`L` is the binding direction" clean.

## naive+L / tweak-L

- **naive+L:** likely better folding *and* binding — promising, nearly-free follow-up; introduces a mixing knob →
  optimization not diagnosis → next paper (now **pilotable on FoldX**, the pipeline is built).
- **Tweaking/optimizing `L`:** *don't*, for this paper — `L`'s value is that it is principled and **zero-shot**
  (the CFG direction ∝ −ΔΔG). A tuned `L` invites "you fit it." The path to a stronger steering result is a
  **better readout** (physics ΔΔG — done), not a tuned `L`.

## L-extension ideas (next-paper)

1. **Calibrate `L` → kcal/mol** — now concrete: we have paired physics ΔΔG (and experimental ΔΔG) on the same
   mutations. Regress to a slope/temperature. Removes limitation (c) and is the bridge to the folding project.
2. **naive+L design recipe** — foldable *and* binding, nearly tune-free.
3. **Design with the 2nd derivative** — co-optimize residue *pairs* via the partner-ablated couplings (epistasis),
   not just single sites.
4. **Generalize the CFG recipe to other conditioners** — ligands, nucleic acids, PTMs, multi-partner complexes.
5. **The joint fold+bind thermocycle** — sum the two legs (ΔΔG_fold + ΔΔG_bind) inside a frozen model.
6. **Per-burial-stratum temperature** — the paper flags kT_model as position-independent; a stratified calibration
   might sharpen `L`'s ordering (careful — edges into tuning).

## `L` beyond physics — the recovered conditional result (NEW, 2026-09-09)

The crude equal-weight L+FoldX ensemble was null; the **conditional** test is not. Partial
Spearman(`L`, experimental ΔΔG | FoldX) = **−0.17 [−0.23, −0.12]**, P(<0)=1.0 → the zero-shot mixed derivative
carries binding rank-signal **beyond the fitted physics energy function**. This is the paper's beyond-X ladder
(geometry → conservation → one-pass log-odds → **physics**), and it *reinforces* the scalar-vs-mixed split: the
scalar KL equals ΔSASA and does **not** beat physics; the mixed derivative adds beyond it. Folded into §4/§8.

## Judge (leverage) vs folder (ipTM) — and now physics

- "+0.159" is the **judge's leverage** (an independent inverse-folding model's own partner-ablation score), **not
  ipTM.**
- **ipTM** = a structure predictor's cross-chain assembly *confidence* (it folds both chains, sees the partner) —
  apples-to-oranges with leverage in magnitude; only the **paired-contrast sign** is valid.
- `naive` **wins** ipTM (packs confidently) but **loses** the judges and **now loses FoldX** (binds worse) —
  "packs well" ≠ "binds well," no contradiction.
- **ipTM is itself largely blind to binding-specificity** — a structure-level confidence, our thesis one level
  up. **FoldX is the tiebreaker with an explicit binding term, and it votes with the judges.**

## Cross-project (thermocycle)

Two legs of one cycle (ΔΔG_fold + ΔΔG_bind, shared ESM-IF1 checkpoint). `personal_projects/thermocycle_joint/`
has README.md (3-paper plan binding → folding → combined; the cycle thesis; killer experiment = **sum the two legs
inside a frozen model**) + FEEDBACK_for_folding_session.md (paste-ready pitch). The **FoldX calibration (idea #1
above)** is now the concrete shared-recipe bridge. Fresh dir, not yet a git repo.

## Hero figure

Both ready; lean = **PyMOL** as the main hero (persuasive, now readable with legend/labels/interface markers),
biotite as backup (no-conda, CSV-traceable). Fig. P (FoldX) is a clean addition to the steering story either way.
Decide at figure-selection.
