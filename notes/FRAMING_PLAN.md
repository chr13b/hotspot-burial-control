# Framing & positioning plan — how to sell this paper (and not get sunk on shape)

The strategic audit's verdict: **accept-trajectory science, not-yet-accept shape.** The results are above
ICLR median; what sinks papers like this is presentation. This plan is the pipeline to fix that. It drives
the W1 (reframe/retitle) and W6 (scope-trim) edits and the order we make them.

## The core diagnosis (what a hostile reviewer sees today)

1. A **"theorem"** in the title/abstract that reduces to *a function of P alone cannot compute a function of
   (P,Q)* — a definitional observation a reviewer can dismiss as trivial.
2. **Nine results competing for nine pages.** ICLR reviewers read pp. 1–4 properly; everything after is skimmed.
3. Defended against a **geometry baseline the hotspot field does not use** (burial+nbr+ΔSASA) — reads as
   beyond-a-strawman, especially since our own catalytic section proves *conservation* is what finds sites.
4. The **actionable claim's CI includes zero** (|L| 0.694 vs ΔSASA 0.664, +0.030 [−0.001, +0.061]).

None of these is a science problem. All four are fixable by **reframing what we lead with**.

## The reframe (W1) — three moves

**Stance (the guiding principle): honest AND confident, never timid.** We are not framing-afraid. We steer the
reviewer's attention to the strongest true claims and present them boldly; the honest hedges on the modest bits
(epistasis sign) are *credibility signals* that make the bold claims land harder, not apologies. Confidence and
honesty are not in tension — the most confident framing is the one that survives scrutiny, so we arm every claim
with its measurement and lead with it. We control the light the paper is seen in; we do not dim it.

**Move 1: keep the formal result, ARM it, present it confidently.**
Do not hide the identifiability result out of fear a reviewer calls it "definitional" — that fear is the timid
move. The confident move is to state it precisely as a **proof** ("confidence is blind to binding by
construction") and inseparably pair it with the measurement that makes it non-trivial, so the "so what, it's a
definition" attack is dead on arrival: we *measured* that it is not a definition. (Prefer "we prove" over the
bare label "theorem" — not to soften, but because "we prove X, and here is the measurement showing X bites" is
strictly more forceful than announcing a theorem and leaving it bare.) Show it is *not* definitional by
measuring how much is left after the "trivial" part is removed:
- The sharp, non-trivial statement (currently missing, and it is the whole point): **X_monomer is a
  deterministic function of X_complex, so leverage L *is* computable from the structure — it is simply not
  computable from P.** The field's scalar readouts factor through a lossy statistic of the same distribution;
  the second pass is a **computational**, not an informational, necessity. This pre-empts "a big enough head
  could learn Q from P" — no, the information is present; the *scalar readout* discards it.
- Back it with the number: **30% (ProteinMPNN) to 73% (ESM-IF1) of leverage's spread survives matching the
  bound distribution P** (full-P-matched pairs). The blindness is quantitative, not a tautology.

**Move 2: retitle around the mixed derivative, not the "conditioning-set artifact."**
Current title advertises the §6 story; the spine is the decomposition. Candidate direction (pick at LaTeX):
*"What inverse-folding models know about binding lives in the mixed derivative, not the confidence."*
Short abstract (~30 words currently) → lead with: confidence is blind at hotspots (a property of the IF
*class*, 2 model families), the binding signal is the partner-ablation mixed derivative, it is a dose law of
backbone accuracy, and it reaches to pairwise epistasis.

**Move 3: turn the scope limit into a result about *published methods* (free).**
The dose law (leverage collapses by ~1 Å of backbone error) is currently a caveat. Reframe as a **field-level
prediction**: BA-Cycle, RedNet, and StaB-ddG all read the *same* mixed derivative, so **all three inherit the
~1 Å cliff** on predicted backbones. Costs nothing (the ladder exists); converts our limitation into a
falsifiable claim about others' methods. (If the ESM-IF1 dose-cliff replicates — a cheap check — it becomes a
law about the readout *class*, not one network.)

## The scope discipline (W6) — 5 in, the rest out

**Main text (the load-bearing arc):**
1. Confidence is not competence — the no-go (identifiability + measured non-vacuity).
2. The decomposition — leverage adds binding info beyond geometry **and beyond the one-pass log-odds** (W2),
   on **two IF model families**, beyond **conservation** (ceiling-raiser, pending).
3. Triage — the actionable payoff: rank by *geometry + the mixed derivative* (the significant combined claim).
4. The dose law — mechanism + field-level prediction.
5. Coupling — the second derivative reaches pairwise epistasis (existence, honestly modest).

**Appendix (with 2–3 sentence pointers in main text):**
- §5 burial artifact → compress to a **one-paragraph corollary** of the no-go (recovery is a scalar of P, so
  the published deficit falls out of the no-go — do not re-litigate it over a page).
- §6 conditioning-set / §7 competing-mechanisms → appendix.
- Catalytic generalization → appendix, but keep **one sentence** in main text because conservation now runs
  through both function types (it stops looking bolted-on).
- Reciprocity, effect-size-vs-supervised, drop-controls → appendix tables.

## How to sell each result (the one-liners)

| result | the sentence that lands | the trap to avoid |
|---|---|---|
| no-go | "The readout the field uses is a lossy statistic; the binding information is present but not in it." | calling it a theorem |
| decomposition | "Beyond geometry, beyond the one-pass score, beyond conservation, on two model families." | "beyond burial" alone (strawman) |
| triage | "Add the mixed derivative to geometry: +0.013 AUROC, significant." | "|L| beats ΔSASA" (CI spans 0) |
| dose law | "The signal — and every method built on it — dies at ~1 Å of backbone error." | presenting only as our caveat |
| coupling | "It reaches the second derivative: the model knows *epistasis*, modestly." | overselling; lead with the ablation mechanism |
| catalytic | "Same dissociation, other function type: conservation finds sites, IF confidence does not." | a separate mini-paper feel |

## The pipeline (execution order)

1. **Ceiling-raiser first** (conservation control, running) — it decides how strong claim #2 can be.
2. **Mechanical quick wins** while it runs: W4 combined ranker, W5 unify triage baselines, W9 CPI unit,
   W8 ProBID dose-response, AB-Bind reconciliation. (Numbers already verified by the audit.)
3. **W1 reframe** in prose (retitle, no-go framing, non-vacuity sentence, field-level dose-law prediction).
4. **W6 scope-trim** — move sections to appendix, compress §5 to a corollary. Do this LAST (after the science
   is locked), at LaTeX conversion.
5. Fold the conservation result into claim #2 and the abstract when it lands.

## The one-paragraph pitch (for the abstract's spine)

> Inverse-folding models are increasingly used to score protein–protein interface hotspots, and their
> confidence is blind there — a property of the model class, not one network. We show why: every scalar the
> field reads off the model is a functional of the bound-conditioned distribution alone, and the binding
> signal lives in a **mixed second derivative** — the change in the model's log-odds when the partner is
> ablated — which no such scalar can express, though the structure determines it. This partner-ablation
> leverage adds binding information beyond geometry, beyond the one-pass log-odds, beyond conservation, and
> across two model families; it is the best single hotspot ranker when added to geometry; it is a **dose law**
> of backbone accuracy that every method built on the same derivative inherits; and it extends to the second
> mixed derivative — pairwise binding epistasis. Confidence is not competence; competence is in the derivative.
