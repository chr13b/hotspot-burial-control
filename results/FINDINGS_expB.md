# FINDINGS — Experiment B: commitment ordering on a coupled co-design model (MultiFlow)

Run 2026-08-10 on Sherlock (MultiFlow monomer co-design checkpoint, `last.ckpt`; V100, torch 2.0.1+cu117).
Every number traces to a CSV in this directory. **In progress** — the ordering sweep (§2) is being built.

## 0. Coupling gate (BRIEF §4) — PASSES

`multiflow/data/interpolant.py`: the discrete (CTMC masking; `_corrupt_aatypes`,
`_aatypes_euler_step[_purity]`) and continuous (R³ translation + SO(3) rotation flows;
`_corrupt_trans/_corrupt_rotmats`, `_*_euler_step`) processes are coupled **only** through a shared
time index — the default path sets `so3_t = r3_t = cat_t = t`, each channel's corruption/step is a
function of its own state + a time variable, no cross-channel entanglement. Rates are independently
config-specifiable (`interpolant.trans.sample_schedule`, `rots.sample_schedule`/`exp_rate`;
`aatypes.{schedule,schedule_exp_rate,temp,noise,do_purity}`); `codesign_separate_t`/`t_nn` even allow
fully decoupled per-channel time. Default inference = **`do_purity: True`, `temp: 0.1`,
`num_timesteps: 500`** — the confidence-ordered unmasking at T=0.1 that BRIEF §2.3 contradicts. The
experiment is well-posed. **Model scope caveat:** the released checkpoint is monomer-trained
(`oligomeric: monomeric`, 60–512 res), so the ordering sweep (§2) uses a monomer fixture with a
**geometric** hotspot proxy (burial / frustration), not the SKEMPI interface fixture — an
in-distribution test of the §2.3 mechanism (decided with the user).

## 1. Commitment times t*_str vs t*_seq (F3) — NUANCED: structure first, but by a small margin

Unconditional monomer co-design (MultiFlow's native task), default 500-step schedule; t* = 0.5-crossing
of normalised agreement between the model's endpoint prediction x_hat_1(t) and the realised final x_1
(token-argmax agreement for sequence; Cα contact-map Jaccard overlap for structure). 3 seeds × 3 length
bins (100/200/300). `src/expB_commitment.py` → `results/expB_commitment.csv`.

| | mean | range | structure-first samples |
|---|---:|---|---:|
| **t*_str** | **0.053** | 0.039–0.070 | — |
| **t*_seq** | **0.094** | 0.053–0.154 | **9 / 9** |
| gap (t*_seq − t*_str) | +0.041 | — | — |

**Structure commits before sequence in every one of the 9 samples** — directionally exactly what §2.3
predicts ("choose the shape first, fit chemistry to it"). **But the mean gap (0.041) is below the
pre-registered F3 margin of 0.05**, so by its literal criterion **F3 fires** (t*_seq ≤ t*_str + 0.05 in
7/9 samples). Honest reading: the structure-before-sequence separation is **real but small** — both
channels' endpoint predictions stabilise very early (t* ≈ 0.05–0.09, the first ~5–9% of the reverse
process), structure marginally first. The window in which "structure is decided but sequence is not" is
narrow (~0.04 in t).

> ### 🟠 F3 fires marginally
> This weakens, without cleanly refuting, the commitment-ordering diagnosis: the ordering *is*
> structure-first, but not by the margin the falsifier required. The decisive question therefore moves
> to the **direct** intervention (§2): if reordering the discrete unmasking measurably changes recovery
> at geometrically-demanding positions, the mechanism is actionable regardless of the small t* gap; if
> not (F4), the knob is inert.

*Caveat — schedule-resolution sensitivity:* t* depends on `num_timesteps` (a coarse 100-step probe gave
a much larger gap, t*_seq≈0.28). The 500-step schedule above is the released default and is canonical;
the sensitivity is reported, not hidden.

## 2. Unmasking-order sweep + demanding-position recovery (F4) — pipeline validated; full run going

`src/expB_ordering.py` → `results/expB_ordering.csv`. Monomer SDEdit with a self-generated reference
(in-distribution): generate a reference monomer by co-design; SDEdit-corrupt both channels to t0=0.5;
re-denoise with the discrete unmasking reordered — **purity** (released default, dynamic confidence) /
**burial-first** / **anti-burial** / **random** — continuous schedule fixed. Metric =
recovery(buried) − recovery(exposed) per order (buried = top-tertile Cα neighbour count, the
geometric demanding-position proxy). §2.3 predicts burial-first should raise buried recovery most.

**Result (8 refs × 3 seeds = 24 per order; `results/expB_ordering.csv`):**

| order | buried − exposed recovery (mean ± SD) |
|---|---:|
| purity (default) | −0.074 ± 0.068 |
| random | −0.078 ± 0.067 |
| anti-burial | −0.084 ± 0.050 |
| burial-first | −0.085 ± 0.077 |

> ### 🔴 F4 FIRES — the ordering knob is inert
> The order-span (max − min of the per-order means) is **0.012**, far below the seed-to-seed SD of
> **0.065**. Reordering the discrete unmasking does **not** move recovery at demanding (buried) positions
> beyond noise. In particular **burial-first** — §2.3's exact prescription (decide the demanding
> positions first, while the continuous channel is still hot) — is indistinguishable from purity and
> random, and if anything trends slightly *worse*. (A noisy n=2 pilot that suggested "order matters" was
> just noise; at n=24 the span collapses to 0.012.)

This is consistent with §1: because both channels' endpoint predictions stabilise very early and close
together (the commitment window is only ~0.04 wide), there is little room for reordering to matter.

*Limitations, stated up front:* (a) monomer fixture with a burial proxy, not SKEMPI interface hotspots
(the released checkpoint is monomer-only); (b) the reference is self-generated under the default
purity order, so purity has a mild home-field advantage — hence the **within-order buried-minus-exposed**
metric rather than raw recovery; (c) single noise level t0=0.5. **Part 4 (binding readout) is N/A for a
monomer model** — no interface exists in monomer designs; its monomer analogue (designability: fold the
designed sequence, scRMSD/pLDDT per arm) would only be informative if the order turns out to matter.

## 3. Verdict

| Falsifier | Pre-registered condition | Measured | Fires? |
|---|---|---|:--:|
| **F3** | t*_seq ≤ t*_str + 0.05 under the default schedule, ≥3 seeds / ≥2 lengths | gap 0.041 (7/9 samples ≤ 0.05) | 🟠 **marginally** |
| **F4** | order sweep moves demanding-position recovery < seed-to-seed SD | span 0.012 ≪ SD 0.065 | 🔴 **YES** |

> # Experiment B: DIAGNOSIS directionally right, PRESCRIPTION not supported (on MultiFlow)
>
> The coupling gate passes and MultiFlow **does** commit structure before sequence (t*_str < t*_seq in
> all 9 samples) — the first direct measurement of commitment ordering on a coupled discrete+continuous
> co-design model, and it agrees with §2.3's picture. **But** the separation is small (F3 marginal), and
> **reordering the discrete unmasking is inert** — burial-first, purity, random and anti-burial give
> statistically indistinguishable recovery at demanding positions (F4 fires). So §2.3's *prescription*
> ("decide the highest-influence discrete variables first, while the continuous channel is hot") does
> **not** convert into a measurable benefit here: the commitment window is too narrow for the knob to
> bite.

**Scope of this conclusion.** It is drawn on the **monomer** MultiFlow checkpoint with a **burial**
proxy for demanding positions — not the SKEMPI interface fixture, which the released model cannot
address (monomer-only). A complex-trained coupled co-design model, and interface-hotspot labels, would
be needed to test the prescription where it was originally aimed (binder interfaces). What can be said
cleanly: on this in-distribution monomer test, the ordering intervention BRIEF §2.3 proposes yields no
recovery gain, consistent with the narrow measured commitment window. This does **not** touch
Experiment A's positive result — the conditioning-set tax on predicted backbones — which stands.

Seeds: reference/sweep 0/1/2. Commands recorded in the `command` column of each CSV.
