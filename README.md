# factorization-tax

Can the staged **backbone → sequence** pipeline reach an interface hotspot at all?

Inverse-folding models maximise `p(sequence | backbone)` with **no binding-energy term**. Hotspots
are frequently *frustrated* — buried polars, strained rotamers, entropically expensive aromatics
that buy affinity rather than stability. So at exactly the positions that make a binder a binder,
the model's mode is the wrong residue and the right one is in the tail:

```
N_hot = exp( Σ_i δ_i / T )
```

If `N_hot` is 10⁴–10⁸, no amount of oversampling recovers native-grade interfaces, and **every
downstream filter is sorting among designs that never contained the chemistry.**

**Phase 0 is inference-only and runs on CPU.** No GPU, no training.

## Start here

| File | What it is |
|---|---|
| **[BRIEF.md](BRIEF.md)** | Full brief — mechanism, the published result this builds on, the published choice it contradicts, three phases, five falsifiers, six pitfalls. Self-contained. |
| **[HANDOFF.md](HANDOFF.md)** | Paste-ready prompt for a fresh session. |
| **[CLAUDE.md](CLAUDE.md)** | Project rules and machine constraints. |
| `notes/lineage-provenance.md` | Where this came from, across three ideation passes. |

## Quickstart

```bash
cd /mnt/c/Users/chris/Desktop/python_projects/personal_projects/factorization-tax
claude          # then paste the block from HANDOFF.md
```

## Phases

| Phase | Needs | Cost |
|---|---|---|
| **0 — the burial-matched control** *decisive* | Laptop, SKEMPI + ProteinMPNN | **hours** |
| **1 — `N_hot`, the constellation cost** | Laptop, CPU | hours |
| **2 — commitment ordering** | **CUDA GPU → Sherlock** | only if Phase 0 passes |

## The one number this project exists to correct

**ProBID-Net** (*Chem. Sci.* 2024) published sequence recovery of **0.334 at hotspots vs 0.472 at
non-hotspots** — and **did not control for burial.** They attributed the gap to conformational
dynamics.

Buried positions are where inverse folding is **most confident**, so an uncontrolled comparison
*hides* the effect rather than inventing it. The burial-matched control is both this project's
strongest contribution and its cheapest experiment.

## Pre-registered falsifiers

Fixed before any data was touched. They do not move.

- **F0** — burial-matched hotspot-minus-control gap CI contains zero. *ProBID-Net's headline is a
  burial artifact.* **Publishable either way — the main reason to run this first.**
- **F1** — burial-controlled partial Spearman between log-odds and ΔΔG_bind ≥ 0.35. *The model is
  not blind to binding energy.*
- **F2** — median log₁₀ `N_hot` < 2 and the matched gap CI contains zero. *The factorisation is not
  costly where claimed.*
- **F3** — `t*_seq ≤ t*_str + 0.05`. *Joint models already decide sequence first; diagnosis wrong.*
- **F4** — the discrete-rate sweep moves hotspot-restricted recovery by less than seed-to-seed SD.
  *The knob is inert.*

## It contradicts a published choice

MultiFlow uses confidence-ordered ("purity") unmasking and **reports it as beneficial.** This work
predicts it is harmful at hotspots, because it unmasks easy stability-determined positions first and
leaves the frustrated ones to be decided last against a backbone that can no longer move. ML venues
reward overturning a specific choice with a measurement — and the burden of proof is ours.

## Status

Not started. **Step 0 is to fetch bioRxiv `10.64898/2026.05.09.722041`** ("Redesign selective protein
binders using contrastive decoding") — a prior sweep read metadata only, and it may contain this
analysis.
