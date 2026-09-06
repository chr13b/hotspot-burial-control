# Python package plan — `leverage` (post-submission; reproducible code + Zenodo for the deadline)

## Why
Backs the paper's *actionable/deployable* framing, and the idea's value is **adoption** — it's an elegant,
easily-reproduced technique, so open tooling maximises impact (low defensibility ⇒ open science beats secrecy).

## Scope & timeline (do NOT let this eat the 20-day runway)
- **Deadline (25 Sept):** ship the **reproducible code + a Zenodo DOI** (anonymised mirror for double-blind).
  This is the packaging step already on the rigor roadmap — NOT the polished package.
- **Camera-ready / post-acceptance:** the polished **PyPI** package, built from that code.

## Design — model-agnostic CORE + validated ADAPTERS
- `leverage/core.py` — the partner-ablation mixed derivative
  `L_i(a) = [logP(a|complex) − logP(wt|complex)] − [logP(a|monomer) − logP(wt|monomer)]`; scalar reductions
  (`L_rms`, `L(→A)`); the CFG-guided logit `logit(·|complex) + α·[logit(·|complex) − logit(·|monomer)]`.
- `leverage/rank.py` — training-free hotspot ranking (|L| + geometry) and the CPI helper.
- `leverage/steer.py` — the frozen-model `+α·L` interface-logit tilt + sampling.
- `leverage/models/` — a `Model` **protocol** (`score(seq | structure)` for complex AND monomer, given a
  structure + interface set) with **validated adapters** `mpnn.py`, `esmif.py` (the benchmarked models). Others
  work via the protocol; we **recommend only the benchmarked adapters** (no untested claims — avoids the
  "benchmark everything" trap).
- `leverage/io.py` — structure parsing (biotite/biopython), interface set (ΔrSASA > 0.05), partner ablation
  (delete the partner's atoms → the monomer conditioning).

## Interface (sketch)
```python
from leverage import Model, leverage, steer, rank
m = Model.mpnn()                                   # or Model.esmif()
L = leverage(m, structure, interface)             # per-residue mixed derivative
triage = rank.hotspots(L, geometry)               # training-free ranking
seqs = steer(m, structure, interface, alpha=2.0, k=64)   # +αL frozen-model design
```

## Deliverables
- Deadline: `src/` scripts (already committed) + a `README` mapping every claim → CSV + `results/INDEX.md`;
  Zenodo DOI (before ~Oct 9 SCRATCH purge); anonymised code mirror (anonymous.4open.science) for submission.
- Post: the pip-installable `leverage` package + docs.

## Non-goals
- Not a design platform (this is a technique/library). Not claiming untested models (protocol-open,
  adapter-validated). Related: [[PIPELINE_POCKET]] (company vs open-source stance).
