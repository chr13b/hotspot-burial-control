# CLAUDE.md — project instructions

## What this project is

Testing one claim: the staged **backbone → sequence** pipeline cannot reach interface hotspots,
because inverse-folding models optimise `p(sequence | backbone)` with **no binding-energy term** and
hotspots are frequently *frustrated* — so the right residue sits in the tail.

Read [`BRIEF.md`](BRIEF.md) first. It is self-contained: mechanism, the published result this
builds on and the published choice it contradicts, three phases, every pre-registered falsifier, and
the six ways to get a false positive.

## Ground rules

1. **Falsifiers are pre-registered (BRIEF.md §4). Do not move one after seeing a number.**
2. **An honest null is a valid outcome — and here it is a paper in both directions.** If the
   burial-matched gap collapses, that means ProBID-Net's published 0.334/0.472 is a burial artifact,
   which is a real correction to the literature. Report it as a finding, not a failure.
3. **Never fabricate, extrapolate or simulate a measurement.** If something cannot be run, say
   exactly why and stop.
4. **Write raw outputs to `results/` as CSV** with the exact command and config. Every number in a
   write-up must trace to a committed CSV.
5. **Citation discipline.** Cite only URLs actually fetched and resolving. Never cite from memory;
   write "unverified: could not fetch" rather than treating something as cleared.
6. **Run positive controls through every search, query or filter path before trusting a zero.** Two
   agents in this project's lineage were saved by exactly this after a silently-broken full-text
   operator produced four false negatives.

## Machine constraints (measured)

- AMD Ryzen 7 5800H, 8 cores / 16 threads.
- **RAM 7.5 GB total, ~4 GB free.** Fine here — SKEMPI is ~7k rows over ~345 structures.
- **No CUDA GPU** (Radeon integrated, `nvidia-smi` absent). **Phases 0 and 1 need none** —
  ProteinMPNN is ~1.7M parameters and CPU inference over 345 complexes is minutes. Phase 2
  (MultiFlow sampling sweeps) requires a GPU and goes to Sherlock.
- Disk: ~198 GB free on the Linux root, ~347 GB on `/mnt/c`. Prefer the Linux root — WSL2
  cross-filesystem IO is slow.
- Installed and verified: python 3.8.10, numpy 1.23.5, scipy 1.10.0, pandas 1.5.3,
  matplotlib 3.6.0, networkx 3.0, torch 2.0.1 (CPU), rdkit 2024.03.5.
- Likely needed and not present: `biopython` (PDB parsing, SASA), `freesasa` or DSSP for relative
  SASA and secondary structure. Check before assuming.

## Order of work

1. **Fetch the contrastive-decoding preprint first** (bioRxiv `10.64898/2026.05.09.722041`,
   *"Redesign selective protein binders using contrastive decoding"*). The sweep read metadata only.
   **If it already contains the burial-matched analysis or the commitment-ordering result, stop and
   report before writing code.**
2. **Phase 0 — the burial-matched control.** This is the whole experiment. Get the matching right
   before computing anything downstream: same complex, rSASA ±0.05, same secondary-structure class,
   neighbour count ±1, paired differences, complex-level bootstrap.
3. **The frustration-versus-dynamics discrimination in the same pass.** ProBID-Net attributes the
   gap to dynamics; that is the alternative hypothesis and it is cheap to test.
4. **Phase 1 — `N_hot`.** Both the direct sampled measurement and the analytic product, with the
   discrepancy reported rather than assumed away.
5. **Stop and report. Do not start Phase 2** without an explicit decision — it needs a GPU this
   machine does not have, and it depends on an unverified assumption about MultiFlow's released code.

## The three things most likely to produce a false positive

Full list in BRIEF.md §5.

1. **Burial, and it cuts against naive expectation.** Buried positions are where inverse folding is
   *most confident*, so an uncontrolled comparison **hides** the effect rather than inventing it.
   Matched pairs within the same complex, always. This is the analogue of the molecular-size confound
   from this project's lineage.
2. **Decoding-order variance.** ProteinMPNN's autoregressive order changes conditional
   probabilities. Average over ≥8 orders and report the spread; a result inside decoding-order
   variance is not a result.
3. **Positional independence in `N_hot`.** The analytic product assumes it and it is false. Measure
   directly where possible and report the gap.

## Style

- Plain numpy / scipy / pandas plus whatever is needed to parse structures. No frameworks.
- One self-contained script per phase: `src/p0_burial_matched.py`, `src/p1_nhot.py`.
- Every script takes `--out`, writes a CSV, and prints a one-line summary.
- Fix and record every seed. Report bootstrap replicate counts alongside every CI.
