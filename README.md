# hotspot-burial-control

**Are inverse-folding models "blind" at protein–protein interface hotspots? We pre-registered the
test, ran it across five architectures, and the answer is: not where the field looked.**

This began as *the factorization tax* — the hypothesis that staged backbone→sequence design cannot
reach interface hotspots because the model's mode is the wrong residue there, so the constellation
cost `N_hot = exp(Σδ_i/T)` explodes. We pre-registered falsifiers (BRIEF.md §4), then measured. The
central hypothesis was **refuted on the bound backbone** — and the measurement located where the
effect actually lives.

## What we found

**Read [`results/FINDINGS.md`](results/FINDINGS.md) for the full account with every number traced to
a CSV.** The short version:

**Negative core — three proposed mechanisms, each measured, none survives on a fixed backbone:**
- **No burial-matched hotspot recovery penalty.** Matched within complex on rSASA/SS(pydssp)/packing,
  across **five architectures** (ProteinMPNN vanilla+soluble, ESM-IF1 142M, PiFold one-shot, MIF) —
  every PRIMARY CI contains zero. The raw ProBID-Net-style gap is a **burial artifact**, and it runs
  in the direction that *flatters* hotspots. (`F0` fires.)
- **`N_hot ≈ 10^10` is a generic property of T = 0.1 sampling**, statistically indistinguishable at
  burial-matched *control* constellations (median Δ = 0.000 log10, p = 0.90) — not a hotspot tax.
- **Commitment ordering is inert.** Deciding true hotspots first changes hotspot recovery by −0.003
  (p = 0.89) — a pre-registered null that refines the schedule mechanism to coupled models only.

**Positive core — the tax is real, but it lives in the conditioning set, not in hotspot chemistry:**
- **The interaction.** Hotspots gain **+0.27 nats [+0.15, +0.41]** more from the partner's presence
  than matched controls do (ΔΔrSASA-adjusted) — the frustration signature, measured and externally
  validated against experimental ΔΔG_bind (ρ = +0.28, adds beyond burial + log-odds).
- **A sequence-free signal.** `KL(p(·|complex backbone) ‖ p(·|monomer backbone))`, computed with **no
  residue identity**, carries the same hotspot information as the sequence-aware statistic
  (removing the sequence costs nothing: ΔAUROC +0.001 [−0.020, +0.023]) and is not a contact-count
  proxy. A diagnostic about what backbone conditioning encodes — not yet a deployable detector.

## Layout

| File | What it is |
|---|---|
| **[results/FINDINGS.md](results/FINDINGS.md)** | The paper-in-progress. Every claim, every number, every correction, traced to a CSV. |
| **[results/PREREG.md](results/PREREG.md)** | Analysis choices fixed before any number was computed. |
| **[BRIEF.md](BRIEF.md)** | Original self-contained brief (the hypothesis, since refined). |
| **[notes/SHERLOCK_HANDOFF.md](notes/SHERLOCK_HANDOFF.md)** | Paste-ready prompt for the two GPU experiments (predicted-backbone transfer + coupled-model ordering). |
| `src/` | One script per analysis; each takes `--out`, writes a CSV with its exact command, prints a one-line summary. |
| `src/models/`, `src/decoding/` | Five-model panel wrappers; the tested ProteinMPNN steering layer. |
| `results/*.csv` | All raw outputs. Large position tables via git-lfs. |

## Reproduce

```bash
git lfs install && git lfs pull
python3 src/validate.py                        # positive-control gate — every path
python3 src/p0_burial_matched.py --out results/p0
python3 src/patch_ss.py --positions results/p0_positions.csv    # pydssp SS
python3 src/p0_burial_matched.py --out results/p0_dssp --cache results/p0_positions.csv
python3 src/regression_estimator.py --out results/regression   # higher-powered F0
python3 src/p1_nhot.py --out results/p1
python3 src/nhot_control.py --out results/nhot_control          # the control that decides what N_hot means
python3 src/frustration_monomer.py --out results/frustration_monomer
python3 src/hardening.py --out results/hardening                # TOST, Holm, external validation
python3 src/kl_detector.py --out results/kl_detector            # sequence-free detector
python3 src/kl_analysis.py --out results/kl_analysis            # paired AUROC, contact baseline, head-to-head
python3 src/p0_multimodel.py --models pifold,mpnn_soluble,mif,esmif --out results/panel
python3 src/junction_sensitivity.py --out results/junction_sensitivity
python3 src/decoding/p2_ordering.py --stage 1 --Lmax 400 --K 100 --out results/p2_ordering
```

## Status

Phases 0 and 1 complete on CPU across five architectures; all falsifiers evaluated; write-up in
`results/FINDINGS.md`. **Next: two GPU experiments on Sherlock** (`notes/SHERLOCK_HANDOFF.md`) —
predicted-backbone transfer (validity of the KL signal) and commitment ordering on a coupled
co-design model with a binding readout (the path from a TMLR-grade to an ICLR-grade contribution).

Data: SKEMPI 2.0 + its PDB bundle (downloaded, not committed — see FINDINGS for URLs). Models:
public ProteinMPNN / ESM-IF1 / PiFold / MIF checkpoints.
