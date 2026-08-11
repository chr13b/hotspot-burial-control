# Sherlock handoff — archival (purge-rescue) + the KL-triage non-native validation

Paste the block below into the Sherlock Claude Code session (the one with the repo clone and the
`$SCRATCH/expC/` data). It does two things that both need the SCRATCH data: (A) rescue the raw artifacts
off SCRATCH into git-LFS before the ~2026-10-09 purge, and (B) run the KL-triage method validation on the
non-native (predicted/generative) backbones. Run it once C2's generation is underway or finished.

---

```
Two housekeeping+analysis tasks on this Sherlock session; both need the $SCRATCH data. Follow CLAUDE.md
(write raw outputs to results/ as CSV with the exact command; run a positive control; commit + push).

=== TASK A — archive the raw artifacts via git-LFS (purge-rescue, before the ~2026-10-09 SCRATCH purge) ===
Load-bearing artifacts to preserve (decided): (1) the Exp C per-position SCORED table (~122 MB — every
Exp C number derives from it; it CPU-regenerates the whole analysis with no GPU); (2) the 715 Exp C
backbone PDBs (~78 MB — the raw experimental output; needed to inspect the divergence/dissolution cases
and to re-score/re-key). Do NOT commit the RFdiffusion install or weights (third-party license) — instead
pin its exact commit hash + the Complex_base checkpoint md5 into environment/README.md.

Do:
1. Methods-as-code (plain text, tiny): `conda list --explicit > environment/se3nv.explicit.txt`;
   `pip freeze > environment/se3nv.pip-freeze.txt`; copy the 2-3 sbatch job files from $SCRATCH into
   environment/. Fill the RFdiffusion commit hash + checkpoint md5 into the environment/README.md TODOs.
2. Big artifacts via LFS: `git lfs install`; `git lfs track "results/*.tar.gz"`; tar the backbones to
   results/expC_backbones.tar.gz; copy the scored table to results/expC_scored_positions.csv (this name
   matches the already-committed LFS pattern results/*_positions.csv). Build the manifest
   results/expC_backbone_manifest.csv from results/expC_gap_perbackbone.csv (backbone_id, complex_id,
   partial_T, irmsd, interface_ok) plus the 18 listed partial_T=40 nan-coordinate exclusions, labelled
   formed / dissolved / nan.
3. POSITIVE CONTROL before pushing: `git lfs ls-files` MUST list BOTH results/expC_backbones.tar.gz and
   results/expC_scored_positions.csv (else the 122 MB file would try to push as a normal blob and hit
   GitHub's 100 MB limit). Also `du -h` both to sanity-check sizes.
4. Commit + push. This push IS the purge-rescue — confirm it lands on origin/main.
GUARDRAIL: GitHub free LFS storage is 1 GB. Exp C is ~200 MB; when you archive C2's backbones the same
way, watch the total — if near 1 GB, keep only the scored tables (regenerable from backbones) or note
that a data pack is needed. Report the running LFS total.

=== TASK B — validate the KL-triage METHOD on non-native backbones (task #12) ===
On crystal backbones we showed (src/kl_triage.py, results/kl_triage.csv) that ranking interface positions
by KL+burial captures significantly more experimental hotspots than the burial heuristic at a fixed
budget (capture@3 0.237 vs 0.139, delta +0.098 [+0.021,+0.175]; capture@25% delta +0.089 [+0.009,+0.169];
n=106). That is a candidate METHOD contribution, but it is crystal-only. Re-run it on the backbones
designers actually use.

PRIMARY — Exp A PREDICTED backbones (the clean one-backbone-per-complex analog of crystal):
1. Assemble a per-position joined table with the SAME schema as results/kl_detector_joined.csv:
   columns complex_id, chain, resnum, icode, is_interface, is_hot, nbr, kl, logp_native. Reuse the
   backbone-INDEPENDENT experimental labels (is_hot, is_interface) keyed by (complex_id, chain, resnum)
   from the crystal table; take kl, logp_native, and nbr (burial) from the Exp A PREDICTED-backbone
   scoring on $SCRATCH (the same KL passes that produced expA_kl_summary.csv, but per-position) — i.e.
   design-time burial and KL computed on the predicted backbone.
2. Run: python3 src/kl_triage.py --joined <that table> --out results/kl_triage_expA.csv
3. POSITIVE CONTROL: point kl_triage.py at results/kl_detector_joined.csv first and confirm it reproduces
   the committed crystal capture@3 delta +0.098 — proves the harness is intact before trusting the new table.

SECONDARY (optional) — Exp C GENERATIVE backbones: build the same joined table from
$SCRATCH/expC/scored_positions.csv restricted to interface-FORMED backbones (aggregate KL/burial per
(complex, position) across the formed backbones, e.g. per noise level), run
kl_triage.py --out results/kl_triage_expC.csv.

READING (fixed before seeing the number): if the predicted-backbone capture@k delta (KL+burial - burial)
CI still EXCLUDES zero, the triage method is design-relevant and enters the paper as a method; if it
collapses to contain zero, report it as a crystal-only property and do NOT upgrade it. Write a one-line
verdict into results/FINDINGS_kl_triage.md §4 (non-native validation) and commit + push both CSVs.
```

---

## Notes
- Task B is cheap (CPU, seconds once the table is built) — the only work is assembling the predicted
  per-position table, which reuses the crystal labels + the Exp A KL passes already on `$SCRATCH`.
- If you would rather I run Task B locally: `scp` the two per-position tables (Exp A predicted joined, and
  optionally `$SCRATCH/expC/scored_positions.csv`) into `results/` on the laptop, and it runs here in
  seconds — no Sherlock needed for the analysis itself, only for producing the tables.
