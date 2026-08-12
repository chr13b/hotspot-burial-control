# Sherlock handoff — archival (purge-rescue) + the KL-triage non-native validation

Paste the block below into the Sherlock Claude Code session (the one with the repo clone and the
`$SCRATCH/expC/` **and** `$SCRATCH/expC2/` data). It (A) rescues the raw artifacts of BOTH Exp C and
Exp C2 off SCRATCH into git-LFS before the ~2026-10-09 purge, and (B) runs the KL-triage method
validation on the non-native (predicted + generative) backbones. Both need the SCRATCH data.

---

```
Two tasks on this Sherlock session; both need the $SCRATCH data. Follow CLAUDE.md (write raw outputs to
results/ as CSV with the exact command; run a positive control; commit + push).

=== TASK A — archive Exp C, Exp C2 AND Exp D raw artifacts via git-LFS (purge-rescue, before ~2026-10-09) ===
Load-bearing artifacts to preserve, for ALL THREE experiments:
  - the per-position SCORED tables (every number derives from them; they CPU-regenerate the analysis),
  - the backbone PDBs (raw output; Exp C 715 RFdiffusion, Exp C2 1705 RFdiffusion, Exp D 141 AF2-multimer),
  - methods-as-code as plain text.
Do NOT commit the RFdiffusion install/weights (third-party license) — pin its commit hash + Complex_base
checkpoint md5 into environment/README.md instead.

Do:
1. Methods-as-code (plain text, tiny): conda list --explicit > environment/se3nv.explicit.txt ;
   pip freeze > environment/se3nv.pip-freeze.txt ; copy the sbatch job files (expC_*.sbatch,
   expC2_*.sbatch) from $SCRATCH into environment/ ; fill the RFdiffusion commit hash + checkpoint md5
   into environment/README.md TODOs.
2. Big artifacts via LFS: git lfs install ; git lfs track "results/*.tar.gz". For EACH of expC, expC2, expD
   (paths differ — confirm each: $SCRATCH/expC, $SCRATCH/expC2, $SCRATCH/ftax/expD; Exp D backbones are the
   AF2 rank_001 PDBs in af2_out):
     tar -czf results/<exp>_backbones.tar.gz -C <that exp's backbones or af2_out dir> .
     cp <that exp's scored_positions.csv> results/<exp>_scored_positions.csv   # matches results/*_positions.csv -> LFS
   Manifests are essentially results/expC_gap_perbackbone.csv, results/expC2_gap_perbackbone.csv, and for
   Exp D results/expD_confidence.csv (per-complex pTM/RMSD) — copy each to results/<exp>_backbone_manifest.csv,
   adding any nan-coordinate exclusions (Exp C had 18 at T40; Exp C2/D reported none).
3. POSITIVE CONTROL before pushing: git lfs ls-files MUST list all six big files
   (expC/expC2/expD × backbones.tar.gz + scored_positions.csv) — else a 100+ MB file would push as a normal
   blob and be rejected. du -h them to sanity-check sizes.
4. Commit + push. This push IS the purge-rescue — confirm it lands on origin/main.
GUARDRAIL: GitHub free LFS storage is 1 GB. Rough total here (expC ~200 MB + expC2 ~400 MB + expD ~150 MB)
approaches the cap — REPORT the running total after push; if it nears 1 GB, the backbones are the priority
to keep (irreplaceable without the GPU env) and the scored tables can be regenerated from them, or add a
data pack.

=== TASK B — validate the KL-triage METHOD on non-native backbones (issue #12) ===
On crystal backbones (src/kl_triage.py, results/kl_triage.csv) ranking interface positions by KL+burial
captures significantly more experimental hotspots than the burial heuristic at a fixed budget (capture@3
0.237 vs 0.139, delta +0.098 [+0.021,+0.175]; capture@25% delta +0.089 [+0.009,+0.169]; n=106). That is a
candidate METHOD contribution but crystal-only. Re-run it on the backbones designers use. THREE non-native
arms are now available — Exp A predicted (OpenFold3), Exp C2 generative, Exp D AF2-multimer ($SCRATCH/ftax/expD)
— run the triage on each (AF2 is the cleanest one-backbone-per-complex arm). Note: C2 and D already showed
the KL DETECTOR (AUROC) generalises across 4 backbone classes; this tests the TRIAGE (capture@k) framing.

POSITIVE CONTROL FIRST: run python3 src/kl_triage.py --joined results/kl_detector_joined.csv and confirm it
reproduces the committed crystal capture@3 delta +0.098 — proves the harness is intact.

Then build, for EACH of {Exp A predicted, Exp C2 generative interface-formed}, a per-position joined table
with the SAME columns as results/kl_detector_joined.csv (complex_id, chain, resnum, icode, is_interface,
is_hot, nbr, kl, logp_native): reuse the backbone-INDEPENDENT experimental labels (is_hot=label=="hot_strict",
is_interface) keyed by (complex_id,chain,resnum) from the crystal table; take kl, logp_native, and nbr
(burial) from that backbone's own scoring ($SCRATCH/expA predicted passes; $SCRATCH/expC2/scored_positions.csv
restricted to interface-formed backbones, aggregated per position). Run:
  python3 src/kl_triage.py --joined <expA table>  --out results/kl_triage_expA.csv
  python3 src/kl_triage.py --joined <expC2 table> --out results/kl_triage_expC2.csv

READING (fixed before seeing numbers): if the capture@k delta (KL+burial - burial) CI still EXCLUDES zero
on the predicted and/or generative backbones, the triage method is design-relevant and enters the paper as
a method; if it collapses to contain zero, report it as a crystal-only property and do NOT upgrade it.
Write a one-line verdict into results/FINDINGS_kl_triage.md §4 (non-native validation) and commit + push
both CSVs.
```

---

## Notes
- Task B is cheap (CPU, seconds once each table is built); the only work is assembling the two joined
  tables from labels already in the repo + the KL passes already on `$SCRATCH`.
- Alternative for Task B: `scp` the two per-position tables (Exp A predicted; `$SCRATCH/expC2/scored_positions.csv`)
  into `results/` on the laptop and it runs locally in seconds — no Sherlock needed for the analysis itself.
- C2's own KL result already used the canonical `label=="hot_strict"` set (`src/expC2_kl_loose.py`); reuse
  that keying for the triage table so the definitions match.
