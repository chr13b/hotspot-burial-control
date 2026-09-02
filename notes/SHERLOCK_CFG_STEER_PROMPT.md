# Sherlock task — unblock the repo, then run the pre-registered CFG-steering sweep

**Paste this whole file to the Sherlock session.** You are on a Sherlock login node with a repo whose `git pull`
is failing on two things at once: (1) `git-lfs` is not installed, so the LFS smudge filter aborts the checkout;
(2) the working tree has stale local edits and untracked files that collide with origin. Fix both (Phase 0),
regenerate the inputs the sweep needs (Phase 1), then run the sweep (Phases 3–4). **Report honestly; a null is a
valid result.** `SEED=20260803` throughout. Repo path in your session:
`/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`.

Goal of the science: the paper names the mixed derivative `L` as the classifier-free-guidance direction. This
sweep tilts a **frozen off-the-shelf ProteinMPNN** by `+α·L` at interface positions and measures whether the
sampled sequences bind better **according to a different, independent model (ESM-IF1)** — an anti-circular test.
Pre-registration (already committed, do not edit): `results/PREREG_cfg_steer.md`.

---

## Phase 0 — unblock the pull (run on the login node; this part is light)

```bash
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control

# (a) git-lfs is missing and the sweep needs no LFS file's real content (the one LFS input it uses,
#     results/leverage_skempi_positions.csv, is regenerated in Phase 1). Make git treat LFS as passthrough
#     so the smudge filter stops aborting the checkout (repo-local config, no git-lfs needed):
git config filter.lfs.smudge cat
git config filter.lfs.clean cat
git config filter.lfs.process cat
git config filter.lfs.required false

# (b) Preserve ALL stale local state (edited notes/figures + colliding untracked files) in a recoverable
#     stash — nothing is destroyed, and the tree is cleared so the pull can fast-forward. Origin (fbfed82)
#     holds the canonical versions of every one of those files.
git stash push --include-untracked -m "sherlock-stale-$(date +%Y%m%d-%H%M%S)"

# (c) Pull:
git pull --no-edit origin main
git rev-parse --short HEAD          # expect fbfed82 or newer
git log --oneline -3
```

If `git stash --include-untracked` itself errors, use this discard-based fallback instead (every listed file is
either regenerable or canonical on origin, so discarding the local copies is safe):
```bash
git checkout -- .                                          # drop stale tracked edits (notes, figures)
git clean -fd notes/ results/figures/                      # remove colliding untracked notes/figures
rm -f results/leverage_bennett_denovo.csv results/leverage_mif.csv results/leverage_mif_mutations.csv
git pull --no-edit origin main
```

---

## Get a compute node before Phase 1 (do NOT score/sample on the login node)

Request the **same node type you used for the R2 / ESM-IF1-dose runs** (a CPU node with ≥16 GB RAM is enough —
the laptop failure was a 4 GB OOM, not a compute limit; a GPU node is faster but optional). For example:
```bash
srun -c 4 --mem=16G -t 3:00:00 --pty bash        # (or your usual GPU request: --partition gpu --gres=gpu:1)
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
```
With real RAM, the committed `src/cfg_steer.py` runs **as-is on CPU** — no code change needed. (Only if you want
it faster on a GPU, apply the minimal CUDA port in the Appendix; it is optional.)

## Phase 1 — regenerate the three inputs the sweep reads

`src/cfg_steer.py` reads three files that the pull cannot give you with real content: the two pq caches
(`results/leverage_pq_skempi.csv`, `results/leverage_pq_skempi_esmif.csv`) are git-ignored, and
`results/leverage_skempi_positions.csv` is an LFS pointer after the bypass. All three are **deterministic outputs
of the committed scorers** — regenerate them. (If real-content copies already sit on `$SCRATCH` from a prior run,
copy them into `results/` and skip this phase.)

```bash
python3 src/leverage_decomposition.py     # writes leverage_pq_skempi.csv AND leverage_skempi_positions.csv (real)
python3 src/leverage_esmif.py             # writes leverage_pq_skempi_esmif.csv (its --cache default)

# sanity gate — the positions file must be REAL CSV, not a ~130-byte LFS pointer:
head -c 60 results/leverage_skempi_positions.csv          # must be a CSV header, NOT "version https://git-lfs…"
wc -l results/leverage_pq_skempi.csv results/leverage_pq_skempi_esmif.csv results/leverage_skempi_positions.csv
```
If the positions file still starts with `version https://git-lfs`, the scorer did not overwrite it — check the
scorer ran to completion; do not proceed until all three are real.

## Phase 3 — positive control FIRST (do not skip; CLAUDE.md rule 6)

```bash
python3 src/cfg_steer.py --limit 3 --K 16 --out results/_smoke_cfg.csv
```
In the printed `=== summary ===`, the **required signature** to trust the rig: for `direction=L`, `Lesmif` must
**rise with α** above the α=0 value; for `direction=random`, `Lesmif` must **not** rise (flat or down). If the
L arm does not beat the random arm, the rig or the inputs are wrong — **STOP and report**, do not run the sweep.

## Phase 4 — the full sweep on the PRE-REGISTERED α grid

The script's *default* is `--alphas 0,0.5,1,2,4`, but the pre-registration fixed **α ∈ {0, 0.25, 0.5, 1, 2}**.
The pre-registration wins.
```bash
python3 src/cfg_steer.py --alphas 0,0.25,0.5,1,2 --K 64 --out results/cfg_steer.csv
```
Both the `L` and `random` arms run automatically. This is the deliverable.

## Phase 5 — OPTIONAL (only if Phase 4 is done and time allows): place predicted backbones on the dose curve

For a separate paper connection, we want the Cα-RMSD-to-native of the OpenFold3 / AF2 predicted backbones used
in the R2 result, so those points can be plotted at their true effective σ on the dose-law x-axis. If the
predicted PDBs are on `$SCRATCH` (see `results/expD_backbone_manifest.csv` for the paths), compute per-complex
Cα-RMSD(predicted, crystal) after backbone-only superposition and write
`results/predicted_backbone_rmsd.csv` with columns `complex_id, of3_ca_rmsd, af2_ca_rmsd`. Skip if the PDBs
aren't readily available — this is secondary to the sweep.

## Phase 6 — deliverables

```bash
# Commit ONLY the small named outputs. Do NOT `git add -A` — that would stage the regenerated LFS/pq/positions
# files under the bypassed filter and corrupt LFS tracking. The two files below are not LFS-matched, so the
# bypass does not affect this commit.
git add results/cfg_steer.csv results/FINDINGS_cfg_steer.md
# (add results/predicted_backbone_rmsd.csv too, if you did Phase 5; and the device-port diff, if you ported)
git commit -m "CFG-steering sweep: <one-line result>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GANjj26GayPau9zeWCmVSG"
git push origin main
```

Write `results/FINDINGS_cfg_steer.md`: the α-curve of **ESM-IF1 leverage** (the independent headline) for the
`L` arm vs the `random` control, with **complex-clustered bootstrap 95% CIs** (nan-aware; disclose dropped
resamples); the interface native-recovery cost per α; the **sweet-spot α** (ESM-IF1 leverage meaningfully up
while interface recovery ≥ ~50% of the α=0 baseline); the non-interface-recovery localization control (should
be ~flat); `n` complexes; `SEED`; the exact command.

## Guardrails
- **Headline = ESM-IF1 leverage**, never MPNN's own L (that is only the by-construction check that the tilt fired).
- Use the pre-registered α grid; do not change it or `K` after seeing any number.
- **Report the falsifier verbatim if it fires** ("ESM-IF1 leverage does not rise with α, or no more than the
  random control → the CFG direction is not actionable"). An honest null is publishable — do not massage it.
- Then message me the one-line result so I can fold it in.

## Appendix — OPTIONAL minimal CUDA port (only if you want GPU speed)
In `src/cfg_steer.py`: after `model, _ = fc.load_mpnn(LD.MPNN_W)`, add
`device = "cuda" if torch.cuda.is_available() else "cpu"; model = model.to(device)`. In
`src/decoding/mpnn_steer.py::draw`, move the `bias_by_res` tensor and the featurized inputs to the model's
device before they enter the network. Change nothing else (not the math, not the seed). Commit the port as its
own small diff so it is auditable, and keep the CPU path working (`device` falls back to `"cpu"`).
