# Sherlock handoff — CFG-steering α-sweep (paste-to-run)

**Why this is on Sherlock.** The paper *names* the mixed derivative `L` as the classifier-free-guidance
(CFG) direction. This run operationalizes it: tilt a **frozen off-the-shelf ProteinMPNN**'s sampling by
`+α·L` at interface positions and ask whether the sampled sequences bind better — measured by a **different,
independent model (ESM-IF1)**, so the test is anti-circular. The pre-registration is already committed:
`results/PREREG_cfg_steer.md` (do not edit it). The local CPU run **OOM-died at complex 1/18** (K=40
autoregressive sampling on a 4 GB box); on a GPU this is seconds/complex. The **smoke test already passed
locally** (steering by L raised the independent ESM-IF1 leverage of the sampled residues −0.09 → +0.21 while a
random direction of matched magnitude lowered it) — Sherlock's job is to reproduce that signature as a
positive control and then run the full, complex-clustered α-sweep.

This does **not** decide the venue — it is a ceiling-raiser that closes the diagnosis → direction →
intervention arc (RedNet operationalizes L in a *trained* decoder; nobody has shown that tilting a *frozen,
off-the-shelf* model by the *measured* L produces sequences a *second* model scores as higher-binding).

---

## 0. Environment (same as prior leverage runs)
- Repo: `github.com/chr13b/hotspot-burial-control`. `git pull` on `main` (HEAD ≥ `af9568e`).
- Data already on the node from prior runs: SKEMPI PDBs at `~/ftax/data/PDBs/`, vanilla ProteinMPNN weights at
  `~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt`. Confirm both exist before starting.
- Python env: the same one used for `leverage_decomposition.py` / `leverage_esmif.py` (torch + the committed
  ProteinMPNN + ESM-IF1 scorers). No new deps.

## 1. Inputs — the two pq caches are NOT in git (regenerate or reuse)
`src/cfg_steer.py` reads the per-position P and Q distributions from two large caches that are **git-ignored**:
- `results/leverage_pq_skempi.csv`  (ProteinMPNN L direction — the steering bias)
- `results/leverage_pq_skempi_esmif.csv`  (ESM-IF1 L — the independent headline metric)

Get them one of two ways:
- **Reuse** if present on `$SCRATCH` from the earlier leverage runs — copy/symlink into `results/`; **or**
- **Regenerate** deterministically from the committed scorers (fast on GPU):
  ```bash
  python3 src/leverage_decomposition.py            # writes results/leverage_pq_skempi.csv (the MPNN pq cache)
  python3 src/leverage_esmif.py                    # writes results/leverage_pq_skempi_esmif.csv (--cache default)
  ```
  These are the same deterministic outputs already used in the paper; the committed
  `results/leverage_skempi_positions.csv` (the interface set) IS in git and needs nothing.

**Sanity gate before proceeding:** both pq files must have finite `lP_*` / `lQ_*` columns for the shared
complexes. `cfg_steer.py` intersects the MPNN pq, the ESM-IF1 pq, and the interface set, then drops complexes
with `n>700` or `<3` usable interface positions — expect ~15–18 complexes. If the intersection is empty, the
pq caches are misaligned; stop and report.

## 2. Device port (minimal, preserve the CPU path)
`src/cfg_steer.py` sets `torch.set_num_threads(4)` and runs `mpnn_steer.draw(...)` on CPU. Port to GPU with the
smallest possible change and **keep the laptop/CPU path working** (this is the established rule — the port must
be device-only):
- pick `device = "cuda" if torch.cuda.is_available() else "cpu"`;
- move the ProteinMPNN model to `device` after `fc.load_mpnn(...)`;
- inside `src/decoding/mpnn_steer.py::draw`, ensure the `bias_by_res` tensor `B` and any per-call tensors are
  created on / moved to the model's device before they are added to the logits.
Do **not** change the math, the seed (`SEED=20260803`), the α handling, or the metric. Commit the device port
as its own small commit so the diff is auditable.

## 3. Positive control FIRST (CLAUDE.md rule 6 — do not skip)
```bash
python3 src/cfg_steer.py --limit 3 --K 16 --out results/_smoke_cfg.csv
```
**Required signature to trust the rig** (reproduce the local smoke qualitatively): in the printed
`=== summary ===`, `Lesmif` for `direction=L` must **rise with α** above the α=0 baseline, and `Lesmif` for
`direction=random` must **not** rise (flat or down). If the L arm does not beat the random arm, the rig is
broken or the pq caches are wrong — **stop and report**, do not run the full sweep.

## 4. The full sweep — use the PRE-REGISTERED α grid, not the script default
The script's *default* is `--alphas 0,0.5,1,2,4`, but `results/PREREG_cfg_steer.md` pre-registered
**α ∈ {0, 0.25, 0.5, 1.0, 2.0}**. The pre-registration wins (rule 1). Run:
```bash
python3 src/cfg_steer.py --alphas 0,0.25,0.5,1,2 --K 64 --out results/cfg_steer.csv
```
(`--K 64` per the pre-reg "K samples"; raise only if wall-clock is trivial and note it.) Both arms (`L` and
`random`) run automatically.

## 5. Deliverables (commit + push, then message me)
1. `results/cfg_steer.csv` (per complex × direction × α: `int_recovery`, `noninterface_recovery`,
   `meanL_mpnn`, `meanL_esmif`, `n_int`).
2. `results/FINDINGS_cfg_steer.md`: the α-curve of **ESM-IF1 leverage** for the L arm vs the random control,
   with **complex-clustered bootstrap 95% CIs** (nan-aware; disclose any dropped resamples), the interface
   native-recovery cost per α, the **sweet-spot α** (ESM-IF1 leverage meaningfully up while interface recovery
   ≥ ~50% of the α=0 baseline), the non-interface-recovery localization control (should be ~flat), `n`
   complexes, `SEED=20260803`, and the exact command.
3. **Report the falsifier outcome verbatim** if it fires: "ESM-IF1 leverage does not rise with α, or rises no
   more than the random control → the CFG direction is not actionable." An honest null here is a valid,
   publishable result (it would bound the framing to a diagnostic) — do **not** massage the grid or K to
   rescue it.
4. Commit messages end with the two trailer lines used across this repo
   (`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` / `Claude-Session: …`). Push to `main`.

## 6. What NOT to do
- Do not commit the two 11 MB pq caches to git (regenerate/reuse per §1; they stay git-ignored).
- Do not change `results/PREREG_cfg_steer.md` or the pre-registered α grid / metric after seeing any number.
- Do not measure success by MPNN's own L (that is the by-construction check only); the **headline is ESM-IF1**.
