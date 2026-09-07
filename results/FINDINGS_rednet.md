# FINDINGS — RedNet head-to-head: measurement BLOCKED, positioned at the code level (mechanism-identical)

**The head-to-head against RedNet (`zw2x/rednet_public`) is measurement-blocked and NOT faked** (CLAUDE.md
rule 3). What the recon *could* establish — from RedNet's released source, fetched and read directly — is a
code-level positioning result: **RedNet's retrained selective-binder method converges on our exact `+α·L`
direction.** This note records the mechanism identity (with line refs), RedNet's genuine additions, and the
blocker, so the record shows the comparison is positioned at the code level and never simulated.

Provenance: `github.com/zw2x/rednet_public` @ `beb04d4` (Apache-2.0), shallow-cloned to `$SCRATCH/ftax/rednet_src`
on 2026-09-06; weights record `zenodo.org/records/20113403`. Pre-reg context: `results/PREREG_naive.md`.

## 1. Mechanism identity — RedNet's contrastive decode IS our `+α·L`
`src/rednet/sampling_utils.py::sample_tokens` (lines 25–26):
```python
if alpha > 0:
    logits_t = (1 + alpha) * logits_t - alpha * con_logits_t
```
Rearranged: `logits = logP_complex + α·(logP_complex − logP_contrast)`. The tilt term `α·(logP_complex −
logP_contrast)` **is our `+α·L`** — `L(a) = (logP_complex(a) − logP_complex(wt)) − (logQ(a) − logQ(wt))` (the
per-position `−(wt)` offsets are softmax-invariant constants). The α is the same knob; a released config sets it
explicitly (`configs/sampling/cd_a2_b07_t0001.yaml`: `alpha: 2`, `beta: 0.7`, `temperature: 0.001`; "cd" =
contrastive decode). `SamplingConfig{temperature, alpha, beta}` at `cli/infer_pipeline.py:168–184`.

**What plays the role of our monomer/partner-ablated `Q`** (the contrast distribution `con_logits_t`):
- `run_hdimer` (heterodimer self-consistency): `con_batch = ubd_batch = _crop_batch(batch["dsn_mask"], batch)` —
  the **unbound monomer** (`cli/infer_pipeline.py:444`, used at `:458–459`). This is exactly our apo/partner-ablated
  contrast `Q`.
- `run_sel` (selective binder design): `con_batch = off_batch` — the **off-target** complex
  (`cli/infer_pipeline.py:282`, used at `:303–305`).

And RedNet's *scorer* computes the same contrastive quantity we do:
`cli/infer_pipeline.py:654` → `"cd_ll": _res['ll'] - _res['ub_ll']` (bound log-likelihood minus **unbound**
log-likelihood). That is the log-ratio our leverage integrates.

**Reading:** the field's dedicated, *retrained* selective-binder method operationalizes the identical
contrastive/binding direction we name and steer with. The direction is not idiosyncratic to our pipeline —
an independent group converged on it for design.

## 2. RedNet's genuine additions (its edge over a frozen tilt)
Two things RedNet has that a frozen ProteinMPNN `+α·L` does not:
1. **A retrained decoder.** RedNet trains its own atom/graph decoder (`src/rednet/rednet_model.py`,
   `atom_gat_model.py`; configs `configs/model/rednet_pretrain_atom_gat_*.yaml`) — the contrast is applied on
   top of purpose-trained logits, not frozen off-the-shelf ones.
2. **A β plausibility mask** (`sampling_utils.py:28–31`): `cst_mask = cst_probs > β·max(cst_probs)` then
   `logits.masked_fill_(~cst_mask, −1e9)` — an adaptive nucleus filter that forbids low-probability residues
   before sampling. Our tilt has no such mask.

So the honest framing (never run, so never claimed as measured): a frozen `+α·L` tilt tests **how much of
RedNet's benefit comes from the shared contrastive *direction* vs. from retraining + the β-mask.** The
`naive-guidance` arm (`PREREG_naive.md`, `FINDINGS_naive.md`) is the accessible half of that question — it shows
the *direction* itself (not any tilt) carries the benefit.

## 3. Why the head-to-head is blocked (stated plainly)
Two independent, external, non-resolvable blockers:
1. **Private companion packages.** `cli/infer_pipeline.py` (lines 21–22) and `cli/make_select_data.py`
   (lines 20–22) import `faust` (`faust.tokenizer`, `faust.tools.struct_align`, `faust.utils.parallel`) and
   `atomtools` (`atomtools.redesign.redesign_worker`) at module top. Neither is in `pyproject.toml`'s
   dependencies (lines 17–40), neither is bundled in the repo, and the author (`zw2x`) has **6 public repos**,
   none named `faust`/`atomtools` (the PyPI packages of those names are unrelated: a stream-processing library
   and a comp-chem helper, with different APIs). Every design entry point (`run_sel`, `run_hdimer`, `run_seq`,
   `run_skempi`) fails at *import*, and the feature-prep (`StructurePipeline`, TM-align) also needs `faust` — so
   our complexes cannot be formatted into their pipeline.
2. **Access-restricted weights.** The Zenodo record `20113403` shows *"The record is publicly accessible, but
   files are restricted. Log in to check if you have access."* — the checkpoint is not freely downloadable.

Either blocker alone prevents running RedNet's sampler on our targets; both hold. Per rule 3 we stop and report
rather than fake a number or pass a re-implementation of their contrastive decode off as "RedNet" (that would
just be our own method relabeled). If access to the weights **and** the `faust`/`atomtools` packages becomes
available (author contact / granted Zenodo access), the true head-to-head — `run_hdimer` on the 60-complex
ipTM set with matched interface positions, judged/folded identically — is the drop-in follow-up.
