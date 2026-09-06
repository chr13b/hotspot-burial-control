# PRE-REGISTRATION — does steering a SECOND frozen model (ESM-IF1) by +α·L also work? (SET-B)

**Committed BEFORE any SET-B number (CLAUDE.md rule 1). Frozen — do not edit after the first score/fold.**
`SEED=20260803`. The standing CFG result steers **ProteinMPNN** by `+α·L` and measures the effect with an
INDEPENDENT model (ESM-IF1 judge, AF2 fold). This tests the converse: steer **frozen ESM-IF1** by `+α·L` and
measure with independent models (ProteinMPNN/PiFold/MIF judges; AF2 fold). If the steering direction is a real
property of the leverage operator and not one model's quirk, the reverse steering should also raise binding-
favorability. This is the within-modality robustness leg (paired with the Boltz-2 cross-modality leg).

## Design
Steer frozen ESM-IF1 by biasing its **native-conditional interface distribution** by `+α·L_i(a)` and sampling
(`src/cfg_steer_esmif.py`): compute ESM-IF1's per-position conditional log-probabilities (its native-context
teacher-forced readout — the same one used for its leverage), add `α·L_i(a)` at the committed interface
positions, and sample K sequences (each interface position drawn from the biased softmax at temperature T;
non-interface positions held at wt — wt-background, steered-interface, exactly like `cfg_steer.py`'s output). `L`
is the **ProteinMPNN-derived** interface leverage already used by `cfg_steer.py` (the operator under test is
model-agnostic; the question is whether ESM-IF1, when tilted by it, samples more binding-favorable interface
residues). Same interface set, α grid {0, 2}, K as `cfg_steer.py`. NOTE (disclosed up front, not after seeing a
number): this is a **one-shot biased-conditional** sampler — interface positions are drawn independently given
the native context, whereas `cfg_steer.py`'s ProteinMPNN path is autoregressively coupled. This is the natural
sampler for ESM-IF1's conditional readout and is reported as such. Emit arms wt / L(k=0..2) / random(k=0..2) on
the **batch-1 60 complexes** (`results/iptm_subset.txt`) → `results/cfg_steer_esmif_seqs.csv`. `random` =
matched-magnitude control: the SAME per-position leverage 20-vector permuted across amino acids (identical
multiset/magnitude, shuffled direction), identical construction to `cfg_steer.py` (`Rperm`).

## Hypotheses
- **H1 (judge-leverage, load-bearing).** The **ProteinMPNN / PiFold / MIF** interface-leverage of the SET-B
  sampled residues is higher for the **L arm than the matched-random arm**, paired per complex (complex-clustered
  bootstrap 95% CI excluding 0), for at least the primary judge (ProteinMPNN). Mirrors the standing ESM-IF1-judge
  result on SET-A (α: −0.20 → +0.27; L−random +0.77). Non-self judges only (NOT ESM-IF1 — it is the steered
  model → circular).
- **H2 (recovery preserved).** ESM-IF1 interface native recovery under the L arm is not degraded relative to wt
  (no foldability/recovery cost from steering).
- **H3 (fold localization).** When SET-B is folded with AF2-multimer, the interface composite/ipTM benefit
  (L vs random) is interface-local — global pTM shifts far less.
- **Fold H1 (transfer).** Paired **composite(L) > composite(random)** AND **ipTM(L) > ipTM(random)** under
  AF2-multimer (both CI>0), mirroring `PREREG_iptm.md`.

## Falsifier (report verbatim if it fires)
If the paired **L − random** judge-leverage (primary judge) ≤ 0, **or** the AF2 paired composite/ipTM L−random
CI includes 0, then steering ESM-IF1 by `+α·L` **does not corroborate** the steering result → reported verbatim
as a bounded within-modality result. A null bounds the claim; it does not erase the standing ProteinMPNN-steer
result.

## Positive controls (rule 6)
1. **wt reuse identity.** SET-B wt is the identical crystal sequence already folded in the AF2 ipTM run — REUSE
   those wt models (do not re-fold); verify the reused wt fold-ids match. Fold only L+random = **360 new folds**.
2. **wt-sanity + determinism** kept visible next to the effect (median wt interface ipTM ~0.6–0.9; seed SD).
3. **Steer-then-measure is anti-circular** by construction (steer ESM-IF1, judge by non-ESM-IF1 models).

Outputs: `results/cfg_steer_esmif_seqs.csv`, `results/iptm_steer_esmif.csv`, `results/iptm_summary_esmif.csv`,
the judge rows appended to `results/cfg_judge_matrix.csv` (`steered_model=ESM-IF1`), `src/cfg_steer_esmif.py`,
`results/FINDINGS_esmif_steer.md`.
