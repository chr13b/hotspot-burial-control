"""Experiment B, part 2 (the F3 gate): commitment times t*_seq and t*_str for MultiFlow co-design.

BRIEF §2.3: each channel has a commitment time t* = the 0.5-crossing of normalised agreement between
the model's endpoint prediction x_hat_1(t) and the realised final value x_1. Sequence agreement =
token-argmax match to the final sequence; structure agreement = Cα contact-map (Jaccard) overlap with
the final structure. If t*_str < t*_seq the model fixes the fold before the chemistry — the staged
pipeline with extra steps.

KILL F3: t*_seq <= t*_str + 0.05 under the default schedule, stable across >=3 seeds and >=2 length
bins, means joint models already decide sequence first and the diagnosis is factually wrong.

Run in the multiflow conda env on a V100/RTX_3090 (torch 2.0.1+cu117):
  python src/expB_commitment.py --mf-dir $SCRATCH/expB/multiflow \
      --ckpt $SCRATCH/expB/multiflow/weights/last.ckpt --lengths 100,200 --seeds 0,1,2 \
      --out results/expB_commitment.csv
"""
import argparse
import os
import sys

import numpy as np


def contact_map(ca, thresh=8.0):
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    m = d < thresh
    np.fill_diagonal(m, False)
    return m


def contact_overlap(ca_t, ca_final):
    a, b = contact_map(ca_t), contact_map(ca_final)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def crossing(ts, agree, level=0.5):
    """First t where agreement rises through `level` (linear interp)."""
    agree = np.asarray(agree)
    for i in range(1, len(agree)):
        if agree[i] >= level and agree[i - 1] < level:
            f = (level - agree[i - 1]) / (agree[i] - agree[i - 1] + 1e-12)
            return float(ts[i - 1] + f * (ts[i] - ts[i - 1]))
    return float(ts[-1]) if agree[-1] >= level else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mf-dir", default=os.path.expandvars("$SCRATCH/expB/multiflow"))
    ap.add_argument("--ckpt", default=os.path.expandvars("$SCRATCH/expB/multiflow/weights/last.ckpt"))
    ap.add_argument("--lengths", default="100,200")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--num-timesteps", type=int, default=500)
    ap.add_argument("--out", default="results/expB_commitment.csv")
    a = ap.parse_args()

    sys.path.insert(0, a.mf_dir)
    import torch
    from omegaconf import OmegaConf
    from multiflow.models.flow_module import FlowModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[commit] device={device} torch={torch.__version__}")

    # Compose the inference config through Hydra (resolves defaults: [base, model, ...]) exactly as the
    # CLI does, then merge the checkpoint's (older) training config on top — replicates
    # inference_se3_flows.py. This supplies model fields the ckpt config predates (aatype_pred_num_tokens).
    from hydra import compose, initialize_config_dir
    ckpt_dir = os.path.dirname(a.ckpt)
    cfg_dir = os.path.join(a.mf_dir, "multiflow/configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        base_cfg = compose(config_name="inference_unconditional")
    ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    OmegaConf.set_struct(base_cfg, False)
    OmegaConf.set_struct(ckpt_cfg, False)
    cfg = OmegaConf.merge(base_cfg, ckpt_cfg)
    # Use the released INFERENCE interpolant (default schedule: do_purity, temp 0.1); set trajectory
    # resolution via num_timesteps. Continuous schedule left as released.
    cfg.interpolant = base_cfg.inference.interpolant
    cfg.interpolant.sampling.num_timesteps = a.num_timesteps

    import multiflow.experiments.utils as eu
    try:
        dataset_cfg = eu.get_dataset_cfg(cfg)
    except Exception:
        dataset_cfg = cfg.get("pdb_post2021_dataset", cfg.get("data", None))

    fm = FlowModule.load_from_checkpoint(
        checkpoint_path=a.ckpt, cfg=cfg, dataset_cfg=dataset_cfg,
        folding_cfg=None, map_location=device, strict=False)
    fm.eval().to(device)
    interp = fm.interpolant
    interp.set_device(device)
    model = fm.model
    CA = 1  # atom37 CA index

    lengths = [int(x) for x in a.lengths.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = []
    for L in lengths:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            with torch.no_grad():
                prot_traj, model_traj = interp.sample(
                    1, L, model,
                    diffuse_mask=torch.ones(1, L, device=device),
                    res_idx=torch.arange(L, device=device)[None].float(),
                    chain_idx=torch.ones(1, L, device=device))
            # realised final value
            final_aa = prot_traj[-1][1][0].numpy()
            final_ca = prot_traj[-1][0][0][:, CA].numpy()
            ts = np.linspace(interp._cfg.min_t, 1.0, len(model_traj))
            seq_ag = [float((mt[1][0].numpy() == final_aa).mean()) for mt in model_traj]
            str_ag = [contact_overlap(mt[0][0][:, CA].numpy(), final_ca) for mt in model_traj]
            t_seq = crossing(ts, seq_ag)
            t_str = crossing(ts, str_ag)
            print(f"  L={L} seed={seed}  t*_seq={t_seq:.3f}  t*_str={t_str:.3f}  "
                  f"(str commits {'FIRST' if t_str < t_seq else 'after'} seq)", flush=True)
            rows.append(dict(length=L, seed=seed, t_seq=t_seq, t_str=t_str,
                             seq_first=int(t_seq <= t_str + 0.05)))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    # F3 verdict
    ts_seq, ts_str = df["t_seq"].mean(), df["t_str"].mean()
    sd = df.groupby("length")[["t_seq", "t_str"]].std().mean().mean()
    f3 = (df["t_seq"] <= df["t_str"] + 0.05).mean()
    print(f"\n[commit] mean t*_seq={ts_seq:.3f}  t*_str={ts_str:.3f}  (seed/len SD ~{sd:.3f})")
    print(f"[commit] fraction seeds with t*_seq<=t*_str+0.05: {f3:.2f}  ->  "
          f"F3 {'FIRES (diagnosis wrong: seq commits first)' if f3 > 0.5 else 'does NOT fire (structure commits first)'}")
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()
