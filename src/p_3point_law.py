"""The constraint-vs-leverage gradient as a multi-point law (pre-registered: results/PREREG_3point_law.md).

Stratify SKEMPI interface-hotspot CONFIDENCE-AUROC by SKEMPI's own Hold_out_type (one pipeline, one label
construction), + de-novo (Bennett) as the anchor. Confidence = log p(native). Circularity control:
burial-residualized confidence-AUROC (if confidence is just burial, it collapses to 0.5). Complex-clustered
bootstrap. Ordering (transience rank) is FROZEN in the pre-reg: TCR/pMHC(1) < AB/AG(2) < Pr/PI(3).

  python3 src/p_3point_law.py --out results/threepoint_law.csv
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats
SEED = 20260803
RANK = {"TCR/pMHC": 1, "AB/AG": 2, "Pr/PI": 3}          # pre-registered a-priori transience order


def auroc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    m = np.isfinite(s) & np.isfinite(y); y, s = y[m], s[m]
    npos, nneg = y.sum(), len(y) - y.sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = stats.rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def resid_on_burial(conf, burial):
    m = np.isfinite(conf) & np.isfinite(burial)
    b = np.polyfit(burial[m], conf[m], 1)
    out = np.full_like(conf, np.nan)
    out[m] = conf[m] - (b[0] * burial[m] + b[1])
    return out


def clustered_ci(df, scorecol, rng, nb=2000):
    comps = df.complex_id.unique()
    grp = {c: (df.loc[df.complex_id == c, "is_hot"].to_numpy(),
               df.loc[df.complex_id == c, scorecol].to_numpy()) for c in comps}
    vals = []
    for _ in range(nb):
        cs = rng.choice(comps, len(comps), replace=True)
        Y = np.concatenate([grp[c][0] for c in cs]); S = np.concatenate([grp[c][1] for c in cs])
        vals.append(auroc(Y, S))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/threepoint_law.csv"); a = ap.parse_args()
    DATA = os.path.expanduser("~/ftax/data")
    sk = pd.read_csv(f"{DATA}/skempi_v2.csv", sep=";", low_memory=False)
    sk["pdb"] = sk["#Pdb"].str.split("_").str[0].str.upper()
    cls = sk.groupby("pdb")["Hold_out_type"].agg(lambda x: x.mode().iat[0] if len(x.mode()) else None)

    d = pd.read_csv("results/leverage_skempi_positions.csv", low_memory=False)
    d = d[d.is_interface == True].copy()                                          # noqa: E712
    d["pdb"] = d.complex_id.str.split("_").str[0].str.upper()
    d["cls"] = d.pdb.map(cls)
    d["is_hot"] = d.is_hot.astype(float)
    d = d.dropna(subset=["conf", "burial", "cls"])
    d["conf_resid"] = resid_on_burial(d.conf.to_numpy(), d.burial.to_numpy())

    rows = []
    print(f"{'stratum':12s} {'n_cx':>5s} {'n_hot':>6s} {'conf-AUROC':>22s} {'burial-AUROC':>12s} {'conf|burial':>12s}")
    for name in ["TCR/pMHC", "AB/AG", "Pr/PI"]:
        g = d[d.cls == name]
        if not len(g):
            continue
        rng = np.random.default_rng(SEED)
        ca = auroc(g.is_hot, g.conf); lo, hi = clustered_ci(g, "conf", rng)
        ba = auroc(g.is_hot, g.burial)
        cra = auroc(g.is_hot, g.conf_resid)
        nhot = int(g.is_hot.sum()); ncx = g.complex_id.nunique()
        flag = " UNDERPOWERED" if (ncx < 5 or nhot < 15) else ""
        print(f"{name:12s} {ncx:5d} {nhot:6d}   {ca:.3f} [{lo:.3f},{hi:.3f}]   {ba:>10.3f}   {cra:>10.3f}{flag}")
        rows.append(dict(stratum=name, transience_rank=RANK[name], n_complexes=ncx,
                         n_interface=len(g), n_hot=nhot, conf_auroc=round(ca, 4),
                         lo=round(lo, 4), hi=round(hi, 4), burial_auroc=round(ba, 4),
                         conf_resid_burial_auroc=round(cra, 4), pipeline="SKEMPI/leverage_positions",
                         underpowered=bool(ncx < 5 or nhot < 15)))

    # de-novo anchor (different pipeline; read whatever confidence-AUROC bennett_conf_fork.csv exposes)
    try:
        bf = pd.read_csv("results/bennett_conf_fork.csv")
        print("\n[de-novo anchor] bennett_conf_fork.csv rows:")
        print(bf.to_string(index=False)[:1500])
        rows.append(dict(stratum="de-novo (Bennett, ANCHOR)", transience_rank=4,
                         pipeline="bennett_conf_fork.csv", note="different pipeline/labels; see CSV printed above"))
    except Exception as e:
        print("  (bennett_conf_fork.csv unreadable:", e, ")")

    # pre-registered H2 trend over the 3 natural classes (powered only)
    nat = [r for r in rows if r.get("pipeline", "").startswith("SKEMPI") and not r.get("underpowered")]
    if len(nat) >= 3:
        rk = [r["transience_rank"] for r in nat]; au = [r["conf_auroc"] for r in nat]
        rho = stats.spearmanr(rk, au).correlation
        mono = all(au[i] <= au[i + 1] for i in range(len(au) - 1)) if [x for _, x in sorted(zip(rk, au))] == au else \
            all(x <= y for x, y in zip(sorted(zip(rk, au)), sorted(zip(rk, au))[1:]))
        srt = [x for _, x in sorted(zip(rk, au))]
        mono = all(srt[i] <= srt[i + 1] for i in range(len(srt) - 1))
        print(f"\n[H2] Spearman(transience-rank, conf-AUROC) over natural classes = {rho:+.3f}; "
              f"monotone rise (TCR<AB<Pr) = {mono}  (sorted-by-rank AUROCs = {[round(x,3) for x in srt]})")

    out = pd.DataFrame(rows); out["seed"] = SEED
    out["command"] = "python3 src/p_3point_law.py"
    out.to_csv(a.out, index=False); print(f"\n[wrote] {a.out}")


if __name__ == "__main__":
    main()
