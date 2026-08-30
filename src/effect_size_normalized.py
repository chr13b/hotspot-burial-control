"""Effect sizes in units an ML reviewer can price: normalize each position-level CPI by the ENTROPY of the
hotspot label. The hotspot label is rare (base rate ~2.4%), so a raw CPI of +0.005 understates the signal;
as a fraction of the label's own entropy it reads as "recovers X% of what's there to recover".

  python3 src/effect_size_normalized.py --out results/effect_size_normalized.csv
"""
import argparse
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/effect_size_normalized.csv"); a = ap.parse_args()
    pos = pd.read_csv("results/leverage_skempi_positions.csv", low_memory=False)
    pos = pos[pos.is_interface == True]                                              # noqa: E712
    n, nhot = len(pos), int(pos.is_hot.sum()); p = nhot / n
    H = float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))                            # binary entropy, nats
    lad = pd.read_csv("results/w_placebo_ladder.csv")
    # pull the position-level CPI for each feature from the committed ladder
    want = {"placebo floor": None, "confidence": None, "negentropy": None, "scalar KL": None,
            "leverage": None}
    rows = [dict(quantity="hotspot_base_rate", value=round(p, 4), n=n, n_hot=nhot),
            dict(quantity="hotspot_label_entropy_nats", value=round(H, 4), n=n, n_hot=nhot)]
    # map ladder rows -> normalized fraction of label entropy
    for _, r in lad.iterrows():
        name = str(r.get("feature", r.iloc[0])).lower()
        cpi = float(r.get("cpi", r.get("value", np.nan)))
        if not np.isfinite(cpi):
            continue
        rows.append(dict(quantity=f"CPI_over_labelH::{name}", value=round(cpi, 5),
                         normalized_pct_of_labelH=round(100 * cpi / H, 2), n=n))
    out = pd.DataFrame(rows)
    print(f"  n={n} interface positions, {nhot} hotspots, base rate = {p:.4f}")
    print(f"  hotspot-label entropy H = {H:.4f} nats")
    print(out[out.quantity.str.startswith('CPI')][['quantity', 'value', 'normalized_pct_of_labelH']].to_string(index=False))
    out["seed"] = 20260803; out["command"] = "python3 src/effect_size_normalized.py"
    out.to_csv(a.out, index=False); print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()
