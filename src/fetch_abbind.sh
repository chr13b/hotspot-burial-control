#!/bin/bash
# Re-fetch the AB-Bind release (Sirin, Apgar, Bennett, Keating 2016, Protein Science 25:393-409) into
# ~/ftax/data/ab-bind for the FoldX Task-2 fixture. Public source; we do NOT vendor the data (kept re-fetchable to
# survive $SCRATCH purge). The authoritative Partners(A_B) chain-group partition + the AB-Bind-specific (renumbered)
# structures both come from here — the generic RCSB PDBs have DIFFERENT numbering and fail the WT-identity gate.
set -euo pipefail
AB="$HOME/ftax/data/ab-bind"
base="https://raw.githubusercontent.com/sarahsirin/AB-Bind-Database/master"
mkdir -p "$AB"; cd "$AB"
curl -fsSL -o AB-Bind_experimental_data.csv "$base/AB-Bind_experimental_data.csv"
echo "expected sha256: 078ab9b5c64459d5aa2e39cfdfbabdedcd946fa6bccfb49d81f59df2e175e29f"
sha256sum AB-Bind_experimental_data.csv
for p in 1AK4 1BJ1 1CZ8 1DQJ 1DVF 1FFW 1JRH 1JTG 1KTZ 1MHP 1MLC 1N8Z 1T83 1VFB 1YY9 2JEL \
         2NY7 2NYY 2NZ9 3BDY 3BE1 3BN9 3HFM 3K2M 3NGB 3NPS 3WJJ; do
  curl -fsSL -o "$p.pdb" "$base/$p.pdb"
done
echo "fetched AB-Bind CSV + $(ls ./*.pdb | wc -l) crystal PDBs into $AB"
