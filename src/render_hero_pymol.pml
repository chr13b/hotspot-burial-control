# render_hero_pymol.pml — publication 3-panel hero for the pre-registered median-effect complex 1GL0_E_I.
# wt / L-steered / random, cartoon colored by pLDDT (B-factor), interface contact patch as sticks,
# CONSISTENT orientation (all superposed on receptor chain A, matching fig_hero.py's RECEPTOR mask), ray-traced grid.
# Run: pymol -cq src/render_hero_pymol.pml   (annotations come from results/hero_pdbs/1GL0_E_I_meta.csv)
reinitialize
bg_color white
set ray_opaque_background, 1
set ray_shadows, 0
set antialias, 2
set cartoon_side_chain_helper, on
set cartoon_transparency, 0.0
set stick_radius, 0.22

load results/hero_pdbs/1GL0_E_I__wt.pdb,     wt
load results/hero_pdbs/1GL0_E_I__L.pdb,      Lst
load results/hero_pdbs/1GL0_E_I__random.pdb, rnd

# consistent orientation: superpose L and random onto wt on RECEPTOR chain A, 241 aa, crystal chain E --
# this matches the RECEPTOR mask in src/fig_hero.py, which fig_hero.py also uses as its fit reference.
# NOTE: fixed from the originally committed script, which superposed on chain B instead -- verified by
# diffing residue names per chain across the three PDBs: BOTH chains carry substitutions in every arm --
# chain A has 15/17 diffs vs wt for L/random -- chain B, the 32-aa binder, has 12/14 diffs -- so neither
# chain is sequence-constant, but chain A is the larger, more rigid anchor and the one fig_hero.py fits on.
super Lst and chain A, wt and chain A
super rnd and chain A, wt and chain A

hide everything
show cartoon
# color by pLDDT stored in B-factor: low 50 = red -> high 90 = blue (AF2 confidence convention)
spectrum b, red_white_blue, minimum=50, maximum=90

# structural interface patch (A<->B contacts) shown as side-chain sticks (single line — .pml has no line continuation)
select iface, byres ((wt and chain A) within 4.5 of (wt and chain B)) or byres ((wt and chain B) within 4.5 of (wt and chain A)) or byres ((Lst and chain A) within 4.5 of (Lst and chain B)) or byres ((Lst and chain B) within 4.5 of (Lst and chain A)) or byres ((rnd and chain A) within 4.5 of (rnd and chain B)) or byres ((rnd and chain B) within 4.5 of (rnd and chain A))
show sticks, iface and not (name C+N+O)

orient wt
turn y, 15

# NOTE: set_title (the movie/state caption) writes into internal-GUI chrome that "pymol -c" never
# creates, so it is invisible in a ray-traced PNG -- confirmed empirically (title text was absent from
# the rendered image even though set_title ran without error). Annotations are instead composited onto
# the PNG below, in the embedded python block, reading straight from 1GL0_E_I_meta.csv (not hardcoded)
# so the numbers cannot drift from the committed CSV.

set grid_mode, 1
set grid_slot, 1, wt
set grid_slot, 2, Lst
set grid_slot, 3, rnd

ray 2700, 1000
png results/figures/fig_hero_pymol.png, dpi=300

python
from PIL import Image, ImageDraw, ImageFont
import csv, os, sys

OUT  = "results/figures/fig_hero_pymol.png"
META = "results/hero_pdbs/1GL0_E_I_meta.csv"
ARMS = ["wt", "L", "random"]
TITLE = {"wt": "wt", "L": "L-steered", "random": "random-matched"}

rows = {r["arm"]: r for r in csv.DictReader(open(META))}

fdir   = os.path.join(sys.prefix, "fonts")
f_bold = ImageFont.truetype(os.path.join(fdir, "Ubuntu-B.ttf"), 30)
f_reg  = ImageFont.truetype(os.path.join(fdir, "Ubuntu-R.ttf"), 24)

img = Image.open(OUT).convert("RGB")
W, H = img.size
PW, BAND = W // 3, 130
canvas = Image.new("RGB", (W, H + BAND), "white")
canvas.paste(img, (0, 0))
draw = ImageDraw.Draw(canvas)

def center_text(cx, y, text, font, fill):
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (x1 - x0) / 2.0, y), text, font=font, fill=fill)

for i, a in enumerate(ARMS):
    r  = rows[a]
    cx = (i + 0.5) * PW
    k  = int(float(r["k"]))
    label = TITLE[a] + (f"  (k={k})" if k >= 0 else "")
    line2 = (f"ipTM {float(r['iptm']):.2f}    iface pLDDT {float(r['interface_plddt']):.1f}    "
             f"iface pAE {float(r['interface_pae']):.2f}")
    line3 = f"arm-mean ipTM {float(r['arm_mean_iptm']):.3f}"
    center_text(cx, H + 8,  label, f_bold, (20, 20, 20))
    center_text(cx, H + 48, line2, f_reg,  (60, 60, 60))
    center_text(cx, H + 84, line3, f_reg,  (90, 90, 90))

canvas.save(OUT, dpi=(300, 300))
print(f"[annotate] captions for {ARMS} composited from {META} -> {OUT}")
python end
