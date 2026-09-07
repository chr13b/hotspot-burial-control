# render_hero_pymol.pml — publication 3-panel hero for the pre-registered median-effect complex 1GL0_E_I.
# wt / L-steered / random, cartoon colored by pLDDT (B-factor), interface contact patch as sticks,
# CONSISTENT orientation (all superposed on receptor chain A, matching fig_hero.py's RECEPTOR mask), ray-traced grid.
# Readability pass: figure-level title+subtitle, a per-panel receptor/binder/interface callout (label +
# thin leader line + ring marker), a pLDDT colorbar legend, and fuller per-panel titles -- all composited
# in the embedded python block below. The three callout pixel targets (IFACE_XY, BINDER_XY) were measured
# empirically: a diagnostic render placed a colored marker at the real chain-A COM / chain-B COM / iface
# COM (computed from the wt object's own coordinates) and located its rendered pixel centroid by color;
# grid_mode was verified to use ONE shared camera for all three cells (the non-white bounding box of each
# panel matches to within a few px), so wt's measured coordinates apply to all three panels.
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
from pymol import cmd

OUT  = "results/figures/fig_hero_pymol.png"
META = "results/hero_pdbs/1GL0_E_I_meta.csv"
ARMS = ["wt", "L", "random"]
OBJ  = {"wt": "wt", "L": "Lst", "random": "rnd"}

TITLE_MAIN = {"wt": "wt", "L": "L-steered (+α·L)", "random": "random"}
TITLE_SUB  = {"wt": "crystal sequence", "L": "the binding-optimised design",
              "random": "matched-magnitude control"}

rows = {r["arm"]: r for r in csv.DictReader(open(META))}

# live interface-residue counts per arm, from the SAME `iface` selection used for the sticks above --
# never hardcoded, so this can never drift from what is actually drawn (and it is NOT the same number
# as fig_hero.py's committed 43 -- that is a different, DeltaSASA-based definition; this one is each
# arm's own byres/4.5A heavy-atom contact set, which is why it varies slightly arm to arm).
NIFACE = {a: cmd.count_atoms(f"iface and {OBJ[a]} and name CA") for a in ARMS}

fdir     = os.path.join(sys.prefix, "fonts")
f_htitle = ImageFont.truetype(os.path.join(fdir, "Ubuntu-B.ttf"), 40)
f_hsub   = ImageFont.truetype(os.path.join(fdir, "Ubuntu-R.ttf"), 26)
f_bold28 = ImageFont.truetype(os.path.join(fdir, "Ubuntu-B.ttf"), 28)
f_bold24 = ImageFont.truetype(os.path.join(fdir, "Ubuntu-B.ttf"), 24)
f_reg24  = ImageFont.truetype(os.path.join(fdir, "Ubuntu-R.ttf"), 24)
f_bold26 = ImageFont.truetype(os.path.join(fdir, "Ubuntu-B.ttf"), 26)
f_reg20  = ImageFont.truetype(os.path.join(fdir, "Ubuntu-R.ttf"), 20)

INK, MUTED, SOFT, RULE = (26, 26, 26), (107, 115, 121), (51, 51, 51), (77, 77, 77)

render = Image.open(OUT).convert("RGB")
W, H = render.size                                     # 2700 x 1000 (3 panels of 900 x 1000)
PW = W // 3
HEADER_H, CAP_H, LEG_H = 150, 150, 165
canvas = Image.new("RGB", (W, HEADER_H + H + CAP_H + LEG_H), "white")
draw = ImageDraw.Draw(canvas)

def center_text(cx, y, text, font, fill):
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (x1 - x0) / 2.0, y), text, font=font, fill=fill)
    return x1 - x0

def right_text(rx, y, text, font, fill):
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    draw.text((rx - (x1 - x0), y), text, font=font, fill=fill)
    return x1 - x0

# ---------------------------------------------------------------------------------- figure header
center_text(W / 2.0, 22, "1GL0_E_I — one complex, three interface-sequence designs",
            f_htitle, INK)
center_text(W / 2.0, 80,
            "Same complex, three interface-sequence designs; colored by AlphaFold pLDDT — the "
            "random control's interface (right) is visibly less confident.", f_hsub, MUTED)

# ---------------------------------------------------------------------------------- the render itself
canvas.paste(render, (0, HEADER_H))

# per-panel callouts: chain labels + an interface pointer, drawn directly onto the pasted render.
# "receptor" needs no leader line (chain A is almost the whole panel, 241 of 273 residues); "interface"
# and "binder" get a thin line + a small ring landing on an empirically measured target pixel (see the
# header comment). Both targets sit inside the render's own blank top margin (rows 0-139 of every
# panel are pure background, verified on the committed PNG), so the label text never overlaps cartoon.
IFACE_XY  = (576, 553)   # centroid of the wt arm's own byres/4.5A contact set
BINDER_XY = (832, 540)   # chain-B CA farthest from that centroid -- unambiguously binder territory,
                          # not shared with the interface callout

def callout(anchor_xy, target_xy, r=9):
    draw.line([anchor_xy, target_xy], fill=RULE, width=3)
    tx, ty = target_xy
    draw.ellipse([tx - r - 2, ty - r - 2, tx + r + 2, ty + r + 2], outline=(255, 255, 255), width=3)
    draw.ellipse([tx - r, ty - r, tx + r, ty + r], outline=(20, 20, 20), width=2)

for i, a in enumerate(ARMS):
    ox, oy = i * PW, HEADER_H
    draw.text((ox + 18, oy + 14), "receptor (chain A · 241 aa)", font=f_bold24, fill=SOFT)
    ix = ox + 500
    center_text(ix, oy + 14, "interface", f_bold24, SOFT)
    callout((ix, oy + 48), (ox + IFACE_XY[0], oy + IFACE_XY[1]))
    bx = ox + 882
    right_text(bx, oy + 14, "binder (chain B · 32 aa)", f_bold24, SOFT)
    callout((bx - 60, oy + 48), (ox + BINDER_XY[0], oy + BINDER_XY[1]))

# ---------------------------------------------------------------------------------- per-panel captions
CAP0 = HEADER_H + H
for i, a in enumerate(ARMS):
    r  = rows[a]
    cx = (i + 0.5) * PW
    k  = int(float(r["k"]))
    title = TITLE_MAIN[a] + (f"  (k={k})" if k >= 0 else "")
    line2 = (f"ipTM {float(r['iptm']):.2f}    iface pLDDT {float(r['interface_plddt']):.1f}    "
             f"iface pAE {float(r['interface_pae']):.2f}")
    line3 = f"arm-mean ipTM {float(r['arm_mean_iptm']):.3f}"
    center_text(cx, CAP0 + 8,   title,             f_bold28, INK)
    center_text(cx, CAP0 + 44,  TITLE_SUB[a],       f_reg24,  MUTED)
    center_text(cx, CAP0 + 80,  line2,              f_reg24,  (60, 60, 60))
    center_text(cx, CAP0 + 114, line3,              f_reg24,  (90, 90, 90))

# ---------------------------------------------------------------------------------- legend: pLDDT bar
# the single most important missing element in the pre-readability-pass figure: a colorbar, matching
# EXACTLY the `spectrum b, red_white_blue, minimum=50, maximum=90` call above (red -> white -> blue,
# linear in each half), so the legend cannot silently drift from what is actually mapped onto the cartoon.
LEG0 = CAP0 + CAP_H
BAR_W, BAR_H = 560, 32
bar_x0, bar_y0 = (W - BAR_W) // 2, LEG0 + 46

def rwb(t):
    RED, WHITE, BLUE = (255, 0, 0), (255, 255, 255), (0, 0, 255)
    if t < 0.5:
        u = t / 0.5
        return tuple(int(round(RED[k] + (WHITE[k] - RED[k]) * u)) for k in range(3))
    u = (t - 0.5) / 0.5
    return tuple(int(round(WHITE[k] + (BLUE[k] - WHITE[k]) * u)) for k in range(3))

bar = Image.new("RGB", (BAR_W, BAR_H))
bpx = bar.load()
for x in range(BAR_W):
    c = rwb(x / (BAR_W - 1))
    for y in range(BAR_H):
        bpx[x, y] = c
canvas.paste(bar, (bar_x0, bar_y0))
draw.rectangle([bar_x0, bar_y0, bar_x0 + BAR_W - 1, bar_y0 + BAR_H - 1], outline=(150, 150, 150), width=1)

center_text(W / 2.0, LEG0 + 8, "per-residue pLDDT", f_bold26, INK)
for frac, lab in [(0.0, "50"), (0.5, "70"), (1.0, "90")]:
    tx = bar_x0 + frac * (BAR_W - 1)
    draw.line([(tx, bar_y0 + BAR_H), (tx, bar_y0 + BAR_H + 6)], fill=RULE, width=2)
    center_text(tx, bar_y0 + BAR_H + 10, lab, f_reg20, RULE)

note = (f"Sticks + ringed callouts mark the receptor–binder contact patch (heavy-atom contact "
        f"≤ 4.5 Å; {NIFACE['wt']}/{NIFACE['L']}/{NIFACE['random']} residues in wt / "
        f"L-steered / random).")
center_text(W / 2.0, bar_y0 + BAR_H + 40, note, f_reg20, MUTED)

canvas.save(OUT, dpi=(300, 300))
print(f"[annotate] header + per-panel receptor/binder/interface callouts + pLDDT legend + captions "
      f"for {ARMS} composited from {META} -> {OUT}  (live interface residue counts: {NIFACE})")
python end
