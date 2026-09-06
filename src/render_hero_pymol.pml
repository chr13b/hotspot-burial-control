# render_hero_pymol.pml — publication 3-panel hero for the pre-registered median-effect complex 1GL0_E_I.
# wt / L-steered / random, cartoon colored by pLDDT (B-factor), interface contact patch as sticks,
# CONSISTENT orientation (all superposed on the constant partner chain B), ray-traced grid.
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

# consistent orientation: superpose L and random onto wt on the constant partner chain B (chain A = steered E)
super Lst and chain B, wt and chain B
super rnd and chain B, wt and chain B

hide everything
show cartoon
# color by pLDDT stored in B-factor: low 50 = red -> high 90 = blue (AF2 confidence convention)
spectrum b, red_white_blue, minimum=50, maximum=90

# structural interface patch (A<->B contacts) shown as side-chain sticks (single line — .pml has no line continuation)
select iface, byres ((wt and chain A) within 4.5 of (wt and chain B)) or byres ((wt and chain B) within 4.5 of (wt and chain A)) or byres ((Lst and chain A) within 4.5 of (Lst and chain B)) or byres ((Lst and chain B) within 4.5 of (Lst and chain A)) or byres ((rnd and chain A) within 4.5 of (rnd and chain B)) or byres ((rnd and chain B) within 4.5 of (rnd and chain A))
show sticks, iface and not (name C+N+O)

orient wt
turn y, 15

# per-panel annotations (values from 1GL0_E_I_meta.csv) shown as grid-cell titles
set_title wt,  1, "wt      | ipTM 0.94 | iface pLDDT 95.8 | iface pAE 2.13"
set_title Lst, 1, "L (k0)  | ipTM 0.90 | iface pLDDT 91.4 | iface pAE 3.02  [arm-mean ipTM 0.907]"
set_title rnd, 1, "random(k1)| ipTM 0.83 | iface pLDDT 76.7 | iface pAE 5.59  [arm-mean ipTM 0.71]"

set grid_mode, 1
set grid_slot, 1, wt
set grid_slot, 2, Lst
set grid_slot, 3, rnd

ray 2700, 1000
png results/figures/fig_hero_pymol.png, dpi=300
