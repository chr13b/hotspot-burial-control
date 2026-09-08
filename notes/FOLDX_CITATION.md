# Citing FoldX (for the paper) + license note

We use FoldX as a **physics ΔΔG_bind comparator** (`PREREG_foldx.md`, `FOLDX_PLAN.md`). When FoldX results land,
add these to the manuscript **Methods** (name the exact version + the two refs) and **References**.

## What to cite (verified from the journal pages, 2026-09-07)
- **FoldX force field / web server** — Schymkowitz, J., Borg, J., Stricher, F., Nys, R., Rousseau, F., & Serrano,
  L. (2005). *The FoldX web server: an online force field.* **Nucleic Acids Research** 33(suppl_2), W382–W388.
  doi:10.1093/nar/gki387. *(verified: academic.oup.com/nar/article/33/suppl_2/W382/2505499)*
- **FoldX 5** (cite this for the v5.x binary we run) — Delgado, J., Radusky, L. G., Cianferoni, D., & Serrano, L.
  (2019). *FoldX 5.0: working with RNA, small molecules and a new graphical interface.* **Bioinformatics** 35(20),
  4168–4170. doi:10.1093/bioinformatics/btz184. *(verified: academic.oup.com/bioinformatics/article/35/20/4168/5381539)*

**Foundational ΔΔG-of-complexes paper — ADDED to the bib and cited in Methods** (this is the *most on-point*
paper for our application: FoldX ΔΔG on mutations at protein–protein interfaces): Guerois, R., Nielsen, J. E., &
Serrano, L. (2002). *Predicting changes in the stability of proteins and protein complexes: a study of more than
1000 mutations.* J. Mol. Biol. 320(2), 369–387. doi:10.1016/S0022-2836(02)00442-4. *(VERIFIED 2026-09-08 via
verify-references skill → Crossref: first-author/year/venue/volume/pages/title all ok, title ratio 1.000.)*
The three together are the correct set: Guerois 2002 = the ΔΔG method we use; Schymkowitz 2005 = the force
field/server; Delgado 2019 = the v5 binary we run. All three verified clean by Crossref.

RRID (reported, verify at scicrunch.org before use): **RRID:SCR_008522**.

## Do we cite the License Agreement? — No
The Academic License Agreement is a legal contract, **not** scientific literature; it is not cited in the paper.
Academic credit is given by citing the FoldX papers above (methods + references). (Confirmed by the standard
practice; the license only governs *use*, not citation.)

## License compliance (operational — matters for how the binary is handled)
The FoldX academic license is **free but non-transferable, non-redistributable, one copy** ("shall not permit any
third party to use the Software … agrees not to assign, transfer, … export … the Software"; "only one (1) copy").
Therefore:
- **Never commit the FoldX binary or rotabase.txt to git / push to GitHub** (this repo is public → redistribution
  = license breach). It is `.gitignore`d as a safety net and lives at `$SCRATCH/ftax/foldx/` (outside the repo).
- Only outputs (ΔΔG numbers in `foldx_*.csv`) go in the repo — those are our data, not the software.

## BibTeX (paste into latex/ when adding to the paper)
```bibtex
@article{schymkowitz2005foldx,
  title   = {The {FoldX} web server: an online force field},
  author  = {Schymkowitz, Joost and Borg, Jesper and Stricher, Francois and Nys, Robby and Rousseau, Frederic and Serrano, Luis},
  journal = {Nucleic Acids Research}, volume = {33}, number = {suppl_2}, pages = {W382--W388},
  year    = {2005}, doi = {10.1093/nar/gki387}
}
@article{delgado2019foldx5,
  title   = {{FoldX} 5.0: working with {RNA}, small molecules and a new graphical interface},
  author  = {Delgado, Javier and Radusky, Leandro G. and Cianferoni, Damiano and Serrano, Luis},
  journal = {Bioinformatics}, volume = {35}, number = {20}, pages = {4168--4170},
  year    = {2019}, doi = {10.1093/bioinformatics/btz184}
}
```
Suggested Methods sentence: "Physics ΔΔG_bind was computed with FoldX 5.<x> (Schymkowitz et al., 2005; Delgado et
al., 2019) via RepairPDB → BuildModel → AnalyseComplex (numberOfRuns=5)."
