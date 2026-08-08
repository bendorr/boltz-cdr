# Example ensemble

`9kfw_20models.pdb.gz` — the complete 20-model solution-NMR ensemble of
[PDB 9KFW](https://www.rcsb.org/structure/9KFW), an anti-OX40 nanobody. Waters, ligands,
hydrogens and alternate conformations are stripped; nothing else is removed.

Twenty is the whole deposition, and it is also the ceiling for this molecule class: no
nanobody or VHH solution-NMR entry in the PDB deposits more than twenty models (checked
against the RCSB search API — the entries that come back are 7V0V, 6K2I, 9KFW, 8UEK, 6IYN,
7EH3, 1G9E and 1IEH, topping out at twenty).

This is the only structural data committed to the repository. Everything else under `data/`
is fetched from RCSB on demand by `scripts/00_fetch_targets.py` and is gitignored. This one
file is committed so that the two visualization cells of `notebooks/colab_boltz_cdr.ipynb`
have something to draw without a GPU: it is what produced the placeholder figures saved in
that notebook, and a clone can redraw them with

    python notebooks/_execute_colab_placeholders.py

`notebooks/cdr_ensemble_visualization.ipynb` draws the same twenty models, fetched from RCSB
rather than read from here, so the two notebooks illustrate the same structures.

An NMR ensemble is used because it is a *measured* conformational ensemble of one molecule
— real CDR loop flexibility, with no prediction and no model weights involved.

Regenerate the file itself with:

    python notebooks/_build_example_ensemble.py
