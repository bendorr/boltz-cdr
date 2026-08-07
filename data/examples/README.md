# Example ensemble

`9kfw_10models.pdb.gz` — ten models sampled across the 20-model solution-NMR ensemble of
[PDB 9KFW](https://www.rcsb.org/structure/9KFW), an anti-OX40 nanobody. Waters, ligands,
hydrogens and alternate conformations are stripped; models 1, 3, 5 … 19 of the deposition
are kept and renumbered 1–10.

This is the only structural data committed to the repository. Everything else under
`data/` is fetched from RCSB on demand by `scripts/00_fetch_targets.py` and is gitignored.
This one file is committed so that the two visualization cells of
`notebooks/colab_boltz_cdr.ipynb` have something to draw without a GPU: it is what produced
the placeholder figures saved in that notebook, and a clone can redraw them with

    python notebooks/_execute_colab_placeholders.py

An NMR ensemble is used because it is a *measured* conformational ensemble of one molecule
— real CDR loop flexibility, with no prediction and no model weights involved.

Regenerate the file itself with:

    python notebooks/_build_example_ensemble.py
