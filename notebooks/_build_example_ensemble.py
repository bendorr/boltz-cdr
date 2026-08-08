"""Write the example ensemble the Colab notebook falls back to.

`colab_boltz_cdr.ipynb` is committed with its two visualization cells already executed, so
the figures are visible in the repository without a GPU, a Boltz-2 run, or a Colab session.
Those placeholder figures need structures that ship with the repository, which is what this
writes: the complete 20-model NMR ensemble of 9KFW, the same entry and the same ensemble the
CPU demo notebook uses, so the two notebooks illustrate the same structures.

All twenty models rather than a slice, because twenty is the whole deposition — and it is
also the ceiling: no nanobody or VHH solution-NMR entry in the PDB deposits more than twenty
models. Gzipped PDB rather than mmCIF because the file is committed, and a clone has to be
able to redraw the figures without fetching anything.

Usage (from the repository root):
    python notebooks/_build_example_ensemble.py
"""

import gzip
import pathlib
import sys

sys.path.insert(0, "src")

import gemmi

from boltz_cdr.pdb_io import fetch_cif, load_models

PDB_ID = "9KFW"
STRIDE = 1
N_MODELS = 20       # the whole deposition
OUT = pathlib.Path("data/examples/9kfw_20models.pdb.gz")


def main() -> int:
    source = fetch_cif(PDB_ID, "data/pdb")
    structure = gemmi.read_structure(str(source))
    structure.setup_entities()
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    structure.remove_ligands_and_waters()
    structure.remove_empty_chains()

    wanted = list(structure)[::STRIDE][:N_MODELS]
    if len(wanted) < N_MODELS:
        msg = f"{PDB_ID} has too few models to take {N_MODELS} at stride {STRIDE}"
        raise RuntimeError(msg)
    kept = [model.num for model in wanted]

    # Rebuilt rather than pruned in place, and renumbered 1..10, so the viewer legend reads
    # "model 1 … model 10" instead of the deposited odd numbers, which would suggest models
    # are missing rather than sampled.
    sliced = gemmi.Structure()
    sliced.name = f"{PDB_ID}_{N_MODELS}models"
    sliced.spacegroup_hm = "P 1"
    sliced.cell = gemmi.UnitCell(1, 1, 1, 90, 90, 90)
    for new_num, model in enumerate(wanted, start=1):
        clone = model.clone()
        clone.num = new_num
        sliced.add_model(clone)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(gzip.compress(sliced.make_pdb_string().encode(), mtime=0))

    check = load_models(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"deposited models kept: {kept}")
    print(f"reads back as {len(check)} models, {check[0].n_res} residues, "
          f"chain {check[0].chain_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
