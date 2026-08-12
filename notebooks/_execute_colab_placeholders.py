"""Execute the visualization cells of both Colab notebooks on the example ensemble.

The Colab notebooks are GPU runners: nothing else in them can be executed here, and their
outputs are meant to be produced by whoever runs them. The visualization section is the one
part worth committing already drawn, so section 9 is not a pair of blank cells in the
repository. This runs exactly those cells — the loader, the structural overlay and the
conformation landscape — with `EXAMPLE_ENSEMBLE` forced on, so they draw the twenty committed
NMR models instead of predictions that do not exist yet. Both notebooks are handled because they are
built from one set of cells and differ only in their prose.

The flag is flipped in memory only. What gets written back is the notebook with its source
unchanged (`EXAMPLE_ENSEMBLE = False`) and placeholder outputs attached, which is the state
that makes a real run overwrite the figures: someone running the notebook top to bottom in
Colab never turns the flag on and never sees the example data.

Usage (from the repository root):
    python notebooks/_build_example_ensemble.py     # once, writes data/examples/
    python notebooks/_execute_colab_placeholders.py

Requires `nbformat`, `nbclient` and `ipykernel`, which are development tools rather than
package dependencies and are not in `requirements.txt`.
"""

import pathlib
import sys

import nbformat
from nbclient import NotebookClient

from _build_overlay_stills import attach_still

NOTEBOOKS = [
    pathlib.Path("notebooks/colab_boltz_cdr.ipynb"),                # ships
    pathlib.Path("notebooks/colab_boltz_cdr_long.ipynb"),           # local, gitignored
]
FLAG = "EXAMPLE_ENSEMBLE = False"
CONFIG_MARKER = "DEMO_TARGET = TARGETS[0]"
CELL_MARKERS = (FLAG, "ensemble_view(", "plot_landscape(")   # the flag is set in one cell

# `%matplotlib inline` because the landscape cell ends on a print rather than `plt.show()`;
# Colab has the inline backend active from the start, a bare kernel does not.
PREAMBLE = '%matplotlib inline\nimport sys\nsys.path.insert(0, "src")\nsys.path.insert(0, "scripts")'


def _find(nb, marker: str):
    matches = [c for c in nb.cells if c.cell_type == "code" and marker in "".join(c.source)]
    if len(matches) != 1:
        msg = f"expected exactly one code cell containing {marker!r}, found {len(matches)}"
        raise RuntimeError(msg)
    return matches[0]


def _run_detached(client, nb, source: str) -> str:
    """Run `source` in the kernel without leaving a cell behind in the notebook."""
    cell = nbformat.v4.new_code_cell(source)
    nb.cells.insert(0, cell)
    try:
        client.execute_cell(cell, 0)
    finally:
        nb.cells.pop(0)
    return "".join("".join(o.get("text", "")) for o in cell.get("outputs", [])).strip()


def placeholders(notebook: pathlib.Path) -> None:
    nb = nbformat.read(notebook, as_version=4)
    loader, overlay, landscape = (_find(nb, marker) for marker in CELL_MARKERS)
    if FLAG not in "".join(loader.source):
        msg = f"{FLAG!r} is not in the loader cell; the notebook and this script disagree"
        raise RuntimeError(msg)

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(pathlib.Path.cwd())}},
        allow_errors=False,
    )
    with client.setup_kernel():
        interpreter = _run_detached(client, nb, "import sys; print(sys.executable)")
        if pathlib.Path(interpreter).resolve() != pathlib.Path(sys.executable).resolve():
            msg = (
                f"kernel interpreter {interpreter} is not the interpreter running this "
                f"script ({sys.executable}); the committed outputs would come from a "
                f"different environment"
            )
            raise RuntimeError(msg)
        print(f"kernel interpreter verified: {pathlib.Path(interpreter).name}")

        # The run configuration cell defines DEMO_TARGET, which the loader reads. Run it
        # detached so section 3 keeps its own outputs clear of this.
        _run_detached(client, nb, PREAMBLE)
        _run_detached(client, nb, "".join(_find(nb, CONFIG_MARKER).source))

        original = loader.source
        loader.source = original.replace(FLAG, FLAG.replace("False", "True"), 1)
        try:
            for cell in (loader, overlay, landscape):
                cell.outputs = []
                client.execute_cell(cell, nb.cells.index(cell))
                cell.metadata.pop("execution", None)
        finally:
            loader.source = original

    if not any("application/3dmoljs_load.v0" in o.get("data", {}) for o in overlay.outputs):
        msg = "the overlay cell produced no viewer; the still would stand in for nothing"
        raise RuntimeError(msg)
    if not any("image/png" in o.get("data", {}) for o in landscape.outputs):
        msg = "the landscape cell produced no figure"
        raise RuntimeError(msg)

    # GitHub strips the viewer's JavaScript, so the committed overlay output is a still of
    # the same scene; running the cell replaces it with the live viewer.
    still = attach_still(overlay, notebook.name)

    nbformat.write(nb, notebook)
    executed = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
    print(f"{notebook}: {executed} cells executed, overlay shows {still}, "
          f"{FLAG!r} restored in the source")


def main() -> int:
    example = pathlib.Path("data/examples/9kfw_20models.pdb.gz")
    if not example.exists():
        print(f"{example} not found; run notebooks/_build_example_ensemble.py first")
        return 1
    present = [n for n in NOTEBOOKS if n.exists()]
    if NOTEBOOKS[0] not in present:
        print(f"{NOTEBOOKS[0]} not found; run notebooks/_build_notebook.py first")
        return 1
    for notebook in present:
        placeholders(notebook)
    return 0


if __name__ == "__main__":
    sys.exit(main())
