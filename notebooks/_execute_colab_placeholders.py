"""Execute the two visualization cells of `colab_boltz_cdr.ipynb` on the example ensemble.

The Colab notebook is a GPU runner: nothing else in it can be executed here, and its
outputs are meant to be produced by whoever runs it. The visualization section is the one
part worth committing already drawn, so section 9 is not a pair of blank cells in the
repository. This runs exactly those cells — the loader, the structural overlay and the
conformation landscape — with `EXAMPLE_ENSEMBLE` forced on, so they draw the ten committed
NMR models instead of predictions that do not exist yet.

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

NOTEBOOK = pathlib.Path("notebooks/colab_boltz_cdr.ipynb")
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


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"{NOTEBOOK} not found; run notebooks/_build_notebook.py first")
        return 1
    example = pathlib.Path("data/examples/9kfw_10models.pdb.gz")
    if not example.exists():
        print(f"{example} not found; run notebooks/_build_example_ensemble.py first")
        return 1

    nb = nbformat.read(NOTEBOOK, as_version=4)
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

    kinds = {
        "viewer": sum("application/3dmoljs_load.v0" in o.get("data", {})
                      for o in overlay.outputs),
        "figure": sum("image/png" in o.get("data", {}) for o in landscape.outputs),
    }
    print(f"placeholder outputs: {kinds['viewer']} interactive viewer, "
          f"{kinds['figure']} figure")
    if not all(kinds.values()):
        msg = f"expected one of each output, got {kinds}"
        raise RuntimeError(msg)

    nbformat.write(nb, NOTEBOOK)
    executed = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
    print(f"saved {NOTEBOOK} with {executed} executed cell(s); "
          f"{FLAG!r} restored in the source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
