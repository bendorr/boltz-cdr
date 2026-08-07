"""Execute `cdr_ensemble_visualization.ipynb` in place and save it with its outputs.

The demo notebook is committed with outputs so its figures are visible without running
anything, which is only honest if the committed outputs were produced by the committed
code. Regenerating and re-executing are therefore two steps of one routine:

    python notebooks/_build_demo_notebook.py
    python notebooks/_execute_demo_notebook.py

Run from the repository root. Requires `nbformat`, `nbclient`, and `ipykernel`, which are
development tools rather than package dependencies and are not in `requirements.txt`.
"""

import pathlib
import sys

import nbformat
from nbclient import NotebookClient

NOTEBOOK = pathlib.Path("notebooks/cdr_ensemble_visualization.ipynb")


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"{NOTEBOOK} not found; run notebooks/_build_demo_notebook.py first")
        return 1

    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(pathlib.Path.cwd())}},
        allow_errors=False,
    )
    print(f"executing {NOTEBOOK} ...")
    client.execute()

    # py3Dmol emits the same viewer twice, as text/html and under a private mimetype.
    # Every renderer that can run the viewer at all reads the HTML, so the duplicate is
    # half a megabyte of committed payload for nothing.
    dropped = 0
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            private = [k for k in data if k.startswith("application/3dmoljs")]
            if private and "text/html" in data:
                for key in private:
                    del data[key]
                    dropped += 1
    if dropped:
        print(f"dropped {dropped} duplicate py3Dmol mimetype payload(s)")

    n_out = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    nbformat.write(nb, NOTEBOOK)
    print(f"saved with outputs: {n_code} code cells produced {n_out} outputs")

    empty = [
        i for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and not c.get("outputs")
    ]
    if empty:
        print(f"note: code cells with no output: {empty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
