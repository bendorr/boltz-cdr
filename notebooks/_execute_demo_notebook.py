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


def _verify_interpreter(nb, client) -> str:
    """Run a probe cell to confirm the kernel is this interpreter, then discard it.

    `kernel_name="python3"` resolves through the Jupyter kernelspec search path, which
    can easily point at an unrelated environment. Committing outputs produced by the
    wrong interpreter would be silent and hard to notice, so the interpreter is checked
    rather than assumed.
    """
    probe = nbformat.v4.new_code_cell("import sys; print(sys.executable)")
    nb.cells.insert(0, probe)
    try:
        client.execute_cell(probe, 0)
        executed_by = "".join(
            "".join(o.get("text", "")) for o in probe.get("outputs", [])
        ).strip()
    finally:
        nb.cells.pop(0)

    if pathlib.Path(executed_by).resolve() != pathlib.Path(sys.executable).resolve():
        msg = (
            f"kernel interpreter {executed_by} is not the interpreter running this "
            f"script ({sys.executable}). The committed outputs would come from a "
            f"different environment. Install a kernelspec for this environment with "
            f"`python -m ipykernel install --user --name <name>` and pass it explicitly."
        )
        raise RuntimeError(msg)
    return executed_by


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
    with client.setup_kernel():
        interpreter = _verify_interpreter(nb, client)
        print(f"kernel interpreter verified: {pathlib.Path(interpreter).name}")
        print(f"executing {NOTEBOOK} ...")
        for index, cell in enumerate(nb.cells):
            client.execute_cell(cell, index)

    # The viewer payload is deliberately kept under both `application/3dmoljs_load.v0`
    # and `text/html`. Stripping the former halves the file size but leaves VSCode
    # rendering a static image instead of an interactive viewer, so it stays.
    viewers = sum(
        1
        for cell in nb.cells
        for output in cell.get("outputs", [])
        if "application/3dmoljs_load.v0" in output.get("data", {})
    )
    print(f"{viewers} interactive viewer output(s) retained")

    # Drop per-cell execution timestamps. They change on every run and would otherwise
    # make each re-execution a diff of the whole notebook, obscuring whether any output
    # actually changed. (py3Dmol's viewer id is randomized per run and still churns.)
    for cell in nb.cells:
        cell.get("metadata", {}).pop("execution", None)

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
