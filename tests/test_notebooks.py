"""The committed notebooks are part of the deliverable, so their invariants are tested.

Both are committed with outputs. That is only useful if the outputs came from the committed
code, and — for the Colab runner, whose figures are placeholders drawn from example data —
only honest if the flag that produced them is off in the source, so that running the
notebook overwrites them.
"""

import gzip
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COLAB = ROOT / "notebooks" / "colab_boltz_cdr.ipynb"
DEMO = ROOT / "notebooks" / "cdr_ensemble_visualization.ipynb"
EXAMPLE = ROOT / "data" / "examples" / "9kfw_10models.pdb.gz"


def cells(path: Path):
    return json.loads(path.read_text())["cells"]


def source(cell) -> str:
    return "".join(cell["source"])


def test_example_ensemble_ships_with_the_repository():
    """The placeholder figures are only reproducible from a clone if their input is in it."""
    assert EXAMPLE.exists(), f"{EXAMPLE} is missing; run _build_example_ensemble.py"
    assert EXAMPLE.stat().st_size < 1_000_000, "an example that big does not belong in git"

    text = gzip.decompress(EXAMPLE.read_bytes()).decode()
    assert text.count("\nMODEL ") == 10, "the example is a 10-model ensemble"


@pytest.mark.parametrize("path", [COLAB, DEMO])
def test_notebook_is_committed_with_outputs(path: Path):
    executed = [c for c in cells(path) if c["cell_type"] == "code" and c.get("outputs")]
    assert executed, f"{path.name} carries no outputs; it must be committed executed"


def test_colab_placeholder_flag_is_off_in_the_committed_source():
    """Off in the source, on in whatever produced the outputs: a real run overwrites them."""
    loaders = [c for c in cells(COLAB) if "EXAMPLE_ENSEMBLE = " in source(c)]
    assert len(loaders) == 1
    assert "EXAMPLE_ENSEMBLE = False" in source(loaders[0])
    assert "PLACEHOLDER" in json.dumps(loaders[0]["outputs"]), (
        "the committed output must say it is a placeholder"
    )


def test_colab_visualization_cells_carry_placeholder_figures():
    viz = [c for c in cells(COLAB) if c["cell_type"] == "code"
           and ("ensemble_view(" in source(c) or "plot_landscape(" in source(c))]
    assert len(viz) == 2, "the overlay cell and the landscape cell"

    published = {mime for cell in viz for out in cell["outputs"] for mime in out.get("data", {})}
    assert "image/png" in published, "the landscape figure must be saved in the notebook"
    assert "application/3dmoljs_load.v0" in published, (
        "the overlay must keep the mimetype VSCode drives the interactive viewer from"
    )
