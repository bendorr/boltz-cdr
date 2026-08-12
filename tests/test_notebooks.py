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
EXAMPLE = ROOT / "data" / "examples" / "9kfw_20models.pdb.gz"
NOTEBOOKS = [COLAB, DEMO]

# The long-form Colab notebook is gitignored working notes, absent from a clone. Where it is
# present it must still agree with the shipped one; a clone simply skips these tests.
LONG = ROOT / "notebooks" / "colab_boltz_cdr_long.ipynb"
LOCAL_ONLY = [LONG]
needs_long = pytest.mark.skipif(not LONG.exists(), reason="long-form notebook is local only")


def cells(path: Path):
    return json.loads(path.read_text())["cells"]


def source(cell) -> str:
    return "".join(cell["source"])


def test_example_ensemble_ships_with_the_repository():
    """The placeholder figures are only reproducible from a clone if their input is in it."""
    assert EXAMPLE.exists(), f"{EXAMPLE} is missing; run _build_example_ensemble.py"
    assert EXAMPLE.stat().st_size < 1_000_000, "an example that big does not belong in git"

    text = gzip.decompress(EXAMPLE.read_bytes()).decode()
    assert text.count("\nMODEL ") == 20, "the example is the complete 20-model ensemble"


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebook_is_committed_with_outputs(path: Path):
    executed = [c for c in cells(path) if c["cell_type"] == "code" and c.get("outputs")]
    assert executed, f"{path.name} carries no outputs; it must be committed executed"


@pytest.mark.parametrize("path", [COLAB])
def test_colab_placeholder_flag_is_off_in_the_committed_source(path: Path):
    """Off in the source, on in whatever produced the outputs: a real run overwrites them."""
    loaders = [c for c in cells(path) if "EXAMPLE_ENSEMBLE = " in source(c)]
    assert len(loaders) == 1
    assert "EXAMPLE_ENSEMBLE = False" in source(loaders[0])
    assert "PLACEHOLDER" in json.dumps(loaders[0]["outputs"]), (
        "the committed output must say it is a placeholder"
    )


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_overlay_cells_are_committed_as_a_still(path: Path):
    """GitHub strips the viewer's JavaScript, so a picture of the scene is committed.

    Only the output is swapped — the source still calls `view.show()`, so running the cell
    replaces the still with the live viewer, which is the whole point of a still.
    """
    overlays = [c for c in cells(path)
                if c["cell_type"] == "code" and "ensemble_view(" in source(c)]
    assert len(overlays) == 1

    published = {m for out in overlays[0]["outputs"] for m in out.get("data", {})}
    assert "image/png" in published, "the overlay must be committed as a still"
    assert "application/3dmoljs_load.v0" not in published, (
        "the stripped viewer must not be committed alongside the still"
    )
    assert "view.show()" in source(overlays[0]), "running the cell must still draw the viewer"

    note = json.dumps(overlays[0]["outputs"])
    assert "static preview" in note and "assets/" in note, (
        "the still must name itself and its source asset"
    )


@pytest.mark.parametrize("path", [COLAB])
def test_colab_landscape_carries_a_placeholder_figure(path: Path):
    landscapes = [c for c in cells(path)
                  if c["cell_type"] == "code" and "plot_landscape(" in source(c)]
    assert len(landscapes) == 1
    published = {m for out in landscapes[0]["outputs"] for m in out.get("data", {})}
    assert "image/png" in published


@needs_long
def test_the_two_colab_notebooks_share_every_code_cell():
    """They are built from one list of cells; only the prose is allowed to differ."""
    short = [source(c) for c in cells(COLAB) if c["cell_type"] == "code"]
    long_form = [source(c) for c in cells(LONG) if c["cell_type"] == "code"]
    assert short == long_form, "the shipped notebook must run exactly the same code"

    def prose(path):
        return sum(len(source(c)) for c in cells(path) if c["cell_type"] == "markdown")

    assert prose(COLAB) < 0.75 * prose(LONG), "the shipped notebook must be the short one"


@pytest.mark.parametrize("path", LOCAL_ONLY, ids=lambda p: p.name)
def test_local_only_notebooks_are_not_tracked_by_git(path: Path):
    """Shipping these would put three runnable Colab notebooks, one with a personal Drive
    path in it, into a clone."""
    import subprocess

    if not path.exists():
        pytest.skip(f"{path.name} is local only")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert tracked.returncode != 0, f"{path.name} must stay gitignored"


def test_the_shipped_notebook_writes_to_drive_under_a_generic_path():
    """Predictions go to Drive so a dead Colab session does not take them with it.

    `MyDrive` is the same literal for every Google account, so the path resolves in any
    reader's Drive rather than naming anyone's. What must not appear is a personal
    identifier — an email, a numeric Drive id, a home directory.
    """
    assignments = [
        line for cell in cells(COLAB) for line in source(cell).splitlines()
        if line.startswith("RESULTS = ") or line.startswith("DRIVE_ROOT = ")
    ]
    assert any("/content/drive/MyDrive" in line for line in assignments), assignments

    text = json.dumps(cells(COLAB))
    for leak in ("/Users/", "/home/", "gmail", "drive.google.com/drive/u/"):
        assert leak not in text, f"{leak} must not appear in a shipped notebook"


def test_every_gpu_stage_can_resume():
    """A crashed session is resumed by re-running the cells, not by starting over.

    Each stage skips arms whose structures are already on disk, which is only useful if the
    notebook actually invokes the stages against the same results tree every time.
    """
    stage_cells = [
        source(c) for c in cells(COLAB)
        if c["cell_type"] == "code" and "!python scripts/0" in source(c)
    ]
    predicting = [s for s in stage_cells if any(
        n in s for n in ("01_global_dock", "02_arm_a", "03_arm_b")
    ) and not s.lstrip().startswith("#")]
    assert len(predicting) == 3, f"expected the three GPU stages, found {len(predicting)}"
    for cell in predicting:
        assert "--results {RESULTS}" in cell, cell


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebooks_carry_no_local_paths(path: Path):
    """Committed outputs are easy to leak a working directory through."""
    text = path.read_text()
    assert "/Users/" not in text and "/home/" not in text
