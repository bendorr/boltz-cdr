"""Render the still images that stand in for the interactive overlay on GitHub.

The structural overlay is a py3Dmol viewer: JavaScript, which GitHub's notebook preview
strips, leaving an empty frame where the structures should be. So each overlay cell is
committed with a ray-traced still of the same scene attached as its output instead. Run the
cell and the live viewer replaces the still — it is a preview of the thing, never a
substitute for running it.

The scene is not re-derived here. The members are superposed and sliced to their CDR loops
by the same calls `ensemble_view` makes, and colored with the colors it assigns, so the
still shows the ensemble the viewer would show, in the same colors, one image per overlay
cell in the two notebooks.

Requires PyMOL on PATH (`brew install pymol` / `conda install -c conda-forge pymol-open-source`).

Usage (from the repository root):
    python notebooks/_build_overlay_stills.py
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, "src")

from boltz_cdr.cdr import annotate_vhh
from boltz_cdr.pdb_io import fetch_cif, load_models
from boltz_cdr.visualize import (
    _in_reference_frame,
    complex_to_pdb,
    ensemble_view,
    superpose_cdr_ensemble,
)

ASSETS = pathlib.Path("assets")

# One entry per distinct overlay scene. `notebooks` lists the notebooks whose overlay cell
# draws it — the two Colab notebooks share a scene because they share their cells — and
# `loader` returns the same members those cells draw.
STILLS = {
    "overlay_9kfw_20models.png": {
        "notebooks": ["cdr_ensemble_visualization.ipynb"],
        "loader": lambda: load_models(fetch_cif("9KFW", "data/pdb")),
    },
    "overlay_9kfw_10models.png": {
        "notebooks": ["colab_boltz_cdr.ipynb", "colab_boltz_cdr_long.ipynb"],
        "loader": lambda: load_models("data/examples/9kfw_10models.pdb.gz"),
    },
}

RAY_SETTINGS = """
set ray_trace_mode, 1
set ray_shadow, 0
set spec_reflect, 0
set spec_power, 0
set cartoon_loop_radius, 0.3
set ray_trace_gain, 0
set ambient, 0.3
set ray_trace_color, black
set cartoon_side_chain_helper, 1
set ray_opaque_background, 0
"""


def still_for(notebook: str) -> pathlib.Path:
    """The still that stands in for `notebook`'s overlay cell."""
    matches = [name for name, spec in STILLS.items() if notebook in spec["notebooks"]]
    if len(matches) != 1:
        msg = f"expected one still for {notebook}, found {matches}"
        raise RuntimeError(msg)
    path = ASSETS / matches[0]
    if not path.exists():
        msg = f"{path} is missing; run python notebooks/_build_overlay_stills.py"
        raise RuntimeError(msg)
    return path


def _n_members(cell) -> int:
    """Read the member count out of the viewer's own text/plain representation."""
    import re

    for output in cell.get("outputs", []):
        text = "".join(output.get("data", {}).get("text/plain", ""))
        match = re.search(r"<EnsembleView: (\d+) members>", text)
        if match:
            return int(match.group(1))
    msg = "the overlay cell produced no EnsembleView; there is nothing to stand in for"
    raise RuntimeError(msg)


def attach_still(cell, notebook: str) -> pathlib.Path:
    """Replace an executed overlay cell's output with the still.

    The cell is executed first and its real output thrown away, which is not wasted: it is
    the only check that the cell still runs, and the member count is read back out of it.
    What is committed is the still, so that the scene is visible on GitHub — and because
    outputs are replaced wholesale, running the cell drops the still and shows the live
    viewer.
    """
    import base64

    import nbformat

    n_members = _n_members(cell)
    png = still_for(notebook)
    cell.outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={
                "image/png": base64.b64encode(png.read_bytes()).decode(),
                "text/plain": (
                    f"<static preview of the {n_members}-member overlay: {png}. "
                    f"Run this cell for the interactive viewer.>"
                ),
            },
            metadata={"boltz_cdr": {"still": str(png), "members": n_members}},
        )
    ]
    return png


def _hex_to_rgb(color: str) -> list[float]:
    color = color.lstrip("#")
    return [int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _script(
    work: pathlib.Path, colors: list[str], subset: list[int], panels: dict[str, pathlib.Path]
) -> str:
    """The .pml for both panels of one still, rendered from a single shared camera.

    Two panels because one still cannot carry what the viewer does. Every member at once
    shows the spread; a handful with their side chains shows the contacts, which is what
    the legend checkboxes and the side-chain toggle are for and is unreadable with twenty
    members drawn over each other.
    """
    lines = [
        f"load {work / 'context.pdb'}, context",
        "hide everything",
        "show cartoon, context",
        "color gray60, context",
        "set cartoon_transparency, 0.35, context",
    ]
    for i, color in enumerate(colors):
        obj = f"m{i:02d}"
        lines += [
            f"load {work / f'{obj}.pdb'}, {obj}",
            f"set_color c{i:02d}, {_hex_to_rgb(color)}",
            f"color c{i:02d}, {obj}",
            f"cartoon loop, {obj}",
        ]

    everything = " or ".join(f"m{i:02d}" for i in range(len(colors)))
    chosen = " or ".join(f"m{i:02d}" for i in subset)
    lines += [
        "remove hydrogens",
        RAY_SETTINGS,
        # Orient once, on the loops, and never move again: the two panels have to be the
        # same view for the second to read as a subset of the first.
        f"orient ({everything})",
        f"zoom ({everything} or context), buffer=-6",
        "turn x, -12",
        f"show cartoon, ({everything})",
        f"png {panels['all'].resolve()}, width=1500, height=1150, dpi=200, ray=1",
        f"hide everything, ({everything})",
        f"show cartoon, ({chosen})",
        f"show sticks, ({chosen}) and (sidechain or name CA)",
        f"util.cnc ({chosen}) and (sidechain or name CA)",
        f"png {panels['subset'].resolve()}, width=1500, height=1150, dpi=200, ray=1",
    ]
    return "\n".join(lines) + "\n"


def _compose(panels: dict[str, pathlib.Path], titles: dict[str, str], out: pathlib.Path):
    """Lay the two ray-traced panels side by side and label them.

    Labelled because the still is what a GitHub reader sees in place of the viewer, and it
    has to say so itself — the caption is the only thing telling them the cell is live.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    for ax, key in zip(axes, ("all", "subset"), strict=True):
        # PyMOL centers the scene in a fixed canvas, leaving wide transparent margins that
        # would otherwise shrink the molecule to a third of the panel.
        panel = Image.open(panels[key])
        ax.imshow(panel.crop(panel.getbbox()))
        ax.set_axis_off()
        ax.set_title(titles[key], fontsize=9.5, color="#6e6e6e")
    fig.suptitle(
        "static preview of the interactive viewer — run the cell to rotate it, "
        "toggle side chains, and isolate members",
        fontsize=9, color="#8a8a8a", y=0.055,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    # The still is embedded in a committed notebook as well as saved here, so it is kept
    # small enough that two copies of it are not the largest thing in the repository.
    fig.savefig(out, dpi=96, transparent=True, bbox_inches="tight",
                pil_kwargs={"optimize": True})
    plt.close(fig)


def render(name: str, spec: dict) -> pathlib.Path:
    from boltz_cdr.visualize import similarity_order

    members = spec["loader"]()
    annotation = annotate_vhh(members[0].seq)

    # The colors the viewer assigns, taken from the viewer itself rather than recomputed,
    # so the still cannot drift from the cell it stands in for.
    colors = ensemble_view(members, annotation, controls=True).colors
    ensemble = superpose_cdr_ensemble(members, annotation)
    loop_residues = ensemble.residue_indices

    # Ends and middle of the similarity order: the two most different conformations in the
    # ensemble and one between them, rather than three arbitrary members.
    order = list(similarity_order(ensemble))
    subset = [order[0], order[len(order) // 2], order[-1]]

    work = pathlib.Path(tempfile.mkdtemp(prefix="overlay-still-"))
    try:
        (work / "context.pdb").write_text(complex_to_pdb(members[0]))
        for i, member in enumerate(members):
            moved = _in_reference_frame(member, members[0], annotation, "framework")
            (work / f"m{i:02d}.pdb").write_text(
                complex_to_pdb(moved, residue_subset=loop_residues)
            )

        panels = {key: work / f"panel_{key}.png" for key in ("all", "subset")}
        pml = work / "render.pml"
        pml.write_text(_script(work, colors, subset, panels))
        result = subprocess.run(
            ["pymol", "-cq", str(pml)], capture_output=True, text=True, check=False
        )
        missing = [str(p) for p in panels.values() if not p.exists()]
        if missing:
            msg = f"pymol did not write {missing}\n{result.stdout}\n{result.stderr}"
            raise RuntimeError(msg)

        out = ASSETS / name
        _compose(
            panels,
            {
                "all": f"all {len(members)} members — CDR loops on the shared framework",
                "subset": "3 members — the extremes of the ensemble, with side chains",
            },
            out,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB)  {len(members)} members, "
          f"subset {[int(i) for i in subset]}  <- {', '.join(spec['notebooks'])}")
    return out


def main() -> int:
    if shutil.which("pymol") is None:
        print("pymol not found on PATH; install it to regenerate the overlay stills")
        return 1
    ASSETS.mkdir(exist_ok=True)
    for name, spec in STILLS.items():
        render(name, spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
