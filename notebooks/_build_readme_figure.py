"""Render the banner at the top of the README.

Three panels of the same ensemble, left to right: the CDR loops of three members with their
side chains, the conformation landscape as a surface, and the same landscape as a contour
map. It is the whole idea of the repository in one image — loops that move, and a reduced
description of how they move — and every panel is drawn by the same code the notebooks call,
so the banner cannot advertise something the package does not do.

Titles are dropped, on both the figure and the panels, and so is the fill/ring key: a banner
carries no argument, the README underneath it does. Axis labels and the colorbar stay,
because a picture of a landscape with unlabelled axes is decoration rather than a figure.

Requires PyMOL on PATH, as `_build_overlay_stills` does.

Usage (from the repository root):
    python notebooks/_build_readme_figure.py
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, "src")

import numpy as np

from _build_overlay_stills import ASSETS, RAY_SETTINGS, _hex_to_rgb
from boltz_cdr.cdr import annotate_vhh
from boltz_cdr.metrics.interface import sasa
from boltz_cdr.pdb_io import load_models
from boltz_cdr.visualize import (
    _in_reference_frame,
    complex_to_pdb,
    conformation_landscape,
    ensemble_view,
    extreme_members,
    plot_landscape,
    superpose_cdr_ensemble,
)

EXAMPLE = "data/examples/9kfw_20models.pdb.gz"
OUT = ASSETS / "readme_banner.png"


def _overlay_panel(work: pathlib.Path, members, annotation, colors, subset) -> pathlib.Path:
    """The three-member overlay, rendered exactly as the notebook still renders it."""
    loop_residues = superpose_cdr_ensemble(members, annotation).residue_indices
    (work / "context.pdb").write_text(complex_to_pdb(members[0]))
    for i in subset:
        moved = _in_reference_frame(members[i], members[0], annotation, "framework")
        (work / f"m{i:02d}.pdb").write_text(
            complex_to_pdb(moved, residue_subset=loop_residues)
        )

    out = work / "overlay.png"
    lines = [
        f"load {work / 'context.pdb'}, context",
        "hide everything",
        "show cartoon, context",
        "color gray60, context",
        "set cartoon_transparency, 0.35, context",
    ]
    for i in subset:
        obj = f"m{i:02d}"
        lines += [
            f"load {work / f'{obj}.pdb'}, {obj}",
            f"set_color c{i:02d}, {_hex_to_rgb(colors[i])}",
            f"color c{i:02d}, {obj}",
            f"cartoon loop, {obj}",
            f"show cartoon, {obj}",
            f"show sticks, {obj} and (sidechain or name CA)",
            f"util.cnc {obj} and (sidechain or name CA)",
        ]
    drawn = " or ".join(f"m{i:02d}" for i in subset)
    lines += [
        "remove hydrogens",
        RAY_SETTINGS,
        f"orient ({drawn})",
        f"zoom ({drawn} or context), buffer=-6",
        "turn x, -12",
        f"png {out.resolve()}, width=1500, height=1150, dpi=200, ray=1",
    ]
    pml = work / "banner.pml"
    pml.write_text("\n".join(lines) + "\n")
    result = subprocess.run(["pymol", "-cq", str(pml)], capture_output=True, text=True,
                            check=False)
    if not out.exists():
        msg = f"pymol did not write {out}\n{result.stdout}\n{result.stderr}"
        raise RuntimeError(msg)
    return out


def _landscape_panels(work: pathlib.Path, members, annotation, colors) -> pathlib.Path:
    """Both landscape panels, with every title and the key removed."""
    import matplotlib.pyplot as plt

    cdr_atoms = members[0].atom_mask_for_residues(annotation.all_indices)
    values = np.array([
        sasa(m.coords, m.atom_elements, n_points=96)[cdr_atoms].sum() for m in members
    ])
    landscape, projection, _ = conformation_landscape(
        members, annotation, values, method="pca"
    )
    fig, (ax3d, ax2d) = plot_landscape(
        landscape, projection,
        value_label="CDR solvent-accessible surface (A$^2$)",
        title=None,                      # no suptitle
        point_colors=colors,
    )
    ax2d.set_title("")                   # no panel title
    for legend in list(fig.legends):     # no fill/ring key
        legend.remove()

    # The z-axis and the colorbar name the same quantity, and with the panels butted
    # together for a banner the z-label runs into the contour panel's y-label. The colorbar
    # keeps the name; the surface keeps its numbers.
    ax3d.set_zlabel("")
    ax3d.set_box_aspect(None, zoom=0.95)
    fig.tight_layout()

    out = work / "landscape.png"
    fig.savefig(out, dpi=150, transparent=True, bbox_inches="tight",
                pil_kwargs={"optimize": True})
    plt.close(fig)
    return out


def _compose(overlay: pathlib.Path, landscape: pathlib.Path, out: pathlib.Path) -> None:
    """One row, equal heights, widths in proportion to each panel's own aspect ratio."""
    import matplotlib.pyplot as plt
    from PIL import Image

    images = []
    for path in (overlay, landscape):
        image = Image.open(path)
        images.append(image.crop(image.getbbox()))   # drop the transparent margin

    aspects = [image.width / image.height for image in images]
    height = 5.6
    fig, axes = plt.subplots(
        1, 2, figsize=(height * sum(aspects) * 1.02, height),
        gridspec_kw={"width_ratios": aspects, "wspace": 0.02},
    )
    for ax, image in zip(axes, images, strict=True):
        ax.imshow(image)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.004, right=0.996, top=0.99, bottom=0.01)
    # A README image renders about 900 px wide, so resolution past ~2400 px buys nothing
    # and costs a megabyte in the repository.
    fig.savefig(out, dpi=115, transparent=True, pil_kwargs={"optimize": True})
    plt.close(fig)


def main() -> int:
    if shutil.which("pymol") is None:
        print("pymol not found on PATH; install it to regenerate the README banner")
        return 1
    if not pathlib.Path(EXAMPLE).exists():
        print(f"{EXAMPLE} not found; run notebooks/_build_example_ensemble.py first")
        return 1

    members = load_models(EXAMPLE)
    annotation = annotate_vhh(members[0].seq)
    colors = ensemble_view(members, annotation, controls=True).colors
    subset = extreme_members(superpose_cdr_ensemble(members, annotation))

    work = pathlib.Path(tempfile.mkdtemp(prefix="readme-banner-"))
    try:
        overlay = _overlay_panel(work, members, annotation, colors, subset)
        landscape = _landscape_panels(work, members, annotation, colors)
        ASSETS.mkdir(exist_ok=True)
        _compose(overlay, landscape, OUT)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"{OUT}  ({OUT.stat().st_size / 1024:.0f} KB)  "
          f"{len(members)} members, {len(subset)} drawn with side chains")
    return 0


if __name__ == "__main__":
    sys.exit(main())
