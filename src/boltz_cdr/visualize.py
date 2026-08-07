"""Visualization of CDR conformational ensembles.

Two views of the same ensemble.

**Structural** (`ensemble_view`) overlays the CDR loops of every ensemble member on a
single shared framework, rendered with py3Dmol and zoomed on the paratope. The framework
and antigen are drawn once; only the loops are drawn per member, so the spread visible in
the viewer is loop conformation rather than rigid-body pose.

**Landscape** (`conformation_landscape`) projects the same loops to two dimensions and
raises a third axis over them — model confidence, DockQ, or any other per-structure
quantity — giving a surface analogous to an energy landscape, with the reaction coordinate
replaced by a reduced description of loop conformation.

Both operate on an arbitrary number of structures. The projection is PCA or classical MDS
implemented directly in numpy, so no additional dependency is required; py3Dmol is
imported lazily and is only needed for the structural view.

A caution on reading the landscape. With the sample counts typical of a single prediction
run (tens of structures), the fitted surface is a smoothing of sparse scattered data, not
a measured free-energy surface. Regions of the plane far from any structure are masked
rather than extrapolated, and the underlying points are always drawn on top of the surface
so that the density of evidence behind any feature is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boltz_cdr.cdr import CDR_NAMES, CDRAnnotation
from boltz_cdr.metrics.correspondence import (
    BACKBONE,
    build_correspondence,
    matched_atoms,
    subset_correspondence,
)
from boltz_cdr.metrics.rmsd import apply_transform, kabsch
from boltz_cdr.pdb_io import Complex


@dataclass
class EnsembleCoordinates:
    """CDR coordinates of every ensemble member, on a common frame."""

    features: np.ndarray  # (n_structures, n_atoms * 3) flattened, superposed
    cdr_coords: np.ndarray  # (n_structures, n_atoms, 3)
    reference: Complex  # the structure everything was superposed onto
    residue_indices: np.ndarray  # CDR residue indices in the reference
    atom_names: tuple[str, ...]
    n_dropped: int  # residues excluded for not being present in every member

    @property
    def n_structures(self) -> int:
        return len(self.features)


def superpose_cdr_ensemble(
    structures: list[Complex],
    annotation: CDRAnnotation,
    *,
    cdrs: tuple[str, ...] = CDR_NAMES,
    atom_names: tuple[str, ...] = BACKBONE,
    align_on: str = "framework",
) -> EnsembleCoordinates:
    """Superpose an ensemble on a common frame and extract CDR coordinates.

    Parameters
    ----------
    align_on
        ``"framework"`` superposes on the antibody's non-CDR residues, so the spread that
        remains is loop conformation alone. ``"antigen"`` superposes on the target
        instead, so the spread also includes rigid-body placement of the binder. The
        distinction matters: the two are separate error modes, and which one a picture is
        showing should be a deliberate choice rather than an accident of alignment.

    Residues or atoms absent from any member are dropped from all of them, so the feature
    matrix is rectangular and every column means the same thing in every row.
    """
    if len(structures) < 1:
        msg = "need at least one structure"
        raise ValueError(msg)
    if align_on not in {"framework", "antigen"}:
        msg = f"align_on must be 'framework' or 'antigen', got {align_on!r}"
        raise ValueError(msg)

    reference = structures[0]
    loop_residues = (
        np.concatenate([annotation[c] for c in cdrs]) if cdrs else annotation.all_indices
    )
    loop_residues = np.unique(loop_residues)

    # Residues present, with all requested atoms, in every member.
    usable = set(loop_residues.tolist())
    correspondences = []
    for structure in structures:
        corr = build_correspondence(structure, reference)
        correspondences.append(corr)
        present = set(corr.ab_b.tolist())
        usable &= present
    common = np.array(sorted(usable), dtype=int)
    n_dropped = len(loop_residues) - len(common)
    if len(common) == 0:
        msg = "no CDR residues are present in every ensemble member"
        raise ValueError(msg)

    coords = []
    for structure, corr in zip(structures, correspondences, strict=True):
        if align_on == "framework":
            anchor = np.setdiff1d(corr.ab_b, loop_residues)
            src, dst = subset_correspondence(corr.ab_b, corr.ab_a, anchor)
            x_src, x_dst = matched_atoms(
                reference.antibody, src, structure.antibody, dst, atom_names
            )
        else:
            x_src, x_dst = matched_atoms(
                reference.antigen, corr.ag_b, structure.antigen, corr.ag_a, atom_names
            )
        if len(x_src) < 3:  # noqa: PLR2004
            msg = f"too few {align_on} atoms to superpose ({len(x_src)})"
            raise ValueError(msg)

        # Transform taking this structure onto the reference frame.
        rot, trans = kabsch(x_dst, x_src)

        loop_ref, loop_this = subset_correspondence(corr.ab_b, corr.ab_a, common)
        _, x_loop = matched_atoms(
            reference.antibody, loop_ref, structure.antibody, loop_this, atom_names
        )
        coords.append(apply_transform(x_loop, rot, trans))

    lengths = {len(c) for c in coords}
    if len(lengths) != 1:
        msg = f"ensemble members yielded different CDR atom counts: {sorted(lengths)}"
        raise ValueError(msg)

    stacked = np.stack(coords)
    return EnsembleCoordinates(
        features=stacked.reshape(len(structures), -1),
        cdr_coords=stacked,
        reference=reference,
        residue_indices=common,
        atom_names=tuple(atom_names),
        n_dropped=n_dropped,
    )


# ------------------------------------------------------------------- projection


@dataclass
class Projection:
    """A 2D embedding of the ensemble."""

    xy: np.ndarray  # (n_structures, 2)
    method: str
    explained_variance: np.ndarray | None  # PCA only: fraction on each axis
    axis_labels: tuple[str, str]


def project_2d(
    ensemble: EnsembleCoordinates, *, method: str = "pca"
) -> Projection:
    """Reduce CDR conformations to two dimensions.

    ``pca`` is the default: deterministic, linear, and interpretable — the axes are the
    dominant modes of loop displacement, and the reported explained variance says how much
    of the ensemble's spread the picture actually captures. ``mds`` performs classical
    multidimensional scaling on pairwise CDR RMSD, which preserves structural distance
    more faithfully when the ensemble is not well described by two linear modes.
    """
    x = ensemble.features
    n = len(x)
    if n < 2:  # noqa: PLR2004
        msg = "need at least 2 structures to project"
        raise ValueError(msg)

    if method == "pca":
        centered = x - x.mean(axis=0, keepdims=True)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        xy = centered @ vt[:2].T
        variance = singular**2
        fraction = variance[:2] / variance.sum() if variance.sum() > 0 else np.zeros(2)
        return Projection(
            xy=xy,
            method="pca",
            explained_variance=fraction,
            axis_labels=(
                f"PC1 ({fraction[0]:.0%} of variance)",
                f"PC2 ({fraction[1]:.0%} of variance)",
            ),
        )

    if method == "mds":
        distance = pairwise_rmsd(ensemble)
        xy = classical_mds(distance)
        return Projection(
            xy=xy,
            method="mds",
            explained_variance=None,
            axis_labels=("MDS 1 (Å)", "MDS 2 (Å)"),
        )

    msg = f"unknown projection method {method!r}; expected 'pca' or 'mds'"
    raise ValueError(msg)


def pairwise_rmsd(ensemble: EnsembleCoordinates) -> np.ndarray:
    """(n, n) RMSD between ensemble members over the already-superposed CDR atoms."""
    coords = ensemble.cdr_coords
    diff = coords[:, None, :, :] - coords[None, :, :, :]
    return np.sqrt((diff**2).sum(axis=-1).mean(axis=-1))


def classical_mds(distance: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Torgerson classical MDS. Pure numpy, so no scikit-learn dependency."""
    n = len(distance)
    squared = distance**2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    values = np.clip(eigenvalues[order], 0, None)
    return eigenvectors[:, order] * np.sqrt(values)


# -------------------------------------------------------------------- landscape


@dataclass
class Landscape:
    """A smoothed surface over a 2D projection."""

    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_z: np.ndarray  # NaN where no structure is close enough to support a value
    xy: np.ndarray
    z: np.ndarray
    bandwidth: float
    coverage: float  # fraction of the grid that is supported by data


def smooth_landscape(
    xy: np.ndarray,
    z: np.ndarray,
    *,
    resolution: int = 90,
    bandwidth: float | None = None,
    support_radius: float | None = None,
    padding: float = 0.10,
    clip_to_hull: bool = False,
) -> Landscape:
    """Nadaraya-Watson Gaussian kernel regression of `z` over the plane.

    Kernel regression rather than interpolation. An interpolating spline or RBF forced
    through tens of scattered points will oscillate between them and invent structure that
    the data do not support; a kernel average cannot exceed the range of the observations
    and degrades gracefully to a flat surface as the bandwidth grows.

    Grid nodes further than `support_radius` from every observation are set to NaN rather
    than filled, so the surface stops where the evidence does instead of extrapolating
    across unsampled regions of conformational space.

    `clip_to_hull` additionally restricts the surface to the convex hull of the
    observations. It is off by default: the hull is a polygon, so it imposes straight
    boundaries that read as rendering artifacts on what is otherwise a smooth field, and
    the distance criterion already excludes everything the hull would.
    """
    xy = np.asarray(xy, dtype=float)
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z) & np.isfinite(xy).all(axis=1)
    xy, z = xy[finite], z[finite]
    if len(z) < 3:  # noqa: PLR2004
        msg = f"need at least 3 structures with finite values, got {len(z)}"
        raise ValueError(msg)

    if bandwidth is None:
        bandwidth = _default_bandwidth(xy)
    if support_radius is None:
        support_radius = 2.5 * bandwidth

    span = xy.max(axis=0) - xy.min(axis=0)
    span[span == 0] = 1.0
    lo = xy.min(axis=0) - padding * span
    hi = xy.max(axis=0) + padding * span
    gx, gy = np.meshgrid(
        np.linspace(lo[0], hi[0], resolution), np.linspace(lo[1], hi[1], resolution)
    )
    nodes = np.column_stack([gx.ravel(), gy.ravel()])

    d2 = ((nodes[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1)
    weights = np.exp(-0.5 * d2 / bandwidth**2)
    total = weights.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = (weights @ z) / total
    values[total <= 1e-12] = np.nan  # noqa: PLR2004

    unsupported = np.sqrt(d2.min(axis=1)) > support_radius
    values[unsupported] = np.nan

    if clip_to_hull:
        values[_outside_hull(nodes, xy)] = np.nan

    grid_z = values.reshape(gx.shape)
    return Landscape(
        grid_x=gx,
        grid_y=gy,
        grid_z=grid_z,
        xy=xy,
        z=z,
        bandwidth=float(bandwidth),
        coverage=float(np.isfinite(grid_z).mean()),
    )


def _outside_hull(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Boolean mask of grid nodes lying outside the convex hull of `points`.

    Degenerate inputs — fewer than three points, or points that are collinear — have no
    2D hull; the mask is then all-False and the distance criterion alone governs.
    """
    if len(points) < 3:  # noqa: PLR2004
        return np.zeros(len(nodes), dtype=bool)
    try:
        from scipy.spatial import Delaunay

        return Delaunay(points).find_simplex(nodes) < 0
    except Exception:
        return np.zeros(len(nodes), dtype=bool)


def _default_bandwidth(xy: np.ndarray) -> float:
    """Kernel bandwidth: the 75th percentile of nearest-neighbor separation, floored at
    5 % of the diagonal of the ensemble's extent.

    Two failure modes to avoid, and neither a Silverman/Scott rule nor a plain
    nearest-neighbor rule avoids both.

    A Silverman bandwidth, ``sigma * n**(-1/6)``, is derived assuming a unimodal
    distribution. Prediction ensembles are frequently multimodal — a cluster of similar
    conformations plus a few outliers — and there ``sigma`` reflects the separation
    *between* modes, giving a bandwidth wide enough to bridge the empty space between
    them and paint a surface over conformations nothing sampled.

    A median nearest-neighbor bandwidth has the opposite problem: ensembles routinely
    contain near-duplicate structures, which drives the median toward zero, and a
    bandwidth that small makes the Gaussian weights underflow at every grid node not
    sitting on an observation, so the surface masks out entirely.

    The 75th percentile is robust to duplicates while still measuring the spacing between
    distinct structures, and the floor keeps it non-degenerate when almost every member is
    a duplicate.
    """
    if len(xy) < 2:  # noqa: PLR2004
        return 1.0
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    spacing = float(np.percentile(np.sqrt(d2.min(axis=1)), 75))
    diagonal = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
    bandwidth = max(spacing, 0.05 * diagonal)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0
    return bandwidth


def conformation_landscape(
    structures: list[Complex],
    annotation: CDRAnnotation,
    values: np.ndarray,
    *,
    cdrs: tuple[str, ...] = CDR_NAMES,
    method: str = "pca",
    align_on: str = "framework",
    **kwargs,
) -> tuple[Landscape, Projection, EnsembleCoordinates]:
    """Superpose, project to 2D, and raise `values` over the plane, in one call."""
    ensemble = superpose_cdr_ensemble(
        structures, annotation, cdrs=cdrs, align_on=align_on
    )
    projection = project_2d(ensemble, method=method)
    landscape = smooth_landscape(projection.xy, np.asarray(values, dtype=float), **kwargs)
    return landscape, projection, ensemble


def plot_landscape(
    landscape: Landscape,
    projection: Projection,
    *,
    value_label: str = "confidence",
    title: str | None = None,
    labels: list[str] | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (13, 5.5),
    elev: float = 30,
    azim: float = -60,
):
    """A 3D surface with the observations on it, beside a 2D contour of the same fit."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(
        landscape.grid_x, landscape.grid_y, landscape.grid_z,
        cmap=cmap, alpha=0.75, linewidth=0, antialiased=True, rstride=2, cstride=2,
    )
    ax3d.scatter(
        landscape.xy[:, 0], landscape.xy[:, 1], landscape.z,
        c="black", s=22, depthshade=False, label=f"{len(landscape.z)} structures",
    )
    ax3d.set_xlabel(projection.axis_labels[0], fontsize=8)
    ax3d.set_ylabel(projection.axis_labels[1], fontsize=8)
    ax3d.set_zlabel(value_label, fontsize=8)
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.tick_params(labelsize=7)
    ax3d.legend(fontsize=7, loc="upper left")

    ax2d = fig.add_subplot(1, 2, 2)
    filled = ax2d.contourf(
        landscape.grid_x, landscape.grid_y, landscape.grid_z, levels=18, cmap=cmap
    )
    ax2d.contour(
        landscape.grid_x, landscape.grid_y, landscape.grid_z,
        levels=18, colors="white", linewidths=0.4, alpha=0.5,
    )
    ax2d.scatter(
        landscape.xy[:, 0], landscape.xy[:, 1],
        c=landscape.z, cmap=cmap, edgecolor="black", linewidth=0.6, s=60, zorder=3,
    )
    if labels is not None:
        for (x, y), label in zip(landscape.xy, labels, strict=True):
            ax2d.annotate(str(label), (x, y), fontsize=6, xytext=(3, 3),
                          textcoords="offset points")
    ax2d.set_xlabel(projection.axis_labels[0], fontsize=9)
    ax2d.set_ylabel(projection.axis_labels[1], fontsize=9)
    fig.colorbar(filled, ax=ax2d, label=value_label)
    ax2d.set_title(
        f"kernel bandwidth {landscape.bandwidth:.2f}, "
        f"{landscape.coverage:.0%} of the plane supported",
        fontsize=8,
    )

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig, (ax3d, ax2d)


# ------------------------------------------------------------- structural view


def complex_to_pdb(cx: Complex, *, residue_subset=None) -> str:
    """Serialize a Complex to a PDB string, optionally restricted to antibody residues."""
    import gemmi

    from boltz_cdr.pdb_io import write_complex_cif  # noqa: F401  (kept API-adjacent)

    structure = gemmi.Structure()
    structure.spacegroup_hm = "P 1"
    structure.cell = gemmi.UnitCell(1, 1, 1, 90, 90, 90)
    model = gemmi.Model("1")

    chains = [cx.antibody, cx.antigen] if residue_subset is None else [cx.antibody]
    for chain_i, chain in enumerate(chains):
        gchain = gemmi.Chain(chain.chain_id)
        wanted = (
            range(chain.n_res)
            if residue_subset is None or chain_i != 0
            else sorted(int(i) for i in residue_subset)
        )
        for res_i in wanted:
            residue = gemmi.Residue()
            residue.name = chain.resnames[res_i]
            residue.seqid = gemmi.SeqId(int(chain.resnums[res_i]), " ")
            residue.het_flag = "A"
            for atom_i in chain.residue_atom_indices(res_i):
                atom = gemmi.Atom()
                atom.name = str(chain.atom_names[atom_i])
                atom.element = gemmi.Element(str(chain.atom_elements[atom_i]))
                atom.pos = gemmi.Position(*chain.coords[atom_i])
                atom.occ = 1.0
                atom.b_iso = float(chain.bfactors[atom_i])
                residue.add_atom(atom)
            gchain.add_residue(residue)
        model.add_chain(gchain)
    structure.add_model(model)
    structure.setup_entities()
    return structure.make_pdb_string()


def ensemble_view(
    structures: list[Complex],
    annotation: CDRAnnotation,
    *,
    cdrs: tuple[str, ...] = CDR_NAMES,
    values: np.ndarray | None = None,
    align_on: str = "framework",
    width: int = 900,
    height: int = 600,
    zoom_padding: float = 4.0,
    max_overlay: int | None = None,
):
    """A py3Dmol view of the CDR loops of every member, on one shared framework.

    The framework and antigen are drawn once, from the first structure; each member then
    contributes only its CDR loops. Color runs through a spectrum by member index, or by
    `values` (confidence, DockQ, anything per-structure) when supplied. The camera is
    zoomed on the loops.

    Returns the `py3Dmol.view`, which renders directly in a notebook.
    """
    try:
        import py3Dmol
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        msg = (
            "py3Dmol is required for the structural view. Install it with "
            "`pip install py3Dmol`; the landscape functions do not need it."
        )
        raise ImportError(msg) from exc

    if not structures:
        msg = "need at least one structure"
        raise ValueError(msg)

    ensemble = superpose_cdr_ensemble(structures, annotation, cdrs=cdrs, align_on=align_on)
    loop_residues = ensemble.residue_indices

    view = py3Dmol.view(width=width, height=height)

    # Context: the whole complex, from the reference member, drawn once and muted.
    view.addModel(complex_to_pdb(structures[0]), "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "lightgray", "opacity": 0.55}})
    view.addStyle(
        {"model": 0, "chain": structures[0].antigen.chain_id},
        {"cartoon": {"color": "#8fb8d8", "opacity": 0.75}},
    )

    order = list(range(len(structures)))
    if max_overlay is not None and len(order) > max_overlay:
        order = list(np.linspace(0, len(structures) - 1, max_overlay).astype(int))

    colors = _value_colors(values, order)
    reference_ab = structures[0].antibody
    for slot, index in enumerate(order):
        # Rebuild each member in the reference frame so the overlay is meaningful.
        moved = _in_reference_frame(structures[index], structures[0], annotation, align_on)
        view.addModel(complex_to_pdb(moved, residue_subset=loop_residues), "pdb")
        view.setStyle(
            {"model": slot + 1},
            {"cartoon": {"color": colors[slot]},
             "stick": {"color": colors[slot], "radius": 0.10}},
        )

    resnums = [int(reference_ab.resnums[i]) for i in loop_residues]
    view.zoomTo({"model": 0, "chain": reference_ab.chain_id, "resi": resnums})
    view.zoom(1.0 - zoom_padding / 100.0)
    return view


def _value_colors(values, order) -> list[str]:
    """Hex colors for each overlaid member, by value if given, else by index."""
    from matplotlib import cm
    from matplotlib.colors import Normalize, to_hex

    if values is None:
        cmap = cm.get_cmap("turbo") if hasattr(cm, "get_cmap") else cm.turbo
        return [to_hex(cmap(i / max(len(order) - 1, 1))) for i in range(len(order))]

    values = np.asarray(values, dtype=float)[order]
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return ["#4c72b0"] * len(order)
    norm = Normalize(vmin=finite.min(), vmax=finite.max())
    cmap = cm.viridis
    return [
        to_hex(cmap(norm(v))) if np.isfinite(v) else "#999999" for v in values
    ]


def _in_reference_frame(
    structure: Complex, reference: Complex, annotation: CDRAnnotation, align_on: str
) -> Complex:
    """Copy of `structure` rigid-body transformed onto `reference`'s frame."""
    import copy

    corr = build_correspondence(structure, reference)
    if align_on == "framework":
        anchor = np.setdiff1d(corr.ab_b, annotation.all_indices)
        src, dst = subset_correspondence(corr.ab_b, corr.ab_a, anchor)
        x_ref, x_this = matched_atoms(
            reference.antibody, src, structure.antibody, dst, BACKBONE
        )
    else:
        x_ref, x_this = matched_atoms(
            reference.antigen, corr.ag_b, structure.antigen, corr.ag_a, BACKBONE
        )
    rot, trans = kabsch(x_this, x_ref)

    moved = copy.deepcopy(structure)
    moved.antibody.coords = apply_transform(moved.antibody.coords, rot, trans)
    moved.antigen.coords = apply_transform(moved.antigen.coords, rot, trans)
    return moved
