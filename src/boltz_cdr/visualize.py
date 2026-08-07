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
    chain_correspondence,
    matched_atoms,
    subset_correspondence,
)
from boltz_cdr.metrics.rmsd import apply_transform, kabsch
from boltz_cdr.pdb_io import Chain, Complex

# An ensemble member is either a full complex or a bare antibody chain. Apo ensembles are
# a legitimate and common case — NMR depositions of unbound nanobodies, or a set of
# predictions made before any docking — and nothing in the loop analysis requires an
# antigen, so both are accepted throughout.
EnsembleMember = Complex | Chain


def _antibody(member: EnsembleMember) -> Chain:
    return member.antibody if isinstance(member, Complex) else member


def _antigen(member: EnsembleMember) -> Chain | None:
    return member.antigen if isinstance(member, Complex) else None


@dataclass
class EnsembleCoordinates:
    """CDR coordinates of every ensemble member, on a common frame."""

    features: np.ndarray  # (n_structures, n_atoms * 3) flattened, superposed
    cdr_coords: np.ndarray  # (n_structures, n_atoms, 3)
    reference: EnsembleMember  # the structure everything was superposed onto
    residue_indices: np.ndarray  # CDR residue indices in the reference
    atom_names: tuple[str, ...]
    n_dropped: int  # residues excluded for not being present in every member

    @property
    def n_structures(self) -> int:
        return len(self.features)


def superpose_cdr_ensemble(
    structures: list[EnsembleMember],
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
    if align_on == "antigen" and any(_antigen(m) is None for m in structures):
        msg = "align_on='antigen' needs every member to carry an antigen chain"
        raise ValueError(msg)

    reference = structures[0]
    reference_ab = _antibody(reference)
    loop_residues = (
        np.concatenate([annotation[c] for c in cdrs]) if cdrs else annotation.all_indices
    )
    loop_residues = np.unique(loop_residues)

    # Residues present, with all requested atoms, in every member.
    usable = set(loop_residues.tolist())
    correspondences = []
    for structure in structures:
        this_ab, ref_ab = _antibody(structure), reference_ab
        ab_this, ab_ref = chain_correspondence(this_ab, ref_ab)
        correspondences.append((ab_this, ab_ref))
        usable &= set(ab_ref.tolist())
    common = np.array(sorted(usable), dtype=int)
    n_dropped = len(loop_residues) - len(common)
    if len(common) == 0:
        msg = "no CDR residues are present in every ensemble member"
        raise ValueError(msg)

    coords = []
    for structure, (ab_this, ab_ref) in zip(structures, correspondences, strict=True):
        this_ab = _antibody(structure)
        if align_on == "framework":
            anchor = np.setdiff1d(ab_ref, loop_residues)
            src, dst = subset_correspondence(ab_ref, ab_this, anchor)
            x_src, x_dst = matched_atoms(reference_ab, src, this_ab, dst, atom_names)
        else:
            ag_this, ag_ref = chain_correspondence(
                _antigen(structure), _antigen(reference)
            )
            x_src, x_dst = matched_atoms(
                _antigen(reference), ag_ref, _antigen(structure), ag_this, atom_names
            )
        if len(x_src) < 3:  # noqa: PLR2004
            msg = f"too few {align_on} atoms to superpose ({len(x_src)})"
            raise ValueError(msg)

        # Transform taking this structure onto the reference frame.
        rot, trans = kabsch(x_dst, x_src)

        loop_ref, loop_this = subset_correspondence(ab_ref, ab_this, common)
        _, x_loop = matched_atoms(reference_ab, loop_ref, this_ab, loop_this, atom_names)
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
    structures: list[EnsembleMember],
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
    cmap: str | None = None,
    figsize: tuple[float, float] = (13, 5.5),
    elev: float = 30,
    azim: float = -60,
    point_colors: list[str] | None = None,
):
    """A 3D surface with the observations on it, beside a 2D contour of the same fit.

    `point_colors` overrides the coloring of the plotted structures — pass
    `EnsembleView.colors` (or `member_colors(n)`) to fill each point with the same color its
    loops have in the 3D overlay, so a feature of the landscape can be traced back to a
    specific conformation. Its ring then carries the structure's own value on the surface's
    color scale, which keeps the check on the fit: a ring that matches the surface under it
    is a point the smoothing reproduces. Without it the points are filled by value instead.
    """
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = "viridis"

    # Ring colors are normalized over the fitted grid, not over the observations, so a ring
    # and the surface beneath it are the same color when they hold the same value.
    value_edges = None
    if point_colors is not None:
        if len(point_colors) != len(landscape.z):
            msg = (
                f"point_colors has {len(point_colors)} entries for {len(landscape.z)} "
                f"structures. `ensemble_view(max_overlay=N)` thins the overlay and returns "
                f"one color per drawn member, so raise max_overlay (or drop it) to key the "
                f"two figures together."
            )
            raise ValueError(msg)
        norm = plt.Normalize(
            vmin=float(np.nanmin(landscape.grid_z)),
            vmax=float(np.nanmax(landscape.grid_z)),
        )
        value_edges = plt.get_cmap(cmap)(norm(landscape.z))

    fig = plt.figure(figsize=figsize)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(
        landscape.grid_x, landscape.grid_y, landscape.grid_z,
        cmap=cmap, alpha=0.75, linewidth=0, antialiased=True, rstride=2, cstride=2,
    )

    if point_colors is not None:
        ax3d.scatter(
            landscape.xy[:, 0], landscape.xy[:, 1], landscape.z,
            c="none", edgecolor="white", linewidth=3.2, s=74, depthshade=False,
        )
    ax3d.scatter(
        landscape.xy[:, 0], landscape.xy[:, 1], landscape.z,
        c=point_colors if point_colors is not None else "black",
        s=48 if point_colors is not None else 22,
        edgecolor=value_edges if point_colors is not None else "none",
        linewidth=2.2 if point_colors is not None else 0,
        depthshade=False, label=f"{len(landscape.z)} structures",
    )

    ax3d.set_xlabel(projection.axis_labels[0], fontsize=8)
    ax3d.set_ylabel(projection.axis_labels[1], fontsize=8)
    ax3d.set_zlabel(value_label, fontsize=8)
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.tick_params(labelsize=7)
    if point_colors is None:
        ax3d.legend(fontsize=7, loc="upper left")

    ax2d = fig.add_subplot(1, 2, 2)
    filled = ax2d.contourf(
        landscape.grid_x, landscape.grid_y, landscape.grid_z, levels=18, cmap=cmap
    )
    ax2d.contour(
        landscape.grid_x, landscape.grid_y, landscape.grid_z,
        levels=18, colors="white", linewidths=0.4, alpha=0.5,
    )
    if point_colors is not None:
        # A white halo outside the value ring. Without it a ring whose value matches the
        # surface beneath it — which is the common case when the fit is good — has no
        # visible outer boundary and the marker reads as a bare fill.
        ax2d.scatter(
            landscape.xy[:, 0], landscape.xy[:, 1],
            c="none", edgecolor="white", linewidth=3.8, s=138, zorder=2,
        )
    ax2d.scatter(
        landscape.xy[:, 0], landscape.xy[:, 1],
        c=point_colors if point_colors is not None else landscape.z,
        cmap=None if point_colors is not None else cmap,
        edgecolor=value_edges if point_colors is not None else "black",
        linewidth=2.6 if point_colors is not None else 0.6,
        s=106 if point_colors is not None else 60, zorder=3,
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
    if point_colors is None:
        fig.tight_layout()
    else:
        _point_key(fig, point_colors, cmap, value_label)
        fig.tight_layout(rect=(0, 0.065, 1, 1))   # leave the strip the key sits in
    return fig, (ax3d, ax2d)


def _point_key(fig, point_colors: list[str], cmap: str, value_label: str) -> None:
    """A figure-wide key for the two color channels a point carries.

    Which channel means what is not guessable from the marks, and it is the reverse of the
    usual convention: the fill identifies the structure and the ring reports its value.
    Each entry samples the scale it names — three points along the overlay spectrum, three
    rings along the value colormap — so the key looks like the marks it explains.
    """
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    def dot(face, edge, edge_width):
        return Line2D([], [], marker="o", linestyle="none", markersize=7.5,
                      markerfacecolor=face, markeredgecolor=edge,
                      markeredgewidth=edge_width)

    n = len(point_colors)
    spectrum = [point_colors[round(t * (n - 1))] for t in (0, 0.5, 1)] if n >= 3 else point_colors
    fills = tuple(dot(color, "0.45", 0.8) for color in spectrum)
    rings = tuple(dot("0.90", plt.get_cmap(cmap)(t), 2.2) for t in (0.12, 0.5, 0.88))

    fig.legend(
        [fills, rings],
        [f"fill: which model — as in the structural overlay ({n} models)",
         f"ring: {value_label} — on the surface's own scale"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.45)},
        loc="lower center", ncol=2, fontsize=9, handlelength=3.2,
        handletextpad=0.7, columnspacing=2.5, borderaxespad=0.4, frameon=False,
    )


# ------------------------------------------------------------- structural view


def complex_to_pdb(cx: EnsembleMember, *, residue_subset=None) -> str:
    """Serialize a complex or a lone antibody chain to a PDB string.

    `residue_subset` restricts the output to those antibody residues and drops the antigen,
    which is how a single member's loops are added to an overlay.
    """
    import gemmi

    structure = gemmi.Structure()
    structure.spacegroup_hm = "P 1"
    structure.cell = gemmi.UnitCell(1, 1, 1, 90, 90, 90)
    model = gemmi.Model("1")

    antibody, antigen = _antibody(cx), _antigen(cx)
    chains = [antibody]
    if residue_subset is None and antigen is not None:
        chains.append(antigen)
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


# Jmol/CPK element colors for side-chain atoms. Carbon is deliberately absent: it takes
# the per-structure color, so that members stay distinguishable while the heteroatoms that
# actually make interface contacts are identifiable at a glance.
ELEMENT_COLORS = {
    "N": "#3050F8",
    "O": "#FF0D0D",
    "S": "#FFFF30",
    "P": "#FF8000",
    "SE": "#FFA100",
    "F": "#90E050",
    "CL": "#1FF01F",
}

BACKBONE_ATOMS = ["N", "CA", "C", "O"]


class EnsembleView:
    """A py3Dmol viewer with toggles, rendered as self-contained HTML.

    The controls are plain JavaScript rather than ipywidgets. A notebook committed with
    its outputs has no live kernel behind it, so widget callbacks would be dead on arrival
    for anyone reading the saved file; JavaScript keeps the toggles working in Colab,
    nbviewer, and any exported HTML.
    """

    def __init__(self, html: str, view, n_members: int, colors=None, labels=None):
        self.html = html
        self.view = view
        self.n_members = n_members
        # The color and label given to each member, in the order they were drawn. Pass
        # `colors` to `plot_landscape(point_colors=...)` to key the two figures together.
        self.colors = list(colors or [])
        self.labels = list(labels or [])

    # py3Dmol publishes its viewer under both `application/3dmoljs_load.v0` and
    # `text/html`, and that is not redundant: VSCode's notebook renderer drives the
    # interactive viewer from the former. Emitting only `text/html` yields a static
    # picture there. Both are therefore published here too, which costs a duplicated
    # payload in the saved notebook and buys interactivity in every host.
    _MIMETYPE = "application/3dmoljs_load.v0"

    def _repr_mimebundle_(self, include=None, exclude=None):
        return {
            self._MIMETYPE: self.html,
            "text/html": self.html,
            "text/plain": f"<EnsembleView: {self.n_members} members>",
        }

    def _repr_html_(self) -> str:
        return self.html

    def write_html(self) -> str:
        return self.html

    def show(self) -> None:
        from IPython.display import publish_display_data

        publish_display_data(self._repr_mimebundle_())


def ensemble_view(
    structures: list[EnsembleMember],
    annotation: CDRAnnotation,
    *,
    cdrs: tuple[str, ...] = CDR_NAMES,
    values: np.ndarray | None = None,
    align_on: str = "framework",
    width: int = 900,
    height: int = 600,
    zoom_padding: float = 4.0,
    max_overlay: int | None = None,
    color_by: str = "similarity",
    side_chains: bool = True,
    color_side_chains_by_element: bool = True,
    show_context: bool = True,
    controls: bool = True,
    labels: list[str] | None = None,
):
    """A py3Dmol view of the CDR loops of every member, on one shared framework.

    The framework — and the antigen, when the members carry one — is drawn once from the
    first structure; each member then contributes only its CDR loops. Color runs through a
    spectrum by member index, or by `values` (confidence, DockQ, anything per-structure)
    when supplied. The camera is zoomed on the loops.

    Parameters
    ----------
    color_by
        ``"similarity"`` (the default) assigns a red-to-purple rainbow along an ordering
        of the members by structural similarity, so that similarly-shaped loops take
        similar colors and the spectrum itself is a coordinate through conformational
        space. ``"value"`` colors by `values` on a sequential scale. ``"index"`` colors
        by position in the list, which carries no structural meaning and is offered only
        for reproducing an existing figure.
    side_chains
        Draw side-chain atoms as sticks in addition to the backbone. Side chains are what
        actually contact the antigen, but they clutter an overlay of many members, so this
        is a toggle rather than a fixed choice.
    color_side_chains_by_element
        Color side-chain carbon with the member's own color and heteroatoms by element
        (N blue, O red, S yellow). This keeps members distinguishable while making the
        polar and charged atoms that form interface contacts identifiable.
    show_context
        Draw the shared framework and antigen behind the loops.
    controls
        Emit HTML controls for the options above plus a per-member legend with
        show/hide checkboxes, and return an `EnsembleView`. With `controls=False` the
        bare `py3Dmol.view` is returned, styled to the options given.
    labels
        Names for the legend, one per member. Defaults to the member index.
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

    # Context: the reference member in full, drawn once and muted.
    view.addModel(complex_to_pdb(structures[0]), "pdb")
    context_style = (
        {"cartoon": {"color": "lightgray", "opacity": 0.55}} if show_context else {}
    )
    view.setStyle({"model": 0}, context_style)
    context_antigen = _antigen(structures[0])
    if context_antigen is not None and show_context:
        view.addStyle(
            {"model": 0, "chain": context_antigen.chain_id},
            {"cartoon": {"color": "#8fb8d8", "opacity": 0.75}},
        )

    order = list(range(len(structures)))
    if max_overlay is not None and len(order) > max_overlay:
        order = list(np.linspace(0, len(structures) - 1, max_overlay).astype(int))

    colors = _resolve_colors(color_by, ensemble, order, values)
    reference_ab = _antibody(structures[0])
    for slot, index in enumerate(order):
        # Rebuild each member in the reference frame so the overlay is meaningful.
        moved = _in_reference_frame(structures[index], structures[0], annotation, align_on)
        view.addModel(complex_to_pdb(moved, residue_subset=loop_residues), "pdb")
        for style in _member_styles(colors[slot], side_chains, color_side_chains_by_element):
            view.addStyle({"model": slot + 1, **style["sel"]}, style["style"])

    resnums = [int(reference_ab.resnums[i]) for i in loop_residues]
    view.zoomTo({"model": 0, "chain": reference_ab.chain_id, "resi": resnums})
    view.zoom(1.0 - zoom_padding / 100.0)

    if not controls:
        return view

    if labels is None:
        # Prefer the identifier the structure came with — a PDB model number, or the file
        # a prediction was read from — over a bare position in the list.
        labels = [
            _antibody(structures[i]).model_id or f"structure {i + 1}" for i in order
        ]
    elif len(labels) == len(structures):
        labels = [labels[i] for i in order]

    legend_values = None
    if values is not None:
        finite = np.asarray(values, dtype=float)[order]
        legend_values = [None if not np.isfinite(v) else float(v) for v in finite]

    html = _view_with_controls(
        view,
        colors=colors,
        labels=[str(label) for label in labels],
        legend_values=legend_values,
        side_chains=side_chains,
        by_element=color_side_chains_by_element,
        show_context=show_context,
        width=width,
    )
    return EnsembleView(html, view, len(order), colors=colors, labels=labels)


def _resolve_colors(color_by: str, ensemble, order, values) -> list[str]:
    """Per-member colors for the requested scheme."""
    if color_by == "similarity":
        everyone = similarity_colors(ensemble)
        return [everyone[i] for i in order]
    if color_by == "value":
        return _value_colors(values, order)
    if color_by == "index":
        return _value_colors(None, order)
    msg = f"color_by must be 'similarity', 'value' or 'index', got {color_by!r}"
    raise ValueError(msg)


def _member_styles(color: str, side_chains: bool, by_element: bool) -> list[dict]:
    """py3Dmol style entries for one overlaid member."""
    styles = [
        {
            "sel": {"atom": BACKBONE_ATOMS},
            "style": {"cartoon": {"color": color},
                      "stick": {"color": color, "radius": 0.10}},
        }
    ]
    if side_chains:
        if by_element:
            scheme = {"prop": "elem", "map": {"C": color, **ELEMENT_COLORS}}
            stick = {"radius": 0.12, "colorscheme": scheme}
        else:
            stick = {"radius": 0.12, "color": color}
        styles.append(
            {"sel": {"atom": BACKBONE_ATOMS, "invert": True}, "style": {"stick": stick}}
        )
    return styles


def _view_with_controls(
    view,
    *,
    colors: list[str],
    labels: list[str],
    legend_values: list[float | None] | None,
    side_chains: bool,
    by_element: bool,
    show_context: bool,
    width: int,
) -> str:
    """Wrap a py3Dmol view in HTML controls driven by plain JavaScript.

    py3Dmol assigns its viewer to a global `viewer_<uid>` inside a `$3Dmolpromise.then`
    callback. Chaining a second callback on the same promise therefore runs after the
    viewer exists, whether or not 3Dmol.js has already been fetched, which is what lets
    the controls attach without ipywidgets or a live kernel.
    """
    import json
    import re

    base = view._make_html()  # py3Dmol's own accessor for its generated HTML
    match = re.search(r"viewer_(\d+)", base)
    if match is None:  # pragma: no cover - py3Dmol has always emitted this
        return base
    uid = match.group(1)

    rows = []
    for i in range(len(labels)):
        suffix = ""
        if legend_values is not None and legend_values[i] is not None:
            suffix = f' <span class="bc-val">{legend_values[i]:.3g}</span>'
        rows.append(
            f'<label class="bc-row"><input type="checkbox" checked '
            f'onchange="BC{uid}.member({i}, this.checked)">'
            f'<span class="bc-swatch" style="background:{colors[i]}"></span>'
            f'<span class="bc-name">{labels[i]}</span>{suffix}</label>'
        )

    def checked(flag: bool) -> str:
        return " checked" if flag else ""

    controls = f"""
<style>
/* Notebook output is rendered against whatever background the host uses, and VSCode's
   dark theme is common. Inheriting the host's text color is unreliable inside the
   output frame, so the panel states its own and flips on the dark-mode media query. */
.bc-wrap{{font:11.5px/1.35 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  max-width:{width}px;margin-top:4px;color:#222}}
@media (prefers-color-scheme: dark) {{
  .bc-wrap{{color:#f0f0f0}}
  .bc-opts{{border-bottom-color:#4a4a4a}}
  .bc-val{{color:#b0b0b0}}
  .bc-btn{{background:#ededed;border-color:#8a8a8a}}   /* stays a light chip: the label is black in both themes */
  .bc-swatch{{border-color:#ffffff59}}
}}
.bc-opts{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:3px 2px;
  border-bottom:1px solid #e3e3e3}}
.bc-opts label{{display:flex;align-items:center;gap:3px;cursor:pointer;white-space:nowrap}}
.bc-opts input{{margin:0}}
.bc-legend{{display:grid;grid-template-columns:repeat(auto-fill,minmax(104px,1fr));
  gap:0 8px;padding:4px 2px;max-height:132px;overflow-y:auto}}
.bc-row{{display:flex;align-items:center;gap:4px;cursor:pointer;line-height:1.5}}
.bc-row input{{margin:0;flex:0 0 auto}}
.bc-swatch{{width:10px;height:10px;border-radius:2px;border:1px solid #00000026;
  flex:0 0 auto}}
.bc-name{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:inherit}}
.bc-val{{color:#888;font-variant-numeric:tabular-nums;margin-left:auto;padding-left:4px}}
.bc-btn{{font:inherit;padding:0 6px;border:1px solid #ccc;border-radius:3px;
  background:#fafafa;color:#000;cursor:pointer;line-height:1.6}}
</style>
<div class="bc-wrap">
  <div class="bc-opts">
    <label><input type="checkbox"{checked(side_chains)}
      onchange="BC{uid}.sideChains(this.checked)"> side chains</label>
    <label><input type="checkbox"{checked(by_element)}
      onchange="BC{uid}.byElement(this.checked)"> color side chains by element</label>
    <label><input type="checkbox"{checked(show_context)}
      onchange="BC{uid}.context(this.checked)"> framework / antigen</label>
    <button class="bc-btn" onclick="BC{uid}.all(true)">show all</button>
    <button class="bc-btn" onclick="BC{uid}.all(false)">hide all</button>
  </div>
  <div class="bc-legend">{"".join(rows)}</div>
</div>
<script>
var BC{uid} = (function() {{
  var colors  = {json.dumps(colors)};
  var visible = colors.map(function() {{ return true; }});
  var opts = {{ side: {json.dumps(side_chains)}, elem: {json.dumps(by_element)},
                context: {json.dumps(show_context)} }};
  var BB = {json.dumps(BACKBONE_ATOMS)};
  var ELEM = {json.dumps(ELEMENT_COLORS)};

  function viewer() {{ return window["viewer_{uid}"]; }}

  function restyle() {{
    var v = viewer();
    if (!v) return;
    v.setStyle({{model: 0}}, opts.context
      ? {{cartoon: {{color: "lightgray", opacity: 0.55}}}} : {{}});
    for (var i = 0; i < colors.length; i++) {{
      var m = i + 1;
      v.setStyle({{model: m}}, {{}});
      if (!visible[i]) continue;
      v.addStyle({{model: m, atom: BB}},
        {{cartoon: {{color: colors[i]}}, stick: {{color: colors[i], radius: 0.10}}}});
      if (opts.side) {{
        var stick;
        if (opts.elem) {{
          var map = JSON.parse(JSON.stringify(ELEM));
          map["C"] = colors[i];
          stick = {{radius: 0.12, colorscheme: {{prop: "elem", map: map}}}};
        }} else {{
          stick = {{radius: 0.12, color: colors[i]}};
        }}
        v.addStyle({{model: m, atom: BB, invert: true}}, {{stick: stick}});
      }}
    }}
    v.render();
  }}

  var api = {{
    member: function(i, on) {{ visible[i] = on; restyle(); }},
    sideChains: function(on) {{ opts.side = on; restyle(); }},
    byElement: function(on) {{ opts.elem = on; restyle(); }},
    context: function(on) {{ opts.context = on; restyle(); }},
    all: function(on) {{
      for (var i = 0; i < visible.length; i++) visible[i] = on;
      var boxes = document.querySelectorAll(".bc-legend input");
      for (var k = 0; k < boxes.length; k++) boxes[k].checked = on;
      restyle();
    }},
    restyle: restyle
  }};
  if (typeof $3Dmolpromise !== "undefined" && $3Dmolpromise) {{
    $3Dmolpromise.then(restyle);
  }}
  return api;
}})();
</script>
"""
    return base + controls


def similarity_order(ensemble: EnsembleCoordinates) -> np.ndarray:
    """Order members so that structurally similar conformations are adjacent.

    Seriation by hierarchical clustering of the pairwise CDR RMSD matrix, with optimal
    leaf ordering. A one-dimensional projection such as PC1 would also give an order, but
    it captures only the dominant mode — typically a third of the variance in a
    prediction ensemble — so two loops that differ mainly along later modes would land
    next to each other. The dendrogram uses the whole distance matrix, and optimal leaf
    ordering then chooses the flip at each node that puts the most similar structures side
    by side.
    """
    n = len(ensemble.features)
    if n < 3:  # noqa: PLR2004
        return np.arange(n)

    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    distance = pairwise_rmsd(ensemble)
    condensed = squareform(np.nan_to_num(distance, nan=0.0), checks=False)
    tree = linkage(condensed, method="average")
    return np.asarray(leaves_list(optimal_leaf_ordering(tree, condensed)))


def similarity_colors(
    ensemble: EnsembleCoordinates, *, cmap: str = "rainbow_r"
) -> list[str]:
    """Rainbow colors assigned along the structural-similarity ordering, red to purple.

    Position in the spectrum then means something: loops of a similar shape are a similar
    color, and a sweep from red to purple is a sweep through conformational space. With
    colors assigned by list position instead, adjacent colors say nothing about the
    structures, which is actively misleading in an overlay of twenty loops.
    """
    import matplotlib.cm as mpl_cm
    from matplotlib.colors import to_hex

    order = similarity_order(ensemble)
    n = len(order)
    palette = mpl_cm.get_cmap(cmap) if hasattr(mpl_cm, "get_cmap") else getattr(mpl_cm, cmap)

    colors = [""] * n
    for rank, member in enumerate(order):
        colors[int(member)] = to_hex(palette(rank / max(n - 1, 1)))
    return colors


def member_colors(n_members: int, values: np.ndarray | None = None) -> list[str]:
    """The colors `ensemble_view` assigns, exposed so other plots can match them.

    Passing the same list to `plot_landscape(point_colors=...)` makes a point on the
    landscape and a loop in the 3D overlay the same color, which is what allows the two
    figures to be read against each other.
    """
    return _value_colors(values, list(range(n_members)))


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
    structure: EnsembleMember,
    reference: EnsembleMember,
    annotation: CDRAnnotation,
    align_on: str,
) -> EnsembleMember:
    """Copy of `structure` rigid-body transformed onto `reference`'s frame."""
    import copy

    this_ab, ref_ab = _antibody(structure), _antibody(reference)
    if align_on == "framework":
        ab_this, ab_ref = chain_correspondence(this_ab, ref_ab)
        anchor = np.setdiff1d(ab_ref, annotation.all_indices)
        src, dst = subset_correspondence(ab_ref, ab_this, anchor)
        x_ref, x_this = matched_atoms(ref_ab, src, this_ab, dst, BACKBONE)
    else:
        ag_this, ag_ref = chain_correspondence(_antigen(structure), _antigen(reference))
        x_ref, x_this = matched_atoms(
            _antigen(reference), ag_ref, _antigen(structure), ag_this, BACKBONE
        )
    rot, trans = kabsch(x_this, x_ref)

    moved = copy.deepcopy(structure)
    _antibody(moved).coords = apply_transform(_antibody(moved).coords, rot, trans)
    if _antigen(moved) is not None:
        _antigen(moved).coords = apply_transform(_antigen(moved).coords, rot, trans)
    return moved
