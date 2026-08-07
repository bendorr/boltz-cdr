"""Ensemble visualization: superposition, projection, and the smoothed landscape.

The properties worth pinning down are invariances and bounds, not pixel output. Framework
superposition must remove rigid-body motion exactly; kernel regression must never report a
value outside the range of the observations; and the surface must be absent wherever no
structure supports it.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from boltz_cdr.cdr import annotate_vhh
from boltz_cdr.visualize import (
    classical_mds,
    complex_to_pdb,
    conformation_landscape,
    pairwise_rmsd,
    project_2d,
    smooth_landscape,
    superpose_cdr_ensemble,
)
from tests.conftest import random_rotation


def perturbed_ensemble(native, n=6, scale=0.6, seed=0):
    """`n` copies with CDR-only noise, so loop conformation genuinely differs."""
    annotation = annotate_vhh(native.antibody.seq)
    cdr_atoms = native.antibody.atom_mask_for_residues(annotation.all_indices)
    rng = np.random.default_rng(seed)
    members = []
    for k in range(n):
        member = copy.deepcopy(native)
        member.antibody.coords[cdr_atoms] += rng.normal(
            scale=scale * (k + 1) / n, size=(cdr_atoms.sum(), 3)
        )
        members.append(member)
    return members, annotation


# ------------------------------------------------------------------ superposition

def test_framework_alignment_removes_rigid_motion(native_complex):
    """A rigidly moved copy must produce identical CDR features. The core invariance."""
    annotation = annotate_vhh(native_complex.antibody.seq)
    rng = np.random.default_rng(3)
    rot, trans = random_rotation(rng), np.array([25.0, -8.0, 11.0])

    moved = copy.deepcopy(native_complex)
    moved.antibody.coords = moved.antibody.coords @ rot.T + trans
    moved.antigen.coords = moved.antigen.coords @ rot.T + trans

    ensemble = superpose_cdr_ensemble([native_complex, moved], annotation)
    assert np.allclose(ensemble.features[0], ensemble.features[1], atol=1e-6)
    assert pairwise_rmsd(ensemble)[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_antigen_alignment_retains_binder_placement(native_complex):
    """Superposing on the antigen must keep rigid-body displacement of the binder visible."""
    annotation = annotate_vhh(native_complex.antibody.seq)
    moved = copy.deepcopy(native_complex)
    moved.antibody.coords = moved.antibody.coords + np.array([5.0, 0.0, 0.0])

    on_framework = superpose_cdr_ensemble([native_complex, moved], annotation)
    on_antigen = superpose_cdr_ensemble(
        [native_complex, moved], annotation, align_on="antigen"
    )
    assert pairwise_rmsd(on_framework)[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert pairwise_rmsd(on_antigen)[0, 1] == pytest.approx(5.0, abs=1e-3)


def test_feature_matrix_shape_and_content(native_complex):
    members, annotation = perturbed_ensemble(native_complex, n=5)
    ensemble = superpose_cdr_ensemble(members, annotation)

    n_res = len(ensemble.residue_indices)
    assert ensemble.features.shape == (5, n_res * len(ensemble.atom_names) * 3)
    assert ensemble.cdr_coords.shape == (5, n_res * len(ensemble.atom_names), 3)
    assert ensemble.n_dropped == 0
    assert np.isfinite(ensemble.features).all()


def test_cdr_subset_selects_fewer_residues(native_complex):
    members, annotation = perturbed_ensemble(native_complex, n=3)
    everything = superpose_cdr_ensemble(members, annotation)
    only_cdr3 = superpose_cdr_ensemble(members, annotation, cdrs=("cdr3",))
    assert len(only_cdr3.residue_indices) == len(annotation.cdr3)
    assert len(only_cdr3.residue_indices) < len(everything.residue_indices)


def test_bad_alignment_target_rejected(native_complex):
    _, annotation = perturbed_ensemble(native_complex, n=2)
    with pytest.raises(ValueError, match="align_on must be"):
        superpose_cdr_ensemble([native_complex], annotation, align_on="elbow")


# --------------------------------------------------------------------- projection

def test_pca_projection(native_complex):
    members, annotation = perturbed_ensemble(native_complex, n=8)
    projection = project_2d(superpose_cdr_ensemble(members, annotation), method="pca")

    assert projection.xy.shape == (8, 2)
    assert projection.explained_variance.shape == (2,)
    assert 0 <= projection.explained_variance.sum() <= 1 + 1e-9
    assert "PC1" in projection.axis_labels[0]
    assert np.allclose(projection.xy.mean(axis=0), 0, atol=1e-6), "PCA output must be centered"


def test_pca_is_deterministic(native_complex):
    members, annotation = perturbed_ensemble(native_complex, n=6)
    ensemble = superpose_cdr_ensemble(members, annotation)
    a = project_2d(ensemble, method="pca").xy
    b = project_2d(ensemble, method="pca").xy
    assert np.array_equal(a, b)


def test_mds_projection_approximates_structural_distance(native_complex):
    members, annotation = perturbed_ensemble(native_complex, n=7)
    ensemble = superpose_cdr_ensemble(members, annotation)
    projection = project_2d(ensemble, method="mds")
    assert projection.xy.shape == (7, 2)

    true_d = pairwise_rmsd(ensemble)
    embedded = np.linalg.norm(
        projection.xy[:, None, :] - projection.xy[None, :, :], axis=-1
    )
    upper = np.triu_indices(len(members), k=1)
    assert np.corrcoef(true_d[upper], embedded[upper])[0, 1] > 0.7


def test_unknown_projection_method_rejected(native_complex):
    members, annotation = perturbed_ensemble(native_complex, n=3)
    with pytest.raises(ValueError, match="unknown projection method"):
        project_2d(superpose_cdr_ensemble(members, annotation), method="umap")


def test_classical_mds_recovers_a_known_configuration():
    """A square's pairwise distances must embed back to a square, up to rigid motion."""
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    distance = np.linalg.norm(square[:, None, :] - square[None, :, :], axis=-1)
    embedded = classical_mds(distance)
    recovered = np.linalg.norm(embedded[:, None, :] - embedded[None, :, :], axis=-1)
    assert np.allclose(recovered, distance, atol=1e-8)


def test_pairwise_rmsd_is_a_metric_matrix(native_complex):
    members, annotation = perturbed_ensemble(native_complex, n=5)
    d = pairwise_rmsd(superpose_cdr_ensemble(members, annotation))
    assert np.allclose(np.diag(d), 0.0, atol=1e-9)
    assert np.allclose(d, d.T)
    assert (d >= 0).all()


# --------------------------------------------------------------------- landscape

def test_kernel_average_never_leaves_the_data_range():
    """The defining property of kernel regression, and why it is used instead of an
    interpolating spline: the surface cannot invent a peak higher than any observation."""
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(30, 2)) * 5
    z = rng.uniform(0.2, 0.9, size=30)
    grid = smooth_landscape(xy, z).grid_z
    finite = grid[np.isfinite(grid)]
    assert finite.min() >= z.min() - 1e-9
    assert finite.max() <= z.max() + 1e-9


def test_unsupported_regions_are_masked_not_extrapolated():
    """Two distant clusters must leave the gap between them empty."""
    xy = np.vstack([np.zeros((8, 2)), np.full((8, 2), 100.0)])
    xy = xy + np.random.default_rng(1).normal(scale=0.4, size=xy.shape)
    z = np.r_[np.full(8, 0.2), np.full(8, 0.9)]
    landscape = smooth_landscape(xy, z, resolution=60)

    assert np.isnan(landscape.grid_z).any(), "the empty gap must not be filled"
    assert landscape.coverage < 0.9
    nodes = np.column_stack([landscape.grid_x.ravel(), landscape.grid_y.ravel()])
    nearest = int(np.argmin(np.linalg.norm(nodes - np.array([50.0, 50.0]), axis=1)))
    assert not np.isfinite(landscape.grid_z.ravel()[nearest]), (
        "the midpoint between two clusters 141 units apart must not be painted"
    )


def test_landscape_handles_a_tightly_clustered_ensemble():
    """Near-duplicate structures must not collapse the bandwidth and blank the surface.

    A nearest-neighbor bandwidth rule fails exactly here: duplicated predictions drive the
    median nearest-neighbor distance toward zero, the Gaussian weights underflow at every
    grid node, and the whole surface masks out.
    """
    rng = np.random.default_rng(2)
    xy = rng.normal(scale=0.001, size=(20, 2))
    xy[0] += [3.0, 3.0]
    z = rng.uniform(0.3, 0.8, size=20)
    landscape = smooth_landscape(xy, z)
    assert landscape.bandwidth > 0
    assert landscape.coverage > 0.01, "surface masked out entirely"


@pytest.mark.parametrize("n", [3, 4, 9, 25])
def test_landscape_works_for_arbitrary_ensemble_sizes(native_complex, n):
    members, annotation = perturbed_ensemble(native_complex, n=n)
    values = np.linspace(0.3, 0.95, n)
    landscape, projection, ensemble = conformation_landscape(members, annotation, values)
    assert landscape.grid_z.shape == landscape.grid_x.shape
    assert ensemble.n_structures == n
    assert projection.xy.shape == (n, 2)
    assert np.isfinite(landscape.grid_z).any()


def test_landscape_needs_at_least_three_points():
    with pytest.raises(ValueError, match="at least 3 structures"):
        smooth_landscape(np.zeros((2, 2)), np.array([0.1, 0.2]))


def test_landscape_ignores_non_finite_values():
    rng = np.random.default_rng(4)
    xy = rng.normal(size=(10, 2))
    z = rng.uniform(size=10)
    z[3] = np.nan
    landscape = smooth_landscape(xy, z)
    assert len(landscape.z) == 9
    assert np.isfinite(landscape.z).all()


def test_hull_clipping_is_stricter(native_complex):
    rng = np.random.default_rng(5)
    xy = rng.normal(size=(20, 2)) * 4
    z = rng.uniform(0.2, 0.9, size=20)
    assert (
        smooth_landscape(xy, z, clip_to_hull=True).coverage
        <= smooth_landscape(xy, z, clip_to_hull=False).coverage
    )


# ------------------------------------------------------------------ pdb + 3d view

def test_complex_to_pdb_roundtrips(native_complex, tmp_path):
    from boltz_cdr.pdb_io import load_chains

    text = complex_to_pdb(native_complex)
    path = tmp_path / "out.pdb"
    path.write_text(text)
    chains = load_chains(path)

    assert set(chains) == {native_complex.antibody.chain_id, native_complex.antigen.chain_id}
    reloaded = chains[native_complex.antibody.chain_id]
    assert reloaded.n_res == native_complex.antibody.n_res
    assert np.allclose(reloaded.coords, native_complex.antibody.coords, atol=1e-2)


def test_complex_to_pdb_residue_subset(native_complex):
    annotation = annotate_vhh(native_complex.antibody.seq)
    full = complex_to_pdb(native_complex)
    subset = complex_to_pdb(native_complex, residue_subset=annotation.cdr3)

    n_full = sum(1 for line in full.splitlines() if line.startswith("ATOM"))
    n_subset = sum(1 for line in subset.splitlines() if line.startswith("ATOM"))
    expected = int(native_complex.antibody.atom_mask_for_residues(annotation.cdr3).sum())
    assert n_subset == expected
    assert 0 < n_subset < n_full


def test_ensemble_view_builds_one_model_per_member(native_complex):
    py3dmol = pytest.importorskip("py3Dmol")  # noqa: F841
    from boltz_cdr.visualize import ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=4)
    html = ensemble_view(members, annotation).write_html()
    assert html.count("addModel") == len(members) + 1, "one context model plus each member"
    assert "zoomTo" in html


def test_ensemble_view_thins_a_large_ensemble(native_complex):
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=10)
    html = ensemble_view(members, annotation, max_overlay=3).write_html()
    assert html.count("addModel") == 4


def test_ensemble_view_colors_by_value(native_complex):
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=5)
    values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    html = ensemble_view(members, annotation, values=values).write_html()
    import re

    colors = set(re.findall(r'"color":\s*"(#[0-9a-fA-F]{6})"', html))
    assert len(colors) >= 4, "distinct values should map to distinct colors"


# ------------------------------------------------------------------ view controls

def test_view_returns_controls_by_default(native_complex):
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import EnsembleView, ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=4)
    view = ensemble_view(members, annotation)
    assert isinstance(view, EnsembleView)
    assert view.n_members == 4
    assert "<script>" in view.write_html()


def test_controls_false_returns_bare_py3dmol_view(native_complex):
    py3dmol = pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=3)
    view = ensemble_view(members, annotation, controls=False)
    assert isinstance(view, py3dmol.view)


def test_control_javascript_is_syntactically_valid(native_complex):
    """The toggles are hand-written JavaScript, so parse it rather than trust it."""
    pytest.importorskip("py3Dmol")
    esprima = pytest.importorskip("esprima")
    from boltz_cdr.visualize import ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=3)
    html = ensemble_view(members, annotation).write_html()
    script = html.rsplit("<script>", 1)[1].rsplit("</script>", 1)[0]
    esprima.parseScript(script)


def test_legend_has_one_toggle_per_member(native_complex):
    pytest.importorskip("py3Dmol")
    import re

    from boltz_cdr.visualize import ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=5)
    html = ensemble_view(members, annotation).write_html()
    uid = re.search(r"viewer_(\d+)", html).group(1)

    assert html.count(f"BC{uid}.member(") == len(members)
    swatches = re.findall(r'bc-swatch" style="background:(#[0-9a-f]{6})', html)
    assert len(swatches) == len(members)
    assert len(set(swatches)) == len(members), "each member needs a distinct swatch"


def test_legend_labels_and_values(native_complex):
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=3)
    html = ensemble_view(
        members, annotation, labels=["low", "mid", "high"], values=np.array([0.11, 0.55, 0.99])
    ).write_html()
    for label in ("low", "mid", "high"):
        assert f">{label}</span>" in html
    for value in ("0.11", "0.55", "0.99"):
        assert value in html


def test_side_chain_and_element_toggles_reach_the_markup(native_complex):
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ELEMENT_COLORS, ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=3)
    html = ensemble_view(members, annotation).write_html()

    # Side chains are everything that is not backbone.
    assert "invert: true" in html
    # Heteroatoms get element colors; carbon takes the member's color.
    for color in ELEMENT_COLORS.values():
        assert color in html
    assert '"C": colors[i]' in html or 'map["C"] = colors[i]' in html


def test_initial_toggle_state_follows_the_arguments(native_complex):
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=3)
    on = ensemble_view(members, annotation, side_chains=True, show_context=True).write_html()
    off = ensemble_view(
        members, annotation, side_chains=False, color_side_chains_by_element=False,
        show_context=False,
    ).write_html()

    assert "side: true" in on and "context: true" in on
    assert "side: false" in off and "elem: false" in off and "context: false" in off


def test_side_chains_off_omits_the_stick_style(native_complex):
    """The static styles, not just the JS state, must honor the argument."""
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=2)
    with_sc = ensemble_view(members, annotation, side_chains=True, controls=False)
    without = ensemble_view(members, annotation, side_chains=False, controls=False)
    assert with_sc.write_html().count("invert") > without.write_html().count("invert")


# ------------------------------------------- model provenance and shared coloring

def test_multimodel_load_records_pdb_model_numbers(tmp_path):
    """Legend labels come from the file's own model numbering, not list position."""
    from boltz_cdr.pdb_io import fetch_cif, load_models

    try:
        path = fetch_cif("9KFW", tmp_path)
    except Exception as exc:
        pytest.skip(f"could not fetch 9KFW: {exc}")

    models = load_models(path)
    assert len(models) > 1
    assert [m.model_id for m in models[:3]] == ["model 1", "model 2", "model 3"]


def test_model_id_survives_residue_subsetting(native_complex):
    assert native_complex.antibody.model_id == "8QF4"
    subset = native_complex.antibody.subset_residues(range(5))
    assert subset.model_id == "8QF4"


def test_default_labels_use_provenance(native_complex):
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=3)
    view = ensemble_view(members, annotation)
    assert view.labels == ["8QF4"] * 3
    assert ">8QF4</span>" in view.write_html()


def test_view_colors_are_exposed_for_keying_other_plots(native_complex):
    """The landscape can only be keyed to the overlay if the colors are retrievable."""
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ensemble_view, member_colors

    members, annotation = perturbed_ensemble(native_complex, n=6)
    view = ensemble_view(members, annotation)
    assert len(view.colors) == len(members)
    assert len(set(view.colors)) == len(members)
    # Index coloring remains available and is what `member_colors` reproduces.
    by_index = ensemble_view(members, annotation, color_by="index")
    assert by_index.colors == member_colors(len(members))


def test_member_colors_follows_values(native_complex):
    from boltz_cdr.visualize import member_colors

    by_index = member_colors(4)
    by_value = member_colors(4, np.array([0.1, 0.2, 0.3, 0.9]))
    assert by_index != by_value
    assert all(c.startswith("#") for c in by_value)


def test_landscape_accepts_matching_point_colors(native_complex):
    import matplotlib
    matplotlib.use("Agg")
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import conformation_landscape, ensemble_view, plot_landscape

    members, annotation = perturbed_ensemble(native_complex, n=6)
    view = ensemble_view(members, annotation)
    values = np.linspace(0.3, 0.9, 6)
    landscape, projection, _ = conformation_landscape(members, annotation, values)

    fig, (_ax3d, ax2d) = plot_landscape(landscape, projection, point_colors=view.colors)
    fig.canvas.draw()
    # Identity is carried by the point outlines; the fill carries the value.
    import matplotlib.colors as mcolors
    rings = set()
    for collection in ax2d.collections:
        if hasattr(collection, "get_edgecolor"):
            rings |= {mcolors.to_hex(c) for c in collection.get_edgecolor()}
    assert set(view.colors) <= rings
    matplotlib.pyplot.close(fig)


def test_surface_keeps_its_value_colormap(native_complex):
    """The surface encodes the value in both modes; identity lives only on the points."""
    import matplotlib
    matplotlib.use("Agg")

    from boltz_cdr.visualize import conformation_landscape, member_colors, plot_landscape

    members, annotation = perturbed_ensemble(native_complex, n=5)
    landscape, projection, _ = conformation_landscape(
        members, annotation, np.linspace(0.2, 0.9, 5)
    )
    for colors in (None, member_colors(5)):
        fig, _ = plot_landscape(landscape, projection, point_colors=colors)
        assert fig.axes[1].collections[0].get_cmap().name == "viridis"
        matplotlib.pyplot.close(fig)


def test_axis_labels_present_on_both_panels(native_complex):
    import matplotlib
    matplotlib.use("Agg")

    from boltz_cdr.visualize import conformation_landscape, member_colors, plot_landscape

    members, annotation = perturbed_ensemble(native_complex, n=4)
    landscape, projection, _ = conformation_landscape(
        members, annotation, np.linspace(0.3, 0.7, 4)
    )
    fig, (ax3d, ax2d) = plot_landscape(
        landscape, projection, value_label="score", point_colors=member_colors(4)
    )
    assert ax3d.get_xlabel() == projection.axis_labels[0]
    assert ax3d.get_ylabel() == projection.axis_labels[1]
    assert ax3d.get_zlabel() == "score"
    assert ax2d.get_xlabel() == projection.axis_labels[0]
    matplotlib.pyplot.close(fig)


# ------------------------------------------------- similarity coloring and dark mode

def test_similarity_order_puts_similar_structures_adjacent(native_complex):
    """The point of seriation: neighbors in the order are neighbors in structure."""
    from boltz_cdr.visualize import similarity_order

    members, annotation = perturbed_ensemble(native_complex, n=12, scale=1.4, seed=7)
    ensemble = superpose_cdr_ensemble(members, annotation)
    distance = pairwise_rmsd(ensemble)
    order = similarity_order(ensemble)

    assert sorted(order.tolist()) == list(range(len(members))), "must be a permutation"
    seriated = np.mean([distance[order[i], order[i + 1]] for i in range(len(order) - 1)])
    overall = distance[~np.eye(len(distance), dtype=bool)].mean()
    assert seriated < overall, "adjacent members should be closer than average"


def test_similarity_order_handles_tiny_ensembles(native_complex):
    from boltz_cdr.visualize import similarity_order

    for n in (1, 2):
        members, annotation = perturbed_ensemble(native_complex, n=n)
        order = similarity_order(superpose_cdr_ensemble(members, annotation))
        assert sorted(order.tolist()) == list(range(n))


def test_similarity_colors_span_red_to_purple(native_complex):
    from boltz_cdr.visualize import similarity_colors, similarity_order

    members, annotation = perturbed_ensemble(native_complex, n=8, scale=1.2)
    ensemble = superpose_cdr_ensemble(members, annotation)
    colors = similarity_colors(ensemble)
    order = similarity_order(ensemble)

    assert len(colors) == len(members)
    assert len(set(colors)) == len(members)
    # First and last in the *similarity* order are the spectrum endpoints.
    assert colors[order[0]] == "#ff0000"
    assert colors[order[-1]] == "#8000ff"


def test_color_by_modes(native_complex):
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=5)
    values = np.linspace(0.2, 0.9, 5)

    modes = {
        mode: ensemble_view(members, annotation, color_by=mode, values=values).colors
        for mode in ("similarity", "value", "index")
    }
    assert modes["similarity"] != modes["index"], "similarity must not be list order"
    assert modes["value"] != modes["index"]
    for colors in modes.values():
        assert len(set(colors)) == len(members)


def test_similarity_is_the_default(native_complex):
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ensemble_view, similarity_colors

    members, annotation = perturbed_ensemble(native_complex, n=6)
    default = ensemble_view(members, annotation).colors
    explicit = similarity_colors(superpose_cdr_ensemble(members, annotation))
    assert default == explicit


def test_unknown_color_by_rejected(native_complex):
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ensemble_view

    members, annotation = perturbed_ensemble(native_complex, n=3)
    with pytest.raises(ValueError, match="color_by must be"):
        ensemble_view(members, annotation, color_by="rainbow")


def test_legend_text_is_readable_on_a_dark_background(native_complex):
    """Notebook output is rendered against the host's theme; VSCode's is often dark."""
    pytest.importorskip("py3Dmol")
    from boltz_cdr.visualize import ensemble_view

    html = ensemble_view(*perturbed_ensemble(native_complex, n=3)).write_html()
    assert "prefers-color-scheme: dark" in html
    dark_block = html.split("prefers-color-scheme: dark")[1].split("}}")[0]
    assert "#f0f0f0" in dark_block or "#fff" in dark_block.lower()


def test_landscape_points_are_filled_by_value_and_ringed_by_identity(native_complex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors

    from boltz_cdr.visualize import conformation_landscape, member_colors, plot_landscape

    members, annotation = perturbed_ensemble(native_complex, n=6)
    landscape, projection, _ = conformation_landscape(
        members, annotation, np.linspace(0.2, 0.9, 6)
    )
    colors = member_colors(6)
    fig, (ax3d, ax2d) = plot_landscape(landscape, projection, point_colors=colors)
    fig.canvas.draw()

    rings = set()
    for collection in ax2d.collections:
        if not hasattr(collection, "get_edgecolor"):
            continue
        rings |= {mcolors.to_hex(c) for c in collection.get_edgecolor()}
    assert set(colors) <= rings, "identity colors must appear as point outlines"

    assert not ax3d.lines, "the connecting stems should be gone"
    matplotlib.pyplot.close(fig)
