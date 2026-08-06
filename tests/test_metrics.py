"""Metric tests built on analytically-known geometry, plus invariance checks.

Deliberately avoids "run it and assert the number it printed". Each test either compares
against a value derivable by hand, or asserts an invariance a correct implementation must
have (rigid-motion invariance, self-comparison identity, monotonicity under degradation).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from boltz_cdr.cdr import annotate_vhh
from boltz_cdr.metrics import (
    build_correspondence,
    compute_dockq,
    contact_report,
    evaluate,
    interface_report,
    residue_contact_map,
    sasa,
    shape_complementarity,
)
from boltz_cdr.metrics.dockq import capri_class, dockq_score
from boltz_cdr.metrics.rmsd import kabsch, ligand_rmsd, rmsd, superposed_rmsd
from tests.conftest import random_rotation

# ----------------------------------------------------------------- Kabsch / RMSD

def test_kabsch_recovers_a_known_transform():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    rot_true = random_rotation(rng)
    trans_true = np.array([3.0, -1.0, 7.0])
    y = x @ rot_true.T + trans_true

    rot, trans = kabsch(x, y)
    assert np.allclose(rot, rot_true, atol=1e-8)
    assert np.allclose(trans, trans_true, atol=1e-8)
    assert superposed_rmsd(x, y) < 1e-9


def test_kabsch_never_reflects():
    """A reflected point set must not be superposable to RMSD 0."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 3))
    mirrored = x * np.array([1.0, 1.0, -1.0])
    assert superposed_rmsd(x, mirrored) > 0.1
    rot, _ = kabsch(x, mirrored)
    assert np.linalg.det(rot) == pytest.approx(1.0)


def test_rmsd_of_known_displacement():
    x = np.zeros((10, 3))
    y = np.tile(np.array([3.0, 4.0, 0.0]), (10, 1))
    assert rmsd(x, y) == pytest.approx(5.0)


# ----------------------------------------------------------------- identity checks

def test_self_evaluation_is_perfect(native_complex):
    annotation = annotate_vhh(native_complex.antibody.seq)
    corr = build_correspondence(native_complex, native_complex)
    report = evaluate(native_complex, native_complex, corr, annotation)

    assert report.dockq.dockq == pytest.approx(1.0)
    assert report.dockq.capri_class == "high"
    assert report.contacts.fnat == pytest.approx(1.0)
    assert report.contacts.epitope_recall == pytest.approx(1.0)
    for value in (
        report.rmsd.complex_rmsd, report.rmsd.ligand_rmsd,
        report.rmsd.interface_rmsd, report.rmsd.cdr3_rmsd,
    ):
        assert value == pytest.approx(0.0, abs=1e-6)


def test_metrics_are_invariant_to_rigid_motion(native_complex):
    """Moving the whole complex must not change any metric."""
    rng = np.random.default_rng(7)
    rot, trans = random_rotation(rng), np.array([12.0, -5.0, 3.0])

    moved = copy.deepcopy(native_complex)
    moved.antibody.coords = moved.antibody.coords @ rot.T + trans
    moved.antigen.coords = moved.antigen.coords @ rot.T + trans

    corr = build_correspondence(moved, native_complex)
    report = compute_dockq(moved, native_complex, corr)
    assert report.dockq == pytest.approx(1.0, abs=1e-6)

    before = interface_report(native_complex)
    after = interface_report(moved)
    assert after.shape_complementarity == pytest.approx(before.shape_complementarity, abs=0.02)
    assert after.n_hbonds == before.n_hbonds
    assert after.n_salt_bridges == before.n_salt_bridges


def test_dockq_degrades_monotonically_with_displacement(native_complex):
    """Pulling the binder off its epitope must monotonically reduce DockQ."""
    direction = np.array([1.0, 0.0, 0.0])
    scores = []
    for distance in (0.0, 1.0, 3.0, 6.0, 12.0):
        moved = copy.deepcopy(native_complex)
        moved.antibody.coords = moved.antibody.coords + distance * direction
        corr = build_correspondence(moved, native_complex)
        scores.append(compute_dockq(moved, native_complex, corr).dockq)
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0.99
    assert scores[-1] < 0.2


def test_ligand_rmsd_equals_displacement(native_complex):
    """With the antigen fixed, L-RMSD of a pure translation is the translation length."""
    moved = copy.deepcopy(native_complex)
    moved.antibody.coords = moved.antibody.coords + np.array([0.0, 5.0, 0.0])
    corr = build_correspondence(moved, native_complex)
    assert ligand_rmsd(moved, native_complex, corr) == pytest.approx(5.0, abs=1e-6)


# ----------------------------------------------------------------- DockQ formula

def test_dockq_formula_matches_published_definition():
    assert dockq_score(1.0, 0.0, 0.0) == pytest.approx(1.0)
    assert dockq_score(0.0, 1e9, 1e9) == pytest.approx(0.0, abs=1e-9)
    # By construction each term contributes 1/3; at the scaling constants each is 1/2.
    assert dockq_score(0.5, 1.5, 8.5) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.95, "high"), (0.80, "high"), (0.60, "medium"), (0.49, "medium"),
     (0.30, "acceptable"), (0.23, "acceptable"), (0.10, "incorrect")],
)
def test_capri_thresholds(score, expected):
    assert capri_class(score) == expected


# ----------------------------------------------------------------- contacts

def test_contact_map_matches_brute_force(native_complex):
    ab = native_complex.antibody.subset_residues(np.arange(12))
    ag = native_complex.antigen.subset_residues(np.arange(12))
    cmap = residue_contact_map(ab, ag, cutoff=6.0)

    expected = np.zeros_like(cmap)
    for i in range(ab.n_res):
        for j in range(ag.n_res):
            d = np.linalg.norm(
                ab.coords[ab.residue_atom_indices(i)][:, None, :]
                - ag.coords[ag.residue_atom_indices(j)][None, :, :],
                axis=-1,
            )
            expected[i, j] = bool((d < 6.0).any())
    assert np.array_equal(cmap, expected)


def test_no_contacts_when_chains_are_separated(native_complex):
    moved = copy.deepcopy(native_complex)
    moved.antibody.coords = moved.antibody.coords + np.array([500.0, 0.0, 0.0])
    corr = build_correspondence(moved, native_complex)
    report = contact_report(moved, native_complex, corr)
    assert report.n_predicted_contacts == 0
    assert report.fnat == pytest.approx(0.0)
    assert report.epitope_recall == pytest.approx(0.0)


# ----------------------------------------------------------------- SASA / Sc

def test_sasa_of_an_isolated_atom_is_the_sphere_area():
    """One carbon alone: SASA = 4*pi*(1.70 + 1.40)^2."""
    area = sasa(np.zeros((1, 3)), np.array(["C"]), n_points=2000)[0]
    assert area == pytest.approx(4 * np.pi * (1.70 + 1.40) ** 2, rel=1e-3)


def test_sasa_decreases_when_atoms_are_brought_together():
    coords_far = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    coords_near = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = np.array(["C", "C"])
    assert sasa(coords_near, elements).sum() < sasa(coords_far, elements).sum()


def test_shape_complementarity_in_published_range(native_complex):
    """Real antibody-antigen interfaces sit at Sc ~ 0.64-0.75 (Lawrence & Colman 1993).

    This is the calibration test for the metric: it caught two wrong surface definitions
    during development, one giving ~0.08 and one giving ~0.47.
    """
    value = shape_complementarity(native_complex)
    assert 0.60 < value < 0.80, f"Sc = {value:.3f} is outside the physical range"


def test_shape_complementarity_drops_for_a_bad_interface(native_complex):
    separated = copy.deepcopy(native_complex)
    separated.antibody.coords = separated.antibody.coords + np.array([8.0, 0.0, 0.0])
    good = shape_complementarity(native_complex)
    bad = shape_complementarity(separated)
    assert np.isnan(bad) or bad < good


def test_crystal_structures_have_no_clashes(native_complex):
    """A refined crystal structure must not register steric overlap."""
    assert interface_report(native_complex).n_clashes == 0
