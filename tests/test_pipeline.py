"""Template construction, YAML generation, and the scorer-comparison machinery."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from boltz_cdr.cdr import annotate_vhh
from boltz_cdr.ensemble import count_clusters, ensemble_report, pairwise_cdr_rmsd
from boltz_cdr.pdb_io import load_chains, write_complex_cif
from boltz_cdr.scoring import add_scores, evaluate_scorers, zscore
from boltz_cdr.templates import build_cdr_masked_template, build_full_template
from boltz_cdr.yaml_io import write_boltz_yaml

# ------------------------------------------------------------------------ templates

def test_cdr_masked_template_removes_exactly_the_cdrs(native_complex, tmp_path):
    annotation = annotate_vhh(native_complex.antibody.seq)
    spec = build_cdr_masked_template(native_complex, annotation, tmp_path / "masked.cif")

    assert spec.n_masked_residues == len(annotation.all_indices)

    chains = load_chains(spec.path)
    templated = chains[spec.antibody_chain_id]
    assert templated.n_res == native_complex.antibody.n_res - len(annotation.all_indices)

    # The retained residues must be exactly the framework, with numbering preserved.
    kept_resnums = set(native_complex.antibody.resnums[spec.kept_antibody_residues].tolist())
    assert set(templated.resnums.tolist()) == kept_resnums
    removed = set(native_complex.antibody.resnums[annotation.all_indices].tolist())
    assert not (set(templated.resnums.tolist()) & removed)


def test_template_preserves_coordinates(native_complex, tmp_path):
    """Masking must not move anything that it keeps."""
    annotation = annotate_vhh(native_complex.antibody.seq)
    spec = build_cdr_masked_template(native_complex, annotation, tmp_path / "masked.cif")

    chains = load_chains(spec.path)
    templated = chains[spec.antibody_chain_id]
    original = native_complex.antibody.subset_residues(spec.kept_antibody_residues)
    assert np.allclose(templated.coords, original.coords, atol=1e-3)

    antigen = chains[spec.antigen_chain_id]
    assert antigen.n_res == native_complex.antigen.n_res
    assert np.allclose(antigen.coords, native_complex.antigen.coords, atol=1e-3)


def test_masking_only_cdr3(native_complex, tmp_path):
    annotation = annotate_vhh(native_complex.antibody.seq)
    spec = build_cdr_masked_template(
        native_complex, annotation, tmp_path / "cdr3.cif", mask_cdrs=("cdr3",)
    )
    assert spec.n_masked_residues == len(annotation.cdr3)


def test_flank_widens_the_mask(native_complex, tmp_path):
    annotation = annotate_vhh(native_complex.antibody.seq)
    narrow = build_cdr_masked_template(
        native_complex, annotation, tmp_path / "n.cif", mask_cdrs=("cdr3",), flank=0
    )
    wide = build_cdr_masked_template(
        native_complex, annotation, tmp_path / "w.cif", mask_cdrs=("cdr3",), flank=2
    )
    assert wide.n_masked_residues == narrow.n_masked_residues + 4


def test_full_template_keeps_everything(native_complex, tmp_path):
    spec = build_full_template(native_complex, tmp_path / "full.cif")
    assert spec.n_masked_residues == 0
    chains = load_chains(spec.path)
    assert chains[spec.antibody_chain_id].n_res == native_complex.antibody.n_res


def test_unknown_cdr_name_rejected(native_complex, tmp_path):
    annotation = annotate_vhh(native_complex.antibody.seq)
    with pytest.raises(ValueError, match="unknown CDR"):
        build_cdr_masked_template(
            native_complex, annotation, tmp_path / "x.cif", mask_cdrs=("cdr4",)
        )


def test_cif_roundtrip_preserves_the_structure(native_complex, tmp_path):
    path = write_complex_cif(native_complex, tmp_path / "rt.cif")
    chains = load_chains(path)
    reloaded = chains[native_complex.antibody.chain_id]
    assert reloaded.n_res == native_complex.antibody.n_res
    assert reloaded.seq == native_complex.antibody.seq
    assert np.allclose(reloaded.coords, native_complex.antibody.coords, atol=1e-3)


# ----------------------------------------------------------------------- yaml input

def test_yaml_has_the_expected_shape(native_complex, tmp_path):
    spec = write_boltz_yaml(
        tmp_path / "in.yaml", native_complex.antibody.seq, native_complex.antigen.seq
    )
    doc = yaml.safe_load(spec.path.read_text())

    assert doc["version"] == 1
    assert len(doc["sequences"]) == 2
    antibody, antigen = doc["sequences"][0]["protein"], doc["sequences"][1]["protein"]
    assert antibody["sequence"] == native_complex.antibody.seq
    assert antibody["msa"] == "empty", "the nanobody should default to single-sequence"
    assert "msa" not in antigen, "the antigen should be left for --use_msa_server"
    assert "templates" not in doc


def test_yaml_with_template(native_complex, tmp_path):
    annotation = annotate_vhh(native_complex.antibody.seq)
    template = build_cdr_masked_template(native_complex, annotation, tmp_path / "t.cif")
    spec = write_boltz_yaml(
        tmp_path / "in.yaml",
        native_complex.antibody.seq,
        native_complex.antigen.seq,
        templates=[template],
        template_force=True,
        template_threshold=4.0,
    )
    doc = yaml.safe_load(spec.path.read_text())
    entry = doc["templates"][0]
    assert entry["force"] is True
    assert entry["threshold"] == 4.0
    assert entry["chain_id"] == ["A", "B"]
    assert entry["cif"].endswith("t.cif")


def test_paired_msa_option(native_complex, tmp_path):
    spec = write_boltz_yaml(
        tmp_path / "in.yaml",
        native_complex.antibody.seq,
        native_complex.antigen.seq,
        single_sequence_antibody=False,
    )
    doc = yaml.safe_load(spec.path.read_text())
    assert "msa" not in doc["sequences"][0]["protein"]


# -------------------------------------------------------------------------- scoring

def test_zscore_handles_degenerate_columns():
    assert np.allclose(zscore(np.array([5.0, 5.0, 5.0])), 0.0)
    assert np.allclose(zscore(np.array([1.0])), 0.0)
    result = zscore(np.array([1.0, 2.0, 3.0]))
    assert result.mean() == pytest.approx(0.0)
    assert result.std() == pytest.approx(1.0)


def test_zscore_ignores_nan():
    result = zscore(np.array([1.0, 2.0, np.nan, 3.0]))
    assert np.isfinite(result).all()
    assert result[2] == 0.0


def _frame() -> pd.DataFrame:
    """A small ensemble where one scorer is perfect and one is anti-correlated."""
    dockq = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    return pd.DataFrame(
        {
            "arm": ["test"] * 5,
            "target": ["X"] * 5,
            "dockq": dockq,
            "capri_class": ["high", "medium", "medium", "acceptable", "incorrect"],
            "conf_iptm": dockq,  # perfect
            "conf_ptm": dockq[::-1],  # perfectly anti-correlated
            "shape_complementarity": dockq * 0.7 + 0.1,
            "n_clashes": (10 * (1 - dockq)).astype(int),
            "hbond_density": np.array([5.0, 4.0, 6.0, 3.0, 4.5]),  # uninformative
            "n_hbonds": (dockq * 12).astype(int),
            "n_salt_bridges": np.array([3, 2, 2, 1, 0]),
            "bsa_total": dockq * 1500 + 300,
            "n_buried_unsatisfied_polars": (30 * (1 - dockq)).astype(int),
        }
    )


def test_scorer_evaluation_ranks_a_perfect_scorer_first():
    result = evaluate_scorers(add_scores(_frame()))
    assert result.iloc[0]["scorer"] == "conf:iptm"
    assert result.iloc[0]["spearman"] == pytest.approx(1.0)
    assert result.iloc[0]["top1_dockq"] == pytest.approx(0.9)
    assert result.iloc[0]["enrichment"] == pytest.approx(1.0)


def test_scorer_evaluation_exposes_an_anticorrelated_scorer():
    result = evaluate_scorers(add_scores(_frame())).set_index("scorer")
    assert result.loc["conf:ptm", "spearman"] == pytest.approx(-1.0)
    assert result.loc["conf:ptm", "top1_dockq"] == pytest.approx(0.1)
    assert result.loc["conf:ptm", "enrichment"] < 0


def test_composite_score_is_added():
    scored = add_scores(_frame())
    assert "score:phys:composite" in scored.columns
    assert "score:hybrid:iptm+physics" in scored.columns
    assert np.isfinite(scored["score:phys:composite"]).all()


def test_evaluate_scorers_requires_ground_truth():
    with pytest.raises(KeyError, match="dockq"):
        evaluate_scorers(pd.DataFrame({"score:x": [1.0, 2.0]}))


# ------------------------------------------------------------------------- ensemble

def test_pairwise_rmsd_is_a_valid_distance_matrix(native_complex):
    annotation = annotate_vhh(native_complex.antibody.seq)
    rng = np.random.default_rng(0)
    cdr_atoms = native_complex.antibody.atom_mask_for_residues(annotation.cdr3)

    members = []
    for scale in (0.0, 0.5, 1.5):
        member = copy.deepcopy(native_complex)
        member.antibody.coords[cdr_atoms] += rng.normal(scale=scale, size=(cdr_atoms.sum(), 3))
        members.append(member)

    distance = pairwise_cdr_rmsd(members, annotation, cdrs=("cdr3",))
    assert distance.shape == (3, 3)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.allclose(distance, distance.T)
    assert distance[0, 1] < distance[0, 2], "larger perturbation must be further away"


def test_identical_structures_have_zero_diversity(native_complex):
    annotation = annotate_vhh(native_complex.antibody.seq)
    members = [copy.deepcopy(native_complex) for _ in range(3)]
    report = ensemble_report("test", members, annotation, [1.0] * 3, [0.0] * 3)
    assert report.mean_pairwise_cdr3_rmsd == pytest.approx(0.0, abs=1e-6)
    assert report.n_clusters == 1
    assert report.best_dockq == pytest.approx(1.0)
    assert report.n_acceptable == 3


def test_cluster_counting():
    # Two tight pairs, far apart.
    distance = np.array(
        [[0, 0.2, 9, 9], [0.2, 0, 9, 9], [9, 9, 0, 0.3], [9, 9, 0.3, 0]], dtype=float
    )
    assert count_clusters(distance, threshold=1.5) == 2
    assert count_clusters(distance, threshold=20.0) == 1
    assert count_clusters(np.zeros((1, 1)), threshold=1.5) == 1


# --------------------------------------------------------- chain-ID canonicalization

def test_canonical_chain_ids_survive_a_write_read_roundtrip(native_complex, tmp_path):
    """The whole point: downstream stages load by literal "A"/"B" and must get it right.

    8QF4's crystal chains are E (nanobody) and A (antigen) — note that "A" is the
    *antigen* there, so a naive round-trip would transpose the roles.
    """
    from boltz_cdr.pdb_io import load_complex, with_canonical_chain_ids

    assert native_complex.antibody.chain_id == "E"
    assert native_complex.antigen.chain_id == "A"

    canonical = with_canonical_chain_ids(native_complex, "A", "B")
    path = write_complex_cif(canonical, tmp_path / "canonical.cif")
    reloaded = load_complex(path, "A", "B")

    assert reloaded.antibody.seq == native_complex.antibody.seq
    assert reloaded.antigen.seq == native_complex.antigen.seq
    assert np.allclose(reloaded.antibody.coords, native_complex.antibody.coords, atol=1e-3)


def test_canonicalization_does_not_mutate_the_input(native_complex):
    from boltz_cdr.pdb_io import with_canonical_chain_ids

    before = native_complex.antibody.chain_id
    renamed = with_canonical_chain_ids(native_complex, "A", "B")
    assert renamed.antibody.chain_id == "A"
    assert native_complex.antibody.chain_id == before, "input must not be mutated"


def test_predicted_complex_role_assignment_then_canonicalization(native_complex, tmp_path):
    """A prediction whose chain names are the reverse of ours must still round-trip."""
    from boltz_cdr.pdb_io import (
        load_complex,
        load_predicted_complex,
        with_canonical_chain_ids,
    )

    # Emit a "prediction" with deliberately unhelpful names: antibody=Z, antigen=A.
    swapped = with_canonical_chain_ids(native_complex, "Z", "A")
    raw = write_complex_cif(swapped, tmp_path / "raw_prediction.cif")

    recovered = load_predicted_complex(raw, native_complex)
    assert recovered.antibody.seq == native_complex.antibody.seq, "role assignment failed"

    path = write_complex_cif(
        with_canonical_chain_ids(recovered, "A", "B"), tmp_path / "canonical.cif"
    )
    final = load_complex(path, "A", "B")
    assert final.antibody.seq == native_complex.antibody.seq
    assert final.antigen.seq == native_complex.antigen.seq
