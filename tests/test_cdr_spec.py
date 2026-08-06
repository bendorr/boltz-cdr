"""User-supplied CDR definitions: parsing, resolution, validation, and precedence.

The failure mode this guards against is not a crash — it is a specification that parses,
runs, and quietly selects the wrong residues. Author numbering is arbitrary (8QF4's
antibody starts at 2, 9EZU's at 0), so most of these tests are about rejecting bad input
loudly rather than about the happy path.
"""

from __future__ import annotations

import numpy as np
import pytest

from boltz_cdr.cdr import CDR_NAMES, annotate_vhh
from boltz_cdr.cdr_spec import (
    CDRSpec,
    CDRSpecError,
    annotation_for_chain,
    resolve_spec,
    transfer_annotation,
)

# ------------------------------------------------------------------------- parsing

def test_positional_spans_map_to_cdr123():
    spec = CDRSpec.parse("E:27-38,56-65,105-117")
    assert spec.chain == "E"
    assert spec.spans["cdr1"] == tuple(range(27, 39))
    assert spec.spans["cdr2"] == tuple(range(56, 66))
    assert spec.spans["cdr3"] == tuple(range(105, 118))


def test_named_spans():
    spec = CDRSpec.parse("E:cdr3=105-117,cdr1=27-38")
    assert set(spec.spans) == {"cdr1", "cdr3"}
    assert spec.spans["cdr3"] == tuple(range(105, 118))


def test_single_residue_and_discontinuous_spans():
    spec = CDRSpec.parse("B:cdr1=42,cdr3=105-110+115-117")
    assert spec.spans["cdr1"] == (42,)
    assert spec.spans["cdr3"] == (105, 106, 107, 108, 109, 110, 115, 116, 117)


def test_fewer_than_three_spans_is_allowed():
    spec = CDRSpec.parse("E:105-117")
    assert spec.spans["cdr1"] == tuple(range(105, 118))
    assert "cdr3" not in spec.spans


def test_whitespace_is_tolerated():
    assert CDRSpec.parse(" E : 27 - 38 , 56-65 ").spans["cdr1"] == tuple(range(27, 39))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("no_colon", "expected 'CHAIN:span"),
        (":27-38", "missing chain ID"),
        ("E:", "no residue spans"),
        ("E:banana", "cannot parse residue span"),
        ("E:38-27", "runs backwards"),
        ("E:cdr9=27-38", "unknown loop name"),
        ("E:cdr1=27-38,cdr1=40-50", "specified more than once"),
        ("E:cdr1=27-38,56-65", "mix of named and unnamed"),
        ("E:1-2,3-4,5-6,7-8", "only 3 loops exist"),
    ],
)
def test_malformed_specifications_are_rejected(text, match):
    with pytest.raises(CDRSpecError, match=match):
        CDRSpec.parse(text)


# --------------------------------------------------------------------- yaml mapping

def test_mapping_form_accepts_strings_and_lists():
    spec = CDRSpec.from_mapping(
        {"chain": "E", "cdr1": "28-35", "cdr2": [53, 54, 55], "cdr3": "99-117"}
    )
    assert spec.chain == "E"
    assert spec.spans["cdr1"] == tuple(range(28, 36))
    assert spec.spans["cdr2"] == (53, 54, 55)
    assert spec.spans["cdr3"] == tuple(range(99, 118))


def test_mapping_comma_separated_string_is_one_loop():
    """In YAML a comma inside a loop's span means 'also these', not 'next loop'."""
    spec = CDRSpec.from_mapping({"chain": "E", "cdr3": "99-105,110-112"})
    assert spec.spans["cdr3"] == tuple(list(range(99, 106)) + list(range(110, 113)))


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        ({"cdr1": "1-5"}, "needs a 'chain' key"),
        ({"chain": "E", "cdr9": "1-5"}, "unknown key"),
        ({"chain": "E", "cdr1": 27}, "must be a span string or a list"),
        ("not a mapping", "must be a mapping"),
    ],
)
def test_malformed_mappings_are_rejected(mapping, match):
    with pytest.raises(CDRSpecError, match=match):
        CDRSpec.from_mapping(mapping)


# ------------------------------------------------------------ resolution against a chain

def test_author_numbering_reproduces_the_automatic_annotation(native_complex):
    """The key round-trip.

    8QF4's antibody chain starts at author residue 2, so the automatic CDR1 at indices
    26-33 is author residues 28-35. Specifying those numbers by hand must produce a
    byte-identical annotation — which is what makes the printed residue numbers in
    `05_backward_pass_demo.py` a safe starting point for a user to edit.
    """
    auto = annotate_vhh(native_complex.antibody.seq)
    spec = CDRSpec.parse("E:28-35,53-60,99-117")
    manual = resolve_spec(spec, native_complex.antibody)

    for name in CDR_NAMES:
        assert np.array_equal(manual[name], auto[name]), name
    assert manual.sequences() == auto.sequences()


def test_offset_is_applied_not_assumed(native_complex):
    """Author residue 2 is index 0 here; a spec starting at 2 must select the N-terminus."""
    spec = CDRSpec.parse("E:cdr1=2-4")
    resolved = resolve_spec(spec, native_complex.antibody, compare_to_auto=False)
    assert np.array_equal(resolved.cdr1, np.array([0, 1, 2]))
    assert resolved.sequences()["cdr1"] == native_complex.antibody.seq[:3]


def test_wrong_chain_is_rejected(native_complex):
    """Naming the antigen chain must fail loudly — it is a very easy mistake."""
    spec = CDRSpec.parse("A:28-35")  # A is 8QF4's antigen
    with pytest.raises(CDRSpecError, match="names chain 'A' but the antibody chain here is 'E'"):
        resolve_spec(spec, native_complex.antibody)


def test_missing_residue_numbers_are_rejected_with_the_valid_range(native_complex):
    spec = CDRSpec.parse("E:9990-9995")
    with pytest.raises(CDRSpecError, match=r"not present in chain E, which spans 2-128"):
        resolve_spec(spec, native_complex.antibody)


def test_off_by_one_past_the_end_is_caught(native_complex):
    """The realistic typo: assuming numbering starts at 1 when it starts at 2."""
    last = int(native_complex.antibody.resnums.max())
    with pytest.raises(CDRSpecError, match="not present"):
        resolve_spec(CDRSpec.parse(f"E:{last}-{last + 1}"), native_complex.antibody)


def test_overlapping_loops_are_rejected(native_complex):
    spec = CDRSpec.parse("E:cdr1=28-40,cdr2=35-45")
    with pytest.raises(CDRSpecError, match="assigned to both"):
        resolve_spec(spec, native_complex.antibody, compare_to_auto=False)


def test_ambiguous_numbering_is_rejected():
    """Insertion codes (Kabat 100A/100B) collapse to duplicate author numbers."""
    from boltz_cdr.pdb_io import Chain

    chain = Chain(
        chain_id="H",
        resnums=np.array([100, 100, 101]),  # 100 and 100A both read as 100
        resnames=["ALA", "GLY", "SER"],
        seq="AGS",
        atom_names=np.array(["CA", "CA", "CA"], dtype="<U4"),
        atom_elements=np.array(["C", "C", "C"], dtype="<U2"),
        atom_res_index=np.array([0, 1, 2]),
        coords=np.zeros((3, 3)),
        bfactors=np.zeros(3),
    )
    with pytest.raises(CDRSpecError, match="appear more than once"):
        resolve_spec(CDRSpec.parse("H:100-101"), chain, compare_to_auto=False)


def test_wildly_divergent_specification_warns_but_is_honored(native_complex):
    """The user's definition always wins; a big divergence is only flagged."""
    spec = CDRSpec.parse("E:cdr1=3-10")  # framework, nowhere near a real CDR
    with pytest.warns(UserWarning, match="overlap the automatic IMGT annotation"):
        resolved = resolve_spec(spec, native_complex.antibody)
    assert len(resolved.cdr1) == 8
    assert len(resolved.cdr2) == 0


def test_no_warning_when_the_specification_agrees(native_complex, recwarn):
    resolve_spec(CDRSpec.parse("E:28-35,53-60,99-117"), native_complex.antibody)
    assert not [w for w in recwarn if "overlap the automatic" in str(w.message)]


# ------------------------------------------------------------------------ dispatch

def test_annotation_for_chain_falls_back_to_the_annotator(native_complex):
    auto = annotate_vhh(native_complex.antibody.seq)
    fallback = annotation_for_chain(native_complex.antibody, None)
    assert np.array_equal(fallback.all_indices, auto.all_indices)


def test_annotation_for_chain_uses_the_specification(native_complex):
    spec = CDRSpec.parse("E:cdr3=99-105")
    resolved = annotation_for_chain(native_complex.antibody, spec)
    assert len(resolved.cdr3) == 7
    assert len(resolved.cdr1) == 0


# ------------------------------------------------------------------------ transfer

def test_transfer_to_an_identical_sequence_is_the_identity(native_complex):
    annotation = annotate_vhh(native_complex.antibody.seq)
    moved = transfer_annotation(annotation, native_complex.antibody.seq)
    assert moved is annotation


def test_transfer_across_an_n_terminal_extension(native_complex):
    """A user spec resolved on the crystal must survive a longer predicted construct."""
    annotation = annotate_vhh(native_complex.antibody.seq)
    extended = "GSGS" + native_complex.antibody.seq
    moved = transfer_annotation(annotation, extended)

    assert moved.sequences() == annotation.sequences()
    assert np.array_equal(moved.all_indices, annotation.all_indices + 4)


def test_transfer_warns_when_residues_are_lost(native_complex):
    annotation = annotate_vhh(native_complex.antibody.seq)
    truncated = native_complex.antibody.seq[:60]  # cuts CDR3 off entirely
    with pytest.warns(UserWarning, match="dropped"):
        moved = transfer_annotation(annotation, truncated)
    assert len(moved.cdr3) == 0


# ----------------------------------------------------------------------- precedence

def test_precedence_cli_over_yaml_over_auto(tmp_path):
    """--cdr-residues beats the YAML block, which beats the automatic annotator."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    from _common import load_targets

    targets_yaml = tmp_path / "targets.yaml"
    targets_yaml.write_text(
        "version: 1\ntargets:\n"
        "  - id: 8QF4\n    name: test\n    antibody_chain: E\n    antigen_chain: A\n"
        "    cdr_residues:\n      chain: E\n      cdr3: '99-105'\n"
    )
    pdb_dir = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "pdb"

    from_yaml = load_targets(targets_yaml, pdb_dir)[0]
    assert from_yaml.cdr_source == "user"
    assert len(from_yaml.annotation.cdr3) == 7
    assert len(from_yaml.annotation.cdr1) == 0

    from_cli = load_targets(targets_yaml, pdb_dir, cdr_residues="E:cdr3=99-110")[0]
    assert len(from_cli.annotation.cdr3) == 12, "CLI must override the YAML block"

    plain = tmp_path / "plain.yaml"
    plain.write_text(
        "version: 1\ntargets:\n"
        "  - id: 8QF4\n    name: test\n    antibody_chain: E\n    antigen_chain: A\n"
    )
    auto = load_targets(plain, pdb_dir)[0]
    assert auto.cdr_source == "automatic IMGT"
    assert len(auto.annotation.cdr3) == 19


def test_target_annotation_for_transfers_a_user_spec(tmp_path):
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from _common import load_targets

    targets_yaml = tmp_path / "t.yaml"
    targets_yaml.write_text(
        "version: 1\ntargets:\n"
        "  - id: 8QF4\n    name: test\n    antibody_chain: E\n    antigen_chain: A\n"
        "    cdr_residues:\n      chain: E\n      cdr3: '99-117'\n"
    )
    pdb_dir = pathlib.Path(__file__).resolve().parents[1] / "data" / "pdb"
    target = load_targets(targets_yaml, pdb_dir)[0]

    same = target.annotation_for(target.antibody_sequence)
    assert np.array_equal(same.all_indices, target.annotation.all_indices)

    extended = "GSGS" + target.antibody_sequence
    moved = target.annotation_for(extended)
    assert np.array_equal(moved.all_indices, target.annotation.all_indices + 4)
    assert moved.sequences()["cdr3"] == target.annotation.sequences()["cdr3"]
