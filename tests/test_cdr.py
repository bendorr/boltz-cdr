"""CDR annotation tests. No network, no Boltz."""

from __future__ import annotations

import numpy as np
import pytest

from boltz_cdr.cdr import CDR_NAMES, annotate_vhh, annotation_from_spans

# Real VHH sequences from the benchmark set, plus the reference itself.
NB_8QF4 = (
    "SEVQLLESGGGLVQAGDSLRLSCAASGRTFSAYAMGWFRQAPGKEREFVAAISWSGNSTYYADSVKGRFTISRDN"
    "AKNTVYLQMNSLKPEDTAIYYCAARKPMYRVDISKGQNYDYWGQGTQVTVSS"
)
NB_9EZU = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYGMGWFRQAPGKEREFVAAIGPFGYETYYADSVKGRFTISRDNA"
    "KNTVYLQMNSLKPEDTAVYYCAAGRWNRYGFEDQDNFDYWGQGTQVTVSSAAA"
)
# A synthetic sybody, not a camelid VHH — the annotator must not assume a camelid framework.
NB_9HUR = (
    "SSQVQLVESGGGSVQAGGSLRLSCAASGSISSITYLGWFRQAPGKEREGVAALMTTDGSTYYANSVKGRFTVSLD"
    "NAKNTVYLQMNSLKPEDTALYYCAAAENGFKIPLWEYIYTYWGQGTQVTVSA"
)


@pytest.mark.parametrize(
    ("seq", "cdr1", "cdr2", "cdr3"),
    [
        (NB_8QF4, "GRTFSAYA", "ISWSGNST", "AARKPMYRVDISKGQNYDY"),
        (NB_9EZU, "GRTFSSYG", "IGPFGYET", "AAGRWNRYGFEDQDNFDY"),
        (NB_9HUR, "GSISSITY", "LMTTDGST", "AAAENGFKIPLWEYIYTY"),
    ],
)
def test_known_cdr_spans(seq, cdr1, cdr2, cdr3):
    """Annotation reproduces the hand-checked IMGT spans for all three benchmark VHHs."""
    annotation = annotate_vhh(seq)
    sequences = annotation.sequences()
    assert sequences["cdr1"] == cdr1
    assert sequences["cdr2"] == cdr2
    assert sequences["cdr3"] == cdr3


@pytest.mark.parametrize("seq", [NB_8QF4, NB_9EZU, NB_9HUR])
def test_invariant_residues(seq):
    """The four invariant V-domain positions must land on C/W/C/W."""
    annotation = annotate_vhh(seq)
    assert annotation.invariants_ok
    assert [annotation.invariant_detail[i] for i in (23, 41, 104, 118)] == ["C", "W", "C", "W"]


@pytest.mark.parametrize("seq", [NB_8QF4, NB_9EZU, NB_9HUR])
def test_spans_are_disjoint_ordered_and_in_range(seq):
    annotation = annotate_vhh(seq)
    spans = [annotation[name] for name in CDR_NAMES]
    for span in spans:
        assert len(span) > 0
        assert span.min() >= 0
        assert span.max() < len(seq)
        assert np.all(np.diff(span) == 1), "a CDR must be a contiguous run"
    assert spans[0].max() < spans[1].min() < spans[1].max() < spans[2].min()
    combined = annotation.all_indices
    assert len(combined) == len(set(combined.tolist())), "CDRs must not overlap"


def test_cdr3_length_varies_with_insertion():
    """A CDR3 insertion must be absorbed, not shifted into the framework.

    This is the case naive IMGT number-transfer gets wrong, and the reason the annotator
    brackets between framework anchors instead of assigning numbers to loop residues.
    """
    base = annotate_vhh(NB_8QF4)
    inserted_seq = NB_8QF4.replace("AARKPMYRVDISKGQNYDY", "AARKPMYRVDISKGQNYGGGGGDY")
    inserted = annotate_vhh(inserted_seq)

    assert len(inserted.cdr3) == len(base.cdr3) + 5
    assert inserted.sequences()["cdr3"] == "AARKPMYRVDISKGQNYGGGGGDY"
    # Framework loops must be untouched by a CDR3 insertion.
    assert inserted.sequences()["cdr1"] == base.sequences()["cdr1"]
    assert inserted.sequences()["cdr2"] == base.sequences()["cdr2"]


def test_mask_marks_exactly_the_cdrs():
    annotation = annotate_vhh(NB_8QF4)
    mask = annotation.mask()
    assert mask.sum() == len(annotation.all_indices)
    assert np.array_equal(np.flatnonzero(mask), annotation.all_indices)


def test_non_antibody_sequence_warns():
    """A sequence that is not a V-domain must be flagged, not silently annotated."""
    with pytest.warns(UserWarning, match="invariant"):
        annotate_vhh("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL")


def test_explicit_spans_override():
    annotation = annotation_from_spans(
        NB_8QF4, {"cdr1": [1, 2, 3], "cdr2": [10, 11], "cdr3": [20, 21, 22]}
    )
    assert annotation.sequences()["cdr1"] == NB_8QF4[1:4]
    assert len(annotation.all_indices) == 8
