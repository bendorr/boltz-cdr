"""IMGT CDR annotation for VHH / nanobody domains, without ANARCI.

ANARCI is the standard tool but needs HMMER, which is a deployment liability in a Colab
notebook. For a single-domain V region we do not need a full HMM: the four IMGT framework
regions are structurally rigid and highly conserved, so numbering transfers reliably by
pairwise alignment to one reference VHH whose IMGT anchors are known.

The trick that makes this robust to hypervariable loop length is that we never assign IMGT
numbers to CDR residues at all. We locate the *framework anchors* by alignment and define
each CDR as the query residues bracketed between two anchors. Insertions of any length in
CDR3 are then absorbed automatically, which is exactly the case that naive
number-transfer gets wrong.

Correctness is checked against the four invariant residues of the V-domain fold
(Cys23, Trp41, Cys104, Trp118 in IMGT numbering); a mismatch raises or warns.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from boltz_cdr.pdb_io import align_sequences

# Reference VHH: 1ZVH chain A (camelid anti-lysozyme VHH).
# IMGT anchor indices below were assigned by hand from the canonical IMGT V-domain
# layout and verified against the four invariant residues (see _REF_INVARIANTS).
_REF_SEQ = (
    "DVQLVESGGGSVQAGGSLRLSCAASGYIASINYLGWFRQAPGKEREGVAAVSPAGGTPYY"
    "ADSVKGRFTVSLDNAENTVYLQMNSLKPEDTALYYCAAARQGWYIPLNSYGYNYWGQGTQVTVS"
)

# 0-based indices into _REF_SEQ.
_REF_ANCHORS = {
    "fr1_end": 24,  # IMGT 26  — last framework-1 residue
    "fr2_start": 33,  # IMGT 39  — first framework-2 residue
    "fr2_end": 49,  # IMGT 55  — last framework-2 residue
    "fr3_start": 58,  # IMGT 66  — first framework-3 residue
    "fr3_end": 95,  # IMGT 104 — the second invariant Cys
    "fr4_start": 114,  # IMGT 118 — the Trp of the WGxG motif
}

# (reference index, expected residue, IMGT position) for the invariant fold positions.
_REF_INVARIANTS = ((21, "C", 23), (35, "W", 41), (95, "C", 104), (114, "W", 118))

CDR_NAMES = ("cdr1", "cdr2", "cdr3")


@dataclass
class CDRAnnotation:
    """CDR spans as 0-based residue indices into the sequence that was annotated."""

    cdr1: np.ndarray
    cdr2: np.ndarray
    cdr3: np.ndarray
    seq: str
    invariants_ok: bool
    invariant_detail: dict[int, str]

    def __getitem__(self, key: str) -> np.ndarray:
        return getattr(self, key)

    @property
    def all_indices(self) -> np.ndarray:
        """Every CDR residue index, sorted — the paratope loop set."""
        return np.sort(np.concatenate([self.cdr1, self.cdr2, self.cdr3]))

    def as_dict(self) -> dict[str, list[int]]:
        return {n: [int(i) for i in self[n]] for n in CDR_NAMES}

    def sequences(self) -> dict[str, str]:
        return {n: "".join(self.seq[i] for i in self[n]) for n in CDR_NAMES}

    def mask(self, n_res: int | None = None) -> np.ndarray:
        """(n_res,) boolean mask that is True on CDR residues."""
        n = n_res if n_res is not None else len(self.seq)
        m = np.zeros(n, dtype=bool)
        idx = self.all_indices
        m[idx[idx < n]] = True
        return m

    def summary(self) -> str:
        seqs = self.sequences()
        parts = [f"{n.upper()}[{len(self[n])}]={seqs[n]}" for n in CDR_NAMES]
        flag = "" if self.invariants_ok else "  (!) invariant check failed"
        return "  ".join(parts) + flag


def annotate_vhh(seq: str, *, strict: bool = False) -> CDRAnnotation:
    """Annotate IMGT CDR1/2/3 of a single VHH domain.

    Parameters
    ----------
    seq
        One-letter amino-acid sequence of the VHH domain.
    strict
        Raise instead of warning when the invariant-residue check fails.
    """
    seq = seq.upper()
    pairs = align_sequences(seq, _REF_SEQ)
    ref_to_query = {r: q for q, r in pairs}

    invariant_detail: dict[int, str] = {}
    ok = True
    for ref_idx, expected, imgt in _REF_INVARIANTS:
        q = ref_to_query.get(ref_idx)
        observed = seq[q] if q is not None else "-"
        invariant_detail[imgt] = observed
        if observed != expected:
            ok = False
    if not ok:
        msg = (
            "VHH invariant-residue check failed; CDR spans may be unreliable. "
            f"IMGT 23/41/104/118 observed as "
            f"{'/'.join(invariant_detail[i] for i in (23, 41, 104, 118))}, expected C/W/C/W."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

    anchors = {
        name: _resolve_anchor(ref_to_query, ref_idx, len(seq))
        for name, ref_idx in _REF_ANCHORS.items()
    }
    spans = {
        "cdr1": _between(anchors["fr1_end"], anchors["fr2_start"]),
        "cdr2": _between(anchors["fr2_end"], anchors["fr3_start"]),
        "cdr3": _between(anchors["fr3_end"], anchors["fr4_start"]),
    }
    for name, span in spans.items():
        if len(span) == 0:
            warnings.warn(f"{name} annotated as empty — check the input sequence.", stacklevel=2)

    return CDRAnnotation(
        cdr1=spans["cdr1"],
        cdr2=spans["cdr2"],
        cdr3=spans["cdr3"],
        seq=seq,
        invariants_ok=ok,
        invariant_detail=invariant_detail,
    )


def _resolve_anchor(ref_to_query: dict[int, int], ref_idx: int, n_query: int) -> int:
    """Query index for a reference anchor, walking outward if that column is a gap."""
    if ref_idx in ref_to_query:
        return ref_to_query[ref_idx]
    for offset in range(1, 8):
        for probe in (ref_idx - offset, ref_idx + offset):
            if probe in ref_to_query:
                # Shift back by the same amount so the anchor stays in register.
                return int(np.clip(ref_to_query[probe] + (ref_idx - probe), 0, n_query - 1))
    msg = f"could not locate framework anchor at reference index {ref_idx}"
    raise ValueError(msg)


def _between(start_anchor: int, end_anchor: int) -> np.ndarray:
    """Residue indices strictly between two framework anchors."""
    lo, hi = start_anchor + 1, end_anchor
    if hi <= lo:
        return np.zeros(0, dtype=int)
    return np.arange(lo, hi, dtype=int)


def cdr_atom_mask(chain, annotation: CDRAnnotation) -> np.ndarray:
    """(n_atom,) boolean mask over a `pdb_io.Chain`, True for atoms in any CDR."""
    return chain.atom_mask_for_residues(annotation.all_indices)


def cdr_backbone_only_mask(chain, annotation: CDRAnnotation) -> np.ndarray:
    """CDR atoms restricted to N, CA, C, O — used for backbone-level comparisons."""
    return cdr_atom_mask(chain, annotation) & np.isin(chain.atom_names, ("N", "CA", "C", "O"))


def annotation_from_spans(seq: str, spans: dict[str, list[int]]) -> CDRAnnotation:
    """Build an annotation from explicit spans (the `cdr_spans` override in targets.yaml)."""
    return CDRAnnotation(
        cdr1=np.asarray(spans["cdr1"], dtype=int),
        cdr2=np.asarray(spans["cdr2"], dtype=int),
        cdr3=np.asarray(spans["cdr3"], dtype=int),
        seq=seq.upper(),
        invariants_ok=True,
        invariant_detail={},
    )
