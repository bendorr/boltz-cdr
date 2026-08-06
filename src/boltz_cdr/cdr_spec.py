"""User-supplied CDR definitions, in author residue numbering.

The automatic IMGT annotator in `cdr.py` is the default, but it should never be the only
option. A user may be working with an engineered scaffold the annotator mis-reads, may
want to resample a non-canonical loop, may want to widen a CDR to include its anchors, or
may simply know something about their molecule that a sequence-alignment heuristic does
not. This module lets them say so explicitly.

**Specifications are in author residue numbers, not indices**, because author numbers are
what a user actually has in front of them in PyMOL or on the RCSB page. That choice is not
free: author numbering is arbitrary and varies wildly — of the benchmark targets, 9EZU's
antibody starts at residue 0, 8QF4's antigen at 211, 9HUR's at 108 — so every user-supplied
number is validated against the structure and a bad one is a loud error, never a silent
mis-selection.

Two input routes, one code path:

    CLI    --cdr-residues "E:27-38,56-65,105-117"
           --cdr-residues "E:cdr1=27-38,cdr3=105-117"

    YAML   cdr_residues:
             chain: E
             cdr1: "27-38"
             cdr2: [56, 57, 58, 59, 60]
             cdr3: "105-117"
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from boltz_cdr.cdr import CDR_NAMES, CDRAnnotation, annotate_vhh
from boltz_cdr.pdb_io import Chain, align_sequences

_RANGE = re.compile(r"^\s*(-?\d+)\s*(?:-\s*(-?\d+)\s*)?$")


class CDRSpecError(ValueError):
    """A user CDR specification that cannot be parsed or does not fit the structure."""


@dataclass(frozen=True)
class CDRSpec:
    """A user CDR definition: a chain ID plus author residue numbers per loop."""

    chain: str
    spans: dict[str, tuple[int, ...]]
    origin: str = "user"
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "chain": self.chain,
            "spans": {name: list(nums) for name, nums in self.spans.items()},
            "origin": self.origin,
        }

    def summary(self) -> str:
        parts = [
            f"{name.upper()}[{len(nums)}]={_compress(nums)}"
            for name, nums in sorted(self.spans.items())
            if nums
        ]
        return f"chain {self.chain}: " + "  ".join(parts)

    @property
    def all_resnums(self) -> tuple[int, ...]:
        return tuple(sorted({n for nums in self.spans.values() for n in nums}))

    # ---------------------------------------------------------------- constructors

    @classmethod
    def parse(cls, text: str, *, origin: str = "--cdr-residues") -> CDRSpec:
        """Parse the CLI form: ``CHAIN:span[,span...]``.

        Spans are assigned positionally to CDR1/2/3 unless named with ``nameupper=``.
        A single loop split across discontinuous stretches uses ``+``, e.g.
        ``cdr3=105-110+115-117``.
        """
        if not text or ":" not in text:
            msg = (
                f"{origin}: expected 'CHAIN:span[,span...]', for example "
                f"'E:27-38,56-65,105-117'. Got {text!r}."
            )
            raise CDRSpecError(msg)

        chain, _, remainder = text.partition(":")
        chain = chain.strip()
        if not chain:
            msg = f"{origin}: missing chain ID before ':' in {text!r}."
            raise CDRSpecError(msg)

        tokens = [t.strip() for t in remainder.split(",") if t.strip()]
        if not tokens:
            msg = f"{origin}: no residue spans given for chain {chain!r}."
            raise CDRSpecError(msg)

        spans: dict[str, tuple[int, ...]] = {}
        positional: list[tuple[int, ...]] = []
        for token in tokens:
            name, sep, value = token.partition("=")
            if sep:
                key = name.strip().lower()
                if key not in CDR_NAMES:
                    msg = (
                        f"{origin}: unknown loop name {name.strip()!r}; "
                        f"expected one of {', '.join(CDR_NAMES)}."
                    )
                    raise CDRSpecError(msg)
                if key in spans:
                    msg = f"{origin}: {key} specified more than once."
                    raise CDRSpecError(msg)
                spans[key] = _parse_multi(value, origin)
            else:
                positional.append(_parse_multi(token, origin))

        if positional and spans:
            msg = (
                f"{origin}: mix of named and unnamed spans. Use either all positional "
                f"('E:27-38,56-65,105-117') or all named ('E:cdr1=27-38,cdr3=105-117')."
            )
            raise CDRSpecError(msg)

        if positional:
            if len(positional) > len(CDR_NAMES):
                msg = (
                    f"{origin}: {len(positional)} positional spans given but only "
                    f"{len(CDR_NAMES)} loops exist. Name them explicitly "
                    f"('cdr3=105-110+115-117') if one loop is discontinuous."
                )
                raise CDRSpecError(msg)
            spans = dict(zip(CDR_NAMES, positional, strict=False))

        return cls(chain=chain, spans=spans, origin=origin)

    @classmethod
    def from_mapping(cls, mapping: dict, *, origin: str = "targets.yaml") -> CDRSpec:
        """Parse the YAML form: a mapping with ``chain`` plus per-loop spans."""
        if not isinstance(mapping, dict):
            msg = f"{origin}: cdr_residues must be a mapping with a 'chain' key."
            raise CDRSpecError(msg)

        chain = mapping.get("chain")
        if not chain:
            msg = f"{origin}: cdr_residues needs a 'chain' key naming the antibody chain."
            raise CDRSpecError(msg)

        spans: dict[str, tuple[int, ...]] = {}
        for key, value in mapping.items():
            if key == "chain":
                continue
            name = str(key).lower()
            if name not in CDR_NAMES:
                msg = (
                    f"{origin}: unknown key {key!r} in cdr_residues; "
                    f"expected 'chain' or one of {', '.join(CDR_NAMES)}."
                )
                raise CDRSpecError(msg)
            if isinstance(value, str):
                spans[name] = _parse_multi(value.replace(",", "+"), origin)
            elif isinstance(value, (list, tuple)):
                spans[name] = tuple(sorted({int(v) for v in value}))
            else:
                msg = f"{origin}: {key} must be a span string or a list of residue numbers."
                raise CDRSpecError(msg)

        if not any(spans.values()):
            msg = f"{origin}: cdr_residues defines no residues."
            raise CDRSpecError(msg)
        return cls(chain=str(chain), spans=spans, origin=origin)


def _parse_multi(text: str, origin: str) -> tuple[int, ...]:
    """Parse ``27-38`` / ``27`` / ``105-110+115-117`` into sorted residue numbers."""
    numbers: set[int] = set()
    for part in text.split("+"):
        match = _RANGE.match(part)
        if not match:
            msg = (
                f"{origin}: cannot parse residue span {part.strip()!r}. "
                f"Expected 'N', 'N-M', or 'N-M+P-Q'."
            )
            raise CDRSpecError(msg)
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if end < start:
            msg = f"{origin}: residue span {part.strip()!r} runs backwards."
            raise CDRSpecError(msg)
        numbers.update(range(start, end + 1))
    if not numbers:
        msg = f"{origin}: empty residue span {text.strip()!r}."
        raise CDRSpecError(msg)
    return tuple(sorted(numbers))


def _compress(numbers) -> str:
    """[27,28,29,31] -> '27-29+31', for readable summaries."""
    numbers = sorted(numbers)
    if not numbers:
        return "-"
    runs, start, previous = [], numbers[0], numbers[0]
    for value in numbers[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous))
        start = previous = value
    runs.append((start, previous))
    return "+".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


# -------------------------------------------------------------------- resolution


def resolve_spec(spec: CDRSpec, chain: Chain, *, compare_to_auto: bool = True) -> CDRAnnotation:
    """Turn author residue numbers into a `CDRAnnotation` indexed on `chain`.

    Raises `CDRSpecError` on anything that would otherwise select the wrong residues:
    the wrong chain, a residue number that does not exist, ambiguous numbering, or
    overlapping loops.
    """
    if spec.chain != chain.chain_id:
        msg = (
            f"{spec.origin}: specification names chain {spec.chain!r} but the antibody "
            f"chain here is {chain.chain_id!r}. CDRs are defined on the antibody chain; "
            f"if {spec.chain!r} is the antigen, this is the wrong chain."
        )
        raise CDRSpecError(msg)

    lookup: dict[int, list[int]] = {}
    for position, resnum in enumerate(chain.resnums.tolist()):
        lookup.setdefault(int(resnum), []).append(position)

    ambiguous = sorted(n for n, positions in lookup.items() if len(positions) > 1)
    requested = set(spec.all_resnums)
    clashing = sorted(requested & set(ambiguous))
    if clashing:
        msg = (
            f"{spec.origin}: residue number(s) {clashing} appear more than once in chain "
            f"{chain.chain_id} — the structure uses insertion codes, which author numbers "
            f"alone cannot disambiguate. Specify these loops by index instead "
            f"(cdr_spans in targets.yaml)."
        )
        raise CDRSpecError(msg)

    missing = sorted(requested - set(lookup))
    if missing:
        present = chain.resnums
        msg = (
            f"{spec.origin}: residue number(s) {missing} are not present in chain "
            f"{chain.chain_id}, which spans {present.min()}-{present.max()} "
            f"({chain.n_res} residues). Author numbering is arbitrary — check the "
            f"structure rather than assuming it starts at 1."
        )
        raise CDRSpecError(msg)

    resolved: dict[str, np.ndarray] = {}
    seen: dict[int, str] = {}
    for name in CDR_NAMES:
        numbers = spec.spans.get(name, ())
        positions = sorted(lookup[n][0] for n in numbers)
        for number in numbers:
            if number in seen and seen[number] != name:
                msg = (
                    f"{spec.origin}: residue {number} is assigned to both "
                    f"{seen[number]} and {name}; loops must not overlap."
                )
                raise CDRSpecError(msg)
            seen[number] = name
        resolved[name] = np.asarray(positions, dtype=int)

    annotation = CDRAnnotation(
        cdr1=resolved["cdr1"],
        cdr2=resolved["cdr2"],
        cdr3=resolved["cdr3"],
        seq=chain.seq,
        invariants_ok=True,
        invariant_detail={},
    )

    if compare_to_auto:
        _report_divergence(spec, annotation, chain)
    return annotation


def _report_divergence(spec: CDRSpec, annotation: CDRAnnotation, chain: Chain) -> None:
    """Compare a user spec against the automatic annotation and note the difference.

    Informational only. The user's definition always wins — the point of the override is
    to disagree with the annotator — but a large divergence is usually a numbering
    mistake, and it is cheap to say so.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto = annotate_vhh(chain.seq)
    except Exception:
        return

    user_set = set(annotation.all_indices.tolist())
    auto_set = set(auto.all_indices.tolist())
    if not auto_set:
        return
    overlap = len(user_set & auto_set) / len(user_set | auto_set)
    if overlap < 0.5:  # noqa: PLR2004
        warnings.warn(
            f"{spec.origin}: the residues you specified overlap the automatic IMGT "
            f"annotation by only {overlap:.0%} (Jaccard). That is fine if deliberate, but "
            f"check the numbering. Yours: {spec.summary()}. Automatic: {auto.summary()}",
            stacklevel=3,
        )


def annotation_for_chain(chain: Chain, spec: CDRSpec | None = None) -> CDRAnnotation:
    """The single entry point: user specification if given, automatic annotation if not."""
    if spec is None:
        return annotate_vhh(chain.seq)
    return resolve_spec(spec, chain)


def transfer_annotation(annotation: CDRAnnotation, target_seq: str) -> CDRAnnotation:
    """Carry an annotation onto a different sequence by pairwise alignment.

    Needed when a user specification is resolved against the crystal structure but has to
    be applied to a prediction whose construct differs. When the sequences are identical
    this is the identity, and that is the common case — Boltz is asked to predict exactly
    the resolved crystal sequence.
    """
    if annotation.seq == target_seq:
        return annotation

    mapping = dict(align_sequences(annotation.seq, target_seq))
    moved = {
        name: np.asarray(
            sorted(mapping[i] for i in annotation[name].tolist() if i in mapping), dtype=int
        )
        for name in CDR_NAMES
    }
    dropped = len(annotation.all_indices) - sum(len(v) for v in moved.values())
    if dropped:
        warnings.warn(
            f"transferring the CDR annotation onto a sequence of length {len(target_seq)} "
            f"dropped {dropped} residue(s) with no aligned counterpart.",
            stacklevel=2,
        )
    return CDRAnnotation(
        cdr1=moved["cdr1"],
        cdr2=moved["cdr2"],
        cdr3=moved["cdr3"],
        seq=target_seq,
        invariants_ok=annotation.invariants_ok,
        invariant_detail=annotation.invariant_detail,
    )
