"""CDR-masked structural templates — Arm A.

The idea in one sentence: hand Boltz-2 back its own docked complex as a template, with the
CDR residues deleted, so that the framework–antigen relative pose is pinned while the
loops are left completely unconstrained.

Why this should work, mechanically. Boltz-2's template features are per-residue-*pair*
distograms and unit vectors, computed only over residues actually present in the template
file. Residues absent from the template contribute no pair features and receive no
restraint. Deleting the CDR rows therefore does something quite specific: it removes every
geometric statement about the loops while retaining every geometric statement relating the
framework to the antigen. The prediction problem changes from "dock this nanobody onto
this antigen and build its loops" into "build these loops, given that everything else is
already placed" — which is the sub-problem where the model's real uncertainty lives.

If the model re-derives the same CDR conformation from framework context regardless, Arm A
produces no extra diversity. That is a possible outcome, which is why ensemble diversity is
measured directly in `ensemble.py` rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from boltz_cdr.cdr import CDR_NAMES, CDRAnnotation
from boltz_cdr.pdb_io import Complex, write_complex_cif


@dataclass
class TemplateSpec:
    """A written template plus the bookkeeping needed to reference it from a YAML."""

    path: Path
    antibody_chain_id: str
    antigen_chain_id: str
    n_masked_residues: int
    masked_cdrs: tuple[str, ...]
    kept_antibody_residues: np.ndarray

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "antibody_chain_id": self.antibody_chain_id,
            "antigen_chain_id": self.antigen_chain_id,
            "n_masked_residues": self.n_masked_residues,
            "masked_cdrs": list(self.masked_cdrs),
        }


def build_cdr_masked_template(
    cx: Complex,
    annotation: CDRAnnotation,
    path: str | Path,
    *,
    mask_cdrs: tuple[str, ...] = CDR_NAMES,
    flank: int = 0,
    antibody_chain_id: str = "A",
    antigen_chain_id: str = "B",
) -> TemplateSpec:
    """Write a template mmCIF holding the docked pose with CDR residues removed.

    Parameters
    ----------
    cx
        A docked complex — normally a top-ranked Stage-0 Boltz-2 prediction.
    annotation
        CDR spans indexing `cx.antibody`.
    mask_cdrs
        Which loops to delete. Masking only `("cdr3",)` is a useful ablation: CDR3
        contributes most of the paratope, so it isolates how much of any gain comes from
        the loop that actually matters.
    flank
        Additionally delete this many residues either side of each masked loop. The
        residues immediately flanking a CDR are its anchors, and leaving them templated
        constrains the loop take-off geometry; masking them frees it at the cost of
        loosening the frame.

    Residue numbering of the retained residues is preserved, so the deleted stretches
    appear as ordinary gaps — the standard mmCIF representation of unmodeled residues,
    and what Boltz's template parser expects.
    """
    unknown = set(mask_cdrs) - set(CDR_NAMES)
    if unknown:
        msg = f"unknown CDR name(s): {sorted(unknown)}"
        raise ValueError(msg)

    to_mask: set[int] = set()
    for name in mask_cdrs:
        span = annotation[name]
        if len(span) == 0:
            continue
        lo, hi = int(span.min()) - flank, int(span.max()) + flank
        to_mask.update(range(max(lo, 0), min(hi + 1, cx.antibody.n_res)))

    keep = np.array([i for i in range(cx.antibody.n_res) if i not in to_mask], dtype=int)
    if len(keep) < 3:  # noqa: PLR2004
        msg = "masking removed almost the whole antibody chain — check the annotation"
        raise ValueError(msg)

    masked_antibody = cx.antibody.subset_residues(keep)
    masked_antibody.chain_id = antibody_chain_id
    antigen = cx.antigen.subset_residues(np.arange(cx.antigen.n_res))
    antigen.chain_id = antigen_chain_id

    template = Complex(name=f"{cx.name}_cdrmask", antibody=masked_antibody, antigen=antigen)
    written = write_complex_cif(template, path, name=f"{cx.name}_cdrmask")

    return TemplateSpec(
        path=written,
        antibody_chain_id=antibody_chain_id,
        antigen_chain_id=antigen_chain_id,
        n_masked_residues=len(to_mask),
        masked_cdrs=tuple(mask_cdrs),
        kept_antibody_residues=keep,
    )


def build_full_template(
    cx: Complex,
    path: str | Path,
    *,
    antibody_chain_id: str = "A",
    antigen_chain_id: str = "B",
) -> TemplateSpec:
    """Write the docked pose as a template with nothing removed.

    The control for Arm A: if re-predicting against a *complete* template reproduces the
    input almost exactly while the CDR-masked template produces a spread of loops, the
    diversity is attributable to the masking rather than to template conditioning per se.
    """
    antibody = cx.antibody.subset_residues(np.arange(cx.antibody.n_res))
    antibody.chain_id = antibody_chain_id
    antigen = cx.antigen.subset_residues(np.arange(cx.antigen.n_res))
    antigen.chain_id = antigen_chain_id

    template = Complex(name=f"{cx.name}_full", antibody=antibody, antigen=antigen)
    written = write_complex_cif(template, path, name=f"{cx.name}_full")
    return TemplateSpec(
        path=written,
        antibody_chain_id=antibody_chain_id,
        antigen_chain_id=antigen_chain_id,
        n_masked_residues=0,
        masked_cdrs=(),
        kept_antibody_residues=np.arange(cx.antibody.n_res),
    )
