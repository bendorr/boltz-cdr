"""Inter-chain contact recapitulation.

RMSD says how far the prediction is from the truth in Angstroms; it does not say whether
the model found the right *interaction*. Two poses with similar ligand RMSD can differ in
whether they reproduce the actual epitope. These metrics score the prediction against the
native contact set directly:

  fnat / fnonnat   the CAPRI quantities — recall of native contacts, and the fraction of
                   predicted contacts that are spurious
  precision/F1     the same information as a retrieval problem
  epitope_recall   which *antigen* residues the binder actually touches — the number a
                   campaign cares about when it is targeting a specific site
  paratope_recall  the antibody-side equivalent
  jaccard          overlap of the binarized residue-pair contact maps
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from boltz_cdr.metrics.correspondence import ComplexCorrespondence
from boltz_cdr.pdb_io import Chain, Complex

CONTACT_CUTOFF = 5.0  # Angstrom, heavy-atom; the DockQ convention for fnat


def residue_contact_map(
    chain_a: Chain, chain_b: Chain, *, cutoff: float = CONTACT_CUTOFF
) -> np.ndarray:
    """(n_res_a, n_res_b) boolean map: True where any heavy-atom pair is within `cutoff`."""
    d = np.linalg.norm(chain_a.coords[:, None, :] - chain_b.coords[None, :, :], axis=-1)
    close = d < cutoff
    out = np.zeros((chain_a.n_res, chain_b.n_res), dtype=bool)
    ia, ib = np.nonzero(close)
    out[chain_a.atom_res_index[ia], chain_b.atom_res_index[ib]] = True
    return out


def contact_pairs(cx: Complex, *, cutoff: float = CONTACT_CUTOFF) -> set[tuple[int, int]]:
    """Native-style contact set as (antibody_res_index, antigen_res_index) pairs."""
    cmap = residue_contact_map(cx.antibody, cx.antigen, cutoff=cutoff)
    return {(int(i), int(j)) for i, j in zip(*np.nonzero(cmap), strict=True)}


@dataclass
class ContactReport:
    fnat: float
    fnonnat: float
    precision: float
    recall: float
    f1: float
    jaccard: float
    epitope_recall: float
    paratope_recall: float
    n_native_contacts: int
    n_predicted_contacts: int
    n_shared_contacts: int
    n_native_epitope_residues: int
    n_epitope_residues_recovered: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def contact_report(
    pred: Complex,
    native: Complex,
    corr: ComplexCorrespondence,
    *,
    cutoff: float = CONTACT_CUTOFF,
) -> ContactReport:
    """Score predicted inter-chain contacts against the native contact set.

    Both contact sets are expressed in *native* residue indices so they are directly
    comparable; predicted contacts involving residues with no native counterpart (e.g. a
    purification tag) are dropped rather than counted as non-native, since they cannot be
    right or wrong.
    """
    ab_p2n = dict(zip(corr.ab_a.tolist(), corr.ab_b.tolist(), strict=True))
    ag_p2n = dict(zip(corr.ag_a.tolist(), corr.ag_b.tolist(), strict=True))

    native_set = contact_pairs(native, cutoff=cutoff)
    pred_set = {
        (ab_p2n[i], ag_p2n[j])
        for i, j in contact_pairs(pred, cutoff=cutoff)
        if i in ab_p2n and j in ag_p2n
    }

    shared = native_set & pred_set
    n_nat, n_pred, n_share = len(native_set), len(pred_set), len(shared)

    fnat = n_share / n_nat if n_nat else float("nan")
    fnonnat = (n_pred - n_share) / n_pred if n_pred else float("nan")
    precision = n_share / n_pred if n_pred else 0.0
    recall = fnat if n_nat else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    union = len(native_set | pred_set)
    jaccard = n_share / union if union else float("nan")

    native_epitope = {j for _, j in native_set}
    pred_epitope = {j for _, j in pred_set}
    native_paratope = {i for i, _ in native_set}
    pred_paratope = {i for i, _ in pred_set}

    return ContactReport(
        fnat=fnat,
        fnonnat=fnonnat,
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        epitope_recall=(
            len(native_epitope & pred_epitope) / len(native_epitope)
            if native_epitope
            else float("nan")
        ),
        paratope_recall=(
            len(native_paratope & pred_paratope) / len(native_paratope)
            if native_paratope
            else float("nan")
        ),
        n_native_contacts=n_nat,
        n_predicted_contacts=n_pred,
        n_shared_contacts=n_share,
        n_native_epitope_residues=len(native_epitope),
        n_epitope_residues_recovered=len(native_epitope & pred_epitope),
    )


def contact_difference(
    pred: Complex,
    native: Complex,
    corr: ComplexCorrespondence,
    *,
    cutoff: float = CONTACT_CUTOFF,
) -> dict[str, list[tuple[int, int, str, str]]]:
    """Per-residue-pair breakdown for inspection: recovered / missed / spurious contacts.

    Each entry is (antibody_resnum, antigen_resnum, antibody_aa, antigen_aa) in native
    numbering, which is what you want when eyeballing whether the model found the real
    epitope or a plausible decoy site next to it.
    """
    ab_p2n = dict(zip(corr.ab_a.tolist(), corr.ab_b.tolist(), strict=True))
    ag_p2n = dict(zip(corr.ag_a.tolist(), corr.ag_b.tolist(), strict=True))

    native_set = contact_pairs(native, cutoff=cutoff)
    pred_set = {
        (ab_p2n[i], ag_p2n[j])
        for i, j in contact_pairs(pred, cutoff=cutoff)
        if i in ab_p2n and j in ag_p2n
    }

    def label(pairs) -> list[tuple[int, int, str, str]]:
        return sorted(
            (
                int(native.antibody.resnums[i]),
                int(native.antigen.resnums[j]),
                native.antibody.seq[i],
                native.antigen.seq[j],
            )
            for i, j in pairs
        )

    return {
        "recovered": label(native_set & pred_set),
        "missed": label(native_set - pred_set),
        "spurious": label(pred_set - native_set),
    }


def epitope_residues(cx: Complex, *, cutoff: float = CONTACT_CUTOFF) -> np.ndarray:
    """Antigen residue indices contacted by the antibody."""
    cmap = residue_contact_map(cx.antibody, cx.antigen, cutoff=cutoff)
    return np.flatnonzero(cmap.any(axis=0))


def paratope_residues(cx: Complex, *, cutoff: float = CONTACT_CUTOFF) -> np.ndarray:
    """Antibody residue indices contacting the antigen."""
    cmap = residue_contact_map(cx.antibody, cx.antigen, cutoff=cutoff)
    return np.flatnonzero(cmap.any(axis=1))
