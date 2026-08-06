"""Residue correspondence between a prediction and its crystal structure.

Boltz renumbers chains from 1 and predicts the full input construct, while the deposited
structure uses author numbering and may be missing disordered termini. Every metric
therefore needs an explicit residue mapping rather than an assumption that index i in one
structure is index i in the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boltz_cdr.pdb_io import Chain, Complex, align_sequences

BACKBONE = ("N", "CA", "C", "O")


@dataclass
class ComplexCorrespondence:
    """Paired residue indices; `*_a` indexes structure A, `*_b` indexes structure B."""

    ab_a: np.ndarray
    ab_b: np.ndarray
    ag_a: np.ndarray
    ag_b: np.ndarray
    ab_identity: float
    ag_identity: float

    @property
    def n_ab(self) -> int:
        return len(self.ab_a)

    @property
    def n_ag(self) -> int:
        return len(self.ag_a)

    def summary(self) -> str:
        return (
            f"antibody {self.n_ab} residues matched (id={self.ab_identity:.2f}), "
            f"antigen {self.n_ag} matched (id={self.ag_identity:.2f})"
        )


def build_correspondence(a: Complex, b: Complex) -> ComplexCorrespondence:
    """Align A to B chain-wise and return matched residue indices."""
    ab_a, ab_b, ab_id = _chain_pairs(a.antibody, b.antibody)
    ag_a, ag_b, ag_id = _chain_pairs(a.antigen, b.antigen)
    return ComplexCorrespondence(ab_a, ab_b, ag_a, ag_b, ab_id, ag_id)


def _chain_pairs(ca: Chain, cb: Chain) -> tuple[np.ndarray, np.ndarray, float]:
    pairs = align_sequences(ca.seq, cb.seq)
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int), 0.0
    ia = np.array([p[0] for p in pairs], dtype=int)
    ib = np.array([p[1] for p in pairs], dtype=int)
    identical = sum(1 for i, j in pairs if ca.seq[i] == cb.seq[j])
    return ia, ib, identical / len(pairs)


def matched_atoms(
    chain_a: Chain,
    idx_a: np.ndarray,
    chain_b: Chain,
    idx_b: np.ndarray,
    atom_names: tuple[str, ...] = ("CA",),
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates for atoms present in both structures, in matched order.

    Returns (X_a, X_b), each (n_matched, 3). Residues missing a requested atom in either
    structure are silently dropped — this is the correct behavior for crystal structures
    with partially-modeled side chains.
    """
    xa, xb = [], []
    for ra, rb in zip(idx_a, idx_b, strict=True):
        map_a = _atom_lookup(chain_a, int(ra))
        map_b = _atom_lookup(chain_b, int(rb))
        for name in atom_names:
            if name in map_a and name in map_b:
                xa.append(chain_a.coords[map_a[name]])
                xb.append(chain_b.coords[map_b[name]])
    if not xa:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.asarray(xa), np.asarray(xb)


def _atom_lookup(chain: Chain, res_index: int) -> dict[str, int]:
    idx = chain.residue_atom_indices(res_index)
    return {str(chain.atom_names[i]): int(i) for i in idx}


def subset_correspondence(
    corr_a: np.ndarray, corr_b: np.ndarray, keep_a: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict a correspondence to residues of A listed in `keep_a`, preserving order."""
    keep = np.isin(corr_a, np.asarray(keep_a, dtype=int))
    return corr_a[keep], corr_b[keep]
