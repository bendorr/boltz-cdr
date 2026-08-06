"""Ensemble-level analysis: does the sampling change actually produce new structures?

This is the module that tests the project's central mechanical claim. Arms A and B are
supposed to widen the distribution of CDR conformations rather than redraw the same one.
That claim is falsifiable and cheap to falsify:

  diversity   mean pairwise CDR RMSD within an arm's ensemble, measured after
              superimposing on the antibody framework so that rigid-body pose differences
              do not masquerade as loop diversity.
  n_clusters  distinct conformations at a given RMSD threshold — a diversity number that
              does not get inflated by one wild outlier the way a mean does.
  coverage    best-of-N DockQ. Diversity is only useful if the ensemble contains something
              *better*; an arm that samples widely and misses every time has bought
              nothing.

Equal diversity between the baseline and Arm A would mean the template masking is not
freeing the loops — a negative result the harness is built to detect rather than obscure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from boltz_cdr.cdr import CDR_NAMES, CDRAnnotation
from boltz_cdr.metrics.correspondence import (
    BACKBONE,
    build_correspondence,
    matched_atoms,
    subset_correspondence,
)
from boltz_cdr.metrics.rmsd import apply_transform, kabsch, rmsd
from boltz_cdr.pdb_io import Complex


def pairwise_cdr_rmsd(
    structures: list[Complex],
    annotation: CDRAnnotation,
    *,
    cdrs: tuple[str, ...] = ("cdr3",),
    atom_names: tuple[str, ...] = BACKBONE,
) -> np.ndarray:
    """(n, n) matrix of CDR backbone RMSDs, each pair superimposed on the framework.

    `annotation` indexes the first structure; correspondences to the others are built by
    sequence alignment, so this tolerates the ensemble members differing in length.
    """
    n = len(structures)
    out = np.zeros((n, n))
    loop_residues = np.concatenate([annotation[c] for c in cdrs]) if cdrs else annotation.all_indices

    for i in range(n):
        for j in range(i + 1, n):
            value = _pair_cdr_rmsd(structures[i], structures[j], loop_residues, atom_names)
            out[i, j] = out[j, i] = value
    return out


def _pair_cdr_rmsd(
    a: Complex, b: Complex, loop_residues: np.ndarray, atom_names: tuple[str, ...]
) -> float:
    corr = build_correspondence(a, b)
    framework = np.setdiff1d(corr.ab_a, loop_residues)
    fw_a, fw_b = subset_correspondence(corr.ab_a, corr.ab_b, framework)
    x_fw_a, x_fw_b = matched_atoms(a.antibody, fw_a, b.antibody, fw_b, atom_names)
    if len(x_fw_a) < 3:  # noqa: PLR2004
        return float("nan")

    rot, trans = kabsch(x_fw_a, x_fw_b)
    loop_a, loop_b = subset_correspondence(corr.ab_a, corr.ab_b, loop_residues)
    x_a, x_b = matched_atoms(a.antibody, loop_a, b.antibody, loop_b, atom_names)
    if len(x_a) == 0:
        return float("nan")
    return rmsd(apply_transform(x_a, rot, trans), x_b)


@dataclass
class EnsembleReport:
    arm: str
    n_samples: int
    mean_pairwise_cdr3_rmsd: float
    max_pairwise_cdr3_rmsd: float
    mean_pairwise_allcdr_rmsd: float
    n_clusters: int
    cluster_threshold: float
    best_dockq: float
    mean_dockq: float
    worst_dockq: float
    best_ligand_rmsd: float
    n_acceptable: int

    def as_dict(self) -> dict:
        return asdict(self)


def count_clusters(distance: np.ndarray, threshold: float) -> int:
    """Number of clusters under average-linkage hierarchical clustering."""
    n = len(distance)
    if n <= 1:
        return n
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    condensed = squareform(np.nan_to_num(distance, nan=0.0), checks=False)
    labels = fcluster(linkage(condensed, method="average"), t=threshold, criterion="distance")
    return len(np.unique(labels))


def ensemble_report(
    arm: str,
    structures: list[Complex],
    annotation: CDRAnnotation,
    dockq: np.ndarray | list[float],
    ligand_rmsd: np.ndarray | list[float],
    *,
    cluster_threshold: float = 1.5,
    acceptable_dockq: float = 0.23,
) -> EnsembleReport:
    """Summarize one arm's ensemble."""
    dockq = np.asarray(dockq, dtype=float)
    ligand_rmsd = np.asarray(ligand_rmsd, dtype=float)

    d_cdr3 = pairwise_cdr_rmsd(structures, annotation, cdrs=("cdr3",))
    d_all = pairwise_cdr_rmsd(structures, annotation, cdrs=CDR_NAMES)
    off_diagonal = ~np.eye(len(structures), dtype=bool)

    def mean_off(matrix: np.ndarray) -> float:
        if len(structures) < 2:  # noqa: PLR2004
            return 0.0
        return float(np.nanmean(matrix[off_diagonal]))

    return EnsembleReport(
        arm=arm,
        n_samples=len(structures),
        mean_pairwise_cdr3_rmsd=mean_off(d_cdr3),
        max_pairwise_cdr3_rmsd=float(np.nanmax(d_cdr3)) if len(structures) > 1 else 0.0,
        mean_pairwise_allcdr_rmsd=mean_off(d_all),
        n_clusters=count_clusters(d_cdr3, cluster_threshold),
        cluster_threshold=cluster_threshold,
        best_dockq=float(np.nanmax(dockq)) if len(dockq) else float("nan"),
        mean_dockq=float(np.nanmean(dockq)) if len(dockq) else float("nan"),
        worst_dockq=float(np.nanmin(dockq)) if len(dockq) else float("nan"),
        best_ligand_rmsd=float(np.nanmin(ligand_rmsd)) if len(ligand_rmsd) else float("nan"),
        n_acceptable=int(np.sum(dockq >= acceptable_dockq)) if len(dockq) else 0,
    )
