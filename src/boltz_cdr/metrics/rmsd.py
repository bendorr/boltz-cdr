"""RMSD family for antibody–antigen complexes.

Four different RMSDs answer four different questions, and reporting only one of them
hides the failure mode that matters:

  complex_rmsd  — global fit. Dominated by whichever chain is larger; a good score here
                  can coexist with a completely wrong epitope.
  ligand_rmsd   — superimpose on the antigen, then measure the antibody. This is the
                  docking-accuracy number: it asks "is the binder in the right place on
                  the target?" and is the primary metric for this project.
  interface_rmsd— local accuracy of the contact region only.
  per-CDR RMSD  — superimpose on the antibody *framework*, then measure each CDR. This
                  isolates loop-conformation error from rigid-body pose error, which is
                  exactly the decomposition the method is built around.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from boltz_cdr.cdr import CDR_NAMES, CDRAnnotation
from boltz_cdr.metrics.correspondence import (
    BACKBONE,
    ComplexCorrespondence,
    matched_atoms,
    subset_correspondence,
)
from boltz_cdr.pdb_io import Complex


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rigid transform taking `mobile` onto `target`.

    Returns (R, t) such that `mobile @ R.T + t` is the superposed mobile set.
    """
    if len(mobile) < 3:  # noqa: PLR2004
        msg = f"need >=3 points to superimpose, got {len(mobile)}"
        raise ValueError(msg)
    mc, tc = mobile.mean(0), target.mean(0)
    cov = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, tc - rot @ mc


def apply_transform(x: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return x @ rot.T + trans


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def superposed_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """RMSD after optimal superposition of the same point set."""
    if len(mobile) < 3:  # noqa: PLR2004
        return float("nan")
    rot, trans = kabsch(mobile, target)
    return rmsd(apply_transform(mobile, rot, trans), target)


@dataclass
class RmsdReport:
    complex_rmsd: float
    ligand_rmsd: float
    interface_rmsd: float
    antibody_internal_rmsd: float
    antigen_internal_rmsd: float
    cdr1_rmsd: float
    cdr2_rmsd: float
    cdr3_rmsd: float
    n_interface_residues: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def complex_rmsd(pred: Complex, native: Complex, corr: ComplexCorrespondence) -> float:
    """Global CA RMSD of both chains after a single joint superposition."""
    xa_ab, xb_ab = matched_atoms(pred.antibody, corr.ab_a, native.antibody, corr.ab_b)
    xa_ag, xb_ag = matched_atoms(pred.antigen, corr.ag_a, native.antigen, corr.ag_b)
    return superposed_rmsd(np.vstack([xa_ab, xa_ag]), np.vstack([xb_ab, xb_ag]))


def ligand_rmsd(
    pred: Complex,
    native: Complex,
    corr: ComplexCorrespondence,
    *,
    atom_names: tuple[str, ...] = BACKBONE,
) -> float:
    """Superimpose on the antigen, then measure antibody backbone RMSD.

    The docking-accuracy metric: how far is the binder from where it should be, given a
    correctly-placed target?
    """
    ag_p, ag_n = matched_atoms(pred.antigen, corr.ag_a, native.antigen, corr.ag_b, atom_names)
    ab_p, ab_n = matched_atoms(pred.antibody, corr.ab_a, native.antibody, corr.ab_b, atom_names)
    if len(ag_p) < 3 or len(ab_p) == 0:  # noqa: PLR2004
        return float("nan")
    rot, trans = kabsch(ag_p, ag_n)
    return rmsd(apply_transform(ab_p, rot, trans), ab_n)


def interface_residues(
    cx: Complex, *, cutoff: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Residue indices of (antibody, antigen) with any heavy atom within `cutoff`.

    10 A is the DockQ convention for defining the interface used by iRMSD.
    """
    d = np.linalg.norm(
        cx.antibody.coords[:, None, :] - cx.antigen.coords[None, :, :], axis=-1
    )
    close = d < cutoff
    ab = np.unique(cx.antibody.atom_res_index[np.any(close, axis=1)])
    ag = np.unique(cx.antigen.atom_res_index[np.any(close, axis=0)])
    return ab, ag


def interface_rmsd(
    pred: Complex,
    native: Complex,
    corr: ComplexCorrespondence,
    *,
    cutoff: float = 10.0,
) -> tuple[float, int]:
    """Backbone RMSD over native interface residues, superimposed on those residues."""
    ab_native_iface, ag_native_iface = interface_residues(native, cutoff=cutoff)

    # Interface is defined on the native, then pulled back through the correspondence.
    ab_b, ab_a = subset_correspondence(corr.ab_b, corr.ab_a, ab_native_iface)
    ag_b, ag_a = subset_correspondence(corr.ag_b, corr.ag_a, ag_native_iface)

    ab_p, ab_n = matched_atoms(pred.antibody, ab_a, native.antibody, ab_b, BACKBONE)
    ag_p, ag_n = matched_atoms(pred.antigen, ag_a, native.antigen, ag_b, BACKBONE)
    xp, xn = np.vstack([ab_p, ag_p]), np.vstack([ab_n, ag_n])
    n_res = len(ab_b) + len(ag_b)
    if len(xp) < 3:  # noqa: PLR2004
        return float("nan"), n_res
    return superposed_rmsd(xp, xn), n_res


def per_cdr_rmsd(
    pred: Complex,
    native: Complex,
    corr: ComplexCorrespondence,
    annotation: CDRAnnotation,
    *,
    atom_names: tuple[str, ...] = BACKBONE,
) -> dict[str, float]:
    """CDR backbone RMSD after superimposing on the antibody framework only.

    `annotation` indexes the *predicted* antibody chain. Superimposing on framework
    rather than on the CDRs themselves is what makes this a measure of loop conformation
    rather than of loop placement.
    """
    framework = np.setdiff1d(corr.ab_a, annotation.all_indices)
    fw_a, fw_b = subset_correspondence(corr.ab_a, corr.ab_b, framework)
    fw_p, fw_n = matched_atoms(pred.antibody, fw_a, native.antibody, fw_b, atom_names)
    if len(fw_p) < 3:  # noqa: PLR2004
        return dict.fromkeys((f"{c}_rmsd" for c in CDR_NAMES), float("nan"))

    rot, trans = kabsch(fw_p, fw_n)
    out: dict[str, float] = {}
    for name in CDR_NAMES:
        loop_a, loop_b = subset_correspondence(corr.ab_a, corr.ab_b, annotation[name])
        xp, xn = matched_atoms(pred.antibody, loop_a, native.antibody, loop_b, atom_names)
        out[f"{name}_rmsd"] = (
            rmsd(apply_transform(xp, rot, trans), xn) if len(xp) else float("nan")
        )
    return out


def rmsd_report(
    pred: Complex,
    native: Complex,
    corr: ComplexCorrespondence,
    annotation: CDRAnnotation,
) -> RmsdReport:
    """All RMSD metrics in one pass."""
    i_rmsd, n_iface = interface_rmsd(pred, native, corr)
    cdrs = per_cdr_rmsd(pred, native, corr, annotation)

    ab_p, ab_n = matched_atoms(pred.antibody, corr.ab_a, native.antibody, corr.ab_b)
    ag_p, ag_n = matched_atoms(pred.antigen, corr.ag_a, native.antigen, corr.ag_b)

    return RmsdReport(
        complex_rmsd=complex_rmsd(pred, native, corr),
        ligand_rmsd=ligand_rmsd(pred, native, corr),
        interface_rmsd=i_rmsd,
        antibody_internal_rmsd=superposed_rmsd(ab_p, ab_n),
        antigen_internal_rmsd=superposed_rmsd(ag_p, ag_n),
        cdr1_rmsd=cdrs["cdr1_rmsd"],
        cdr2_rmsd=cdrs["cdr2_rmsd"],
        cdr3_rmsd=cdrs["cdr3_rmsd"],
        n_interface_residues=n_iface,
    )
