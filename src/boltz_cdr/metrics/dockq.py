"""DockQ — the standard single-number quality score for a predicted complex.

Basu & Wallner (2016), PLoS ONE 11:e0161879:

    DockQ = ( fnat + 1/(1 + (iRMSD/1.5)^2) + 1/(1 + (LRMSD/8.5)^2) ) / 3

It is used here as the *target* variable for the scorer comparison in `scoring.py`: a
selection score is only useful if it correlates with DockQ, so DockQ is what we regress
the confidence and physics scores against.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from boltz_cdr.cdr import CDRAnnotation
from boltz_cdr.metrics.contacts import ContactReport, contact_report
from boltz_cdr.metrics.correspondence import ComplexCorrespondence
from boltz_cdr.metrics.rmsd import RmsdReport, interface_rmsd, ligand_rmsd, rmsd_report
from boltz_cdr.pdb_io import Complex

# CAPRI-equivalent quality bands, as defined in the DockQ paper.
CAPRI_THRESHOLDS = ((0.80, "high"), (0.49, "medium"), (0.23, "acceptable"))


def dockq_score(fnat: float, i_rmsd: float, l_rmsd: float) -> float:
    """Combine the three CAPRI components into DockQ."""
    scaled_irms = 1.0 / (1.0 + (i_rmsd / 1.5) ** 2)
    scaled_lrms = 1.0 / (1.0 + (l_rmsd / 8.5) ** 2)
    return float((fnat + scaled_irms + scaled_lrms) / 3.0)


def capri_class(dockq: float) -> str:
    for threshold, label in CAPRI_THRESHOLDS:
        if dockq >= threshold:
            return label
    return "incorrect"


@dataclass
class DockQReport:
    dockq: float
    capri_class: str
    fnat: float
    interface_rmsd: float
    ligand_rmsd: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_dockq(
    pred: Complex, native: Complex, corr: ComplexCorrespondence
) -> DockQReport:
    contacts = contact_report(pred, native, corr)
    i_rmsd, _ = interface_rmsd(pred, native, corr)
    l_rmsd = ligand_rmsd(pred, native, corr)
    score = dockq_score(contacts.fnat, i_rmsd, l_rmsd)
    return DockQReport(
        dockq=score,
        capri_class=capri_class(score),
        fnat=contacts.fnat,
        interface_rmsd=i_rmsd,
        ligand_rmsd=l_rmsd,
    )


@dataclass
class FullReport:
    """Everything we know about one predicted structure, relative to ground truth."""

    dockq: DockQReport
    rmsd: RmsdReport
    contacts: ContactReport

    def flat(self) -> dict:
        out = {"dockq": self.dockq.dockq, "capri_class": self.dockq.capri_class}
        out.update(self.rmsd.as_dict())
        out.update(self.contacts.as_dict())
        return out


def evaluate(
    pred: Complex,
    native: Complex,
    corr: ComplexCorrespondence,
    annotation: CDRAnnotation,
) -> FullReport:
    """Full ground-truth evaluation of one prediction."""
    return FullReport(
        dockq=compute_dockq(pred, native, corr),
        rmsd=rmsd_report(pred, native, corr, annotation),
        contacts=contact_report(pred, native, corr),
    )
