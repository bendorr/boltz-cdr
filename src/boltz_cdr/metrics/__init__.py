"""Ground-truth evaluation metrics. Pure numpy/scipy — no torch, no Boltz, no GPU."""

from boltz_cdr.metrics.contacts import (
    ContactReport,
    contact_difference,
    contact_report,
    epitope_residues,
    paratope_residues,
    residue_contact_map,
)
from boltz_cdr.metrics.correspondence import (
    ComplexCorrespondence,
    build_correspondence,
    matched_atoms,
)
from boltz_cdr.metrics.dockq import (
    DockQReport,
    FullReport,
    capri_class,
    compute_dockq,
    dockq_score,
    evaluate,
)
from boltz_cdr.metrics.interface import (
    InterfaceReport,
    buried_sasa,
    hydrogen_bonds,
    interface_report,
    salt_bridges,
    sasa,
    shape_complementarity,
)
from boltz_cdr.metrics.rmsd import (
    RmsdReport,
    complex_rmsd,
    interface_rmsd,
    kabsch,
    ligand_rmsd,
    per_cdr_rmsd,
    rmsd_report,
    superposed_rmsd,
)

__all__ = [
    "ComplexCorrespondence", "ContactReport", "DockQReport", "FullReport",
    "InterfaceReport", "RmsdReport", "build_correspondence", "buried_sasa",
    "capri_class", "complex_rmsd", "compute_dockq", "contact_difference",
    "contact_report", "dockq_score", "epitope_residues", "evaluate",
    "hydrogen_bonds", "interface_report", "interface_rmsd", "kabsch",
    "ligand_rmsd", "matched_atoms", "paratope_residues", "per_cdr_rmsd",
    "residue_contact_map", "rmsd_report", "salt_bridges", "sasa",
    "shape_complementarity", "superposed_rmsd",
]
