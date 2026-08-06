"""Differentiable CDR–epitope interface potential — the gradient-guided arm (B3).

Boltz-2's diffusion sampler already supports inference-time steering: at each denoising
step it asks every registered `Potential` for `dE/dx` on the current x0 prediction and
nudges the coordinates down that gradient. Boltz's own potentials hand-derive their
analytic derivatives. Ours does not need to — we write the energy as a differentiable
torch expression and let `torch.autograd` produce the gradient.

Guidance is therefore a backward pass through a differentiable objective, evaluated on the
model's own coordinate predictions once per guidance step of the reverse diffusion
trajectory.

The energy has three terms, all flat-bottomed so that a geometry which is already good
contributes exactly zero and receives exactly zero gradient:

  attraction  the `n_contacts` CDR residues closest to the epitope are pulled to within
              `contact_distance`. Only the closest few, because a real paratope buries
              10-20 of its ~34 CDR residues, not all of them — pulling every CDR residue
              onto the antigen would produce a physically absurd collapsed interface.
  repulsion   CDR atoms are pushed out of steric overlap with antigen atoms.
  epitope     optional. When a specific epitope is named, contacts are scored against
              only those antigen residues, which turns the sampler into an
              epitope-directed generator — the "conditional generation given target
              constraints" capability, obtained without retraining anything.

Gradients are masked to CDR atoms. The framework and the antigen keep whatever trajectory
the unmodified network gives them; we only reshape the loops.

This module imports torch but NOT boltz, so the energy and its gradient are testable on a
laptop with no model weights. The Boltz `Potential` subclass is built by
`make_cdr_potential()`, which imports boltz lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from boltz_cdr.masks import atom_mask, atom_residue_labels, build_token_index


@dataclass
class CDRGuidanceConfig:
    """Geometry and weighting of the CDR–epitope potential."""

    antibody_chain: int = 0
    antigen_chain: int = 1
    cdr_residues: tuple[int, ...] = ()
    epitope_residues: tuple[int, ...] | None = None

    contact_distance: float = 4.5  # A — flat-bottom edge for the attractive term
    clash_distance: float = 3.2  # A — flat-bottom edge for the repulsive term
    n_contacts: int = 8  # how many CDR residues to pull into contact
    attraction_weight: float = 1.0
    repulsion_weight: float = 2.0

    def __post_init__(self) -> None:
        if self.n_contacts < 1:
            msg = "n_contacts must be >= 1"
            raise ValueError(msg)
        if self.clash_distance >= self.contact_distance:
            msg = "clash_distance must be smaller than contact_distance"
            raise ValueError(msg)


@dataclass
class CDRAtomSelection:
    """Resolved atom indices for one system. Built once, reused every diffusion step."""

    cdr_atoms: torch.Tensor  # (n_cdr_atom,) long
    epitope_atoms: torch.Tensor  # (n_epitope_atom,) long
    cdr_residue_labels: torch.Tensor  # (n_cdr_atom,) long, groups atoms by residue
    n_atom_total: int
    meta: dict = field(default_factory=dict)
    _group_matrix: torch.Tensor | None = None

    @property
    def is_empty(self) -> bool:
        return len(self.cdr_atoms) == 0 or len(self.epitope_atoms) == 0

    def residue_group_matrix(self) -> torch.Tensor:
        """(n_residue, n_cdr_atom) boolean membership matrix, built once and cached."""
        if self._group_matrix is None:
            unique, inverse = torch.unique(self.cdr_residue_labels, return_inverse=True)
            matrix = torch.zeros(
                len(unique), len(self.cdr_residue_labels),
                dtype=torch.bool, device=self.cdr_residue_labels.device,
            )
            matrix[inverse, torch.arange(len(inverse), device=matrix.device)] = True
            self._group_matrix = matrix
        return self._group_matrix


def resolve_selection(feats: dict, cfg: CDRGuidanceConfig, *, batch: int = 0) -> CDRAtomSelection:
    """Turn a `CDRGuidanceConfig` into concrete atom indices for a featurized system."""
    index = build_token_index(feats, batch=batch)

    cdr_mask = atom_mask(feats, cfg.antibody_chain, cfg.cdr_residues, batch=batch, index=index)
    epi_mask = atom_mask(feats, cfg.antigen_chain, cfg.epitope_residues, batch=batch, index=index)

    labels = atom_residue_labels(feats, batch=batch, index=index)
    cdr_atoms = torch.nonzero(cdr_mask, as_tuple=False).squeeze(-1)

    return CDRAtomSelection(
        cdr_atoms=cdr_atoms,
        epitope_atoms=torch.nonzero(epi_mask, as_tuple=False).squeeze(-1),
        cdr_residue_labels=labels[cdr_atoms],
        n_atom_total=int(cdr_mask.shape[0]),
        meta={
            "n_cdr_atoms": int(cdr_mask.sum().item()),
            "n_epitope_atoms": int(epi_mask.sum().item()),
            "n_cdr_residues": int(torch.unique(labels[cdr_atoms]).numel()) if len(cdr_atoms) else 0,
        },
    )


def _residue_min_distance(
    dist: torch.Tensor, group_matrix: torch.Tensor
) -> torch.Tensor:
    """Minimum distance from each CDR residue to the epitope.

    `dist` is (B, A, E) atom-atom distances; `group_matrix` is a (R, A) boolean matrix
    assigning CDR atoms to residues. Returns (B, R).

    A hard minimum rather than a log-sum-exp softmin: softmin is biased low by up to
    log(M)/beta, and with M in the hundreds of epitope atoms that shifts the apparent
    contact distance by ~1 A, so the potential stops short of its own flat bottom.
    Exactness matters more than smoothness here because the threshold is a physical claim
    about contact distance, and the many small steps of the guidance loop damp the
    discontinuity as the argmin switches.
    """
    per_atom = dist.min(dim=-1).values  # (B, A)
    expanded = per_atom.unsqueeze(1).expand(-1, group_matrix.shape[0], -1)  # (B, R, A)
    masked = torch.where(
        group_matrix.unsqueeze(0), expanded, torch.full_like(expanded, float("inf"))
    )
    return masked.min(dim=-1).values


def cdr_interface_energy(
    coords: torch.Tensor,
    selection: CDRAtomSelection,
    cfg: CDRGuidanceConfig,
) -> torch.Tensor:
    """Flat-bottom CDR–epitope interface energy.

    Parameters
    ----------
    coords
        (batch, n_atom, 3) coordinates. In the sampler this is the current x0 prediction.
    selection
        Atom indices, from `resolve_selection`.
    cfg
        Weights and distance thresholds.

    Returns
    -------
    (batch,) energy — zero when every term is inside its flat bottom.
    """
    if selection.is_empty:
        return torch.zeros(coords.shape[0], device=coords.device, dtype=coords.dtype)

    cdr = coords[:, selection.cdr_atoms, :]  # (B, A, 3)
    epi = coords[:, selection.epitope_atoms, :]  # (B, E, 3)
    dist = torch.cdist(cdr, epi)  # (B, A, E)

    # --- repulsion: every CDR/antigen atom pair, flat-bottomed above clash_distance ---
    overlap = torch.clamp(cfg.clash_distance - dist, min=0.0)
    e_rep = (overlap**2).sum(dim=(-1, -2)) / max(len(selection.cdr_atoms), 1)

    # --- attraction: per-residue distance to the epitope, then the closest n_contacts ---
    per_residue = _residue_min_distance(dist, selection.residue_group_matrix())  # (B, R)

    k = min(cfg.n_contacts, per_residue.shape[-1])
    closest = torch.topk(per_residue, k, dim=-1, largest=False).values
    shortfall = torch.clamp(closest - cfg.contact_distance, min=0.0)
    e_att = (shortfall**2).mean(dim=-1)

    return cfg.attraction_weight * e_att + cfg.repulsion_weight * e_rep


def cdr_interface_gradient(
    coords: torch.Tensor,
    selection: CDRAtomSelection,
    cfg: CDRGuidanceConfig,
) -> torch.Tensor:
    """dE/dx via autograd, zeroed outside the CDR atoms.

    Called from inside the sampler's `torch.no_grad()` region, so it opens its own
    grad-enabled scope. Returns a tensor shaped like `coords`.
    """
    if selection.is_empty:
        return torch.zeros_like(coords)

    with torch.enable_grad():
        x = coords.detach().clone().requires_grad_(True)
        energy = cdr_interface_energy(x, selection, cfg).sum()
        (grad,) = torch.autograd.grad(energy, x, allow_unused=False)

    # Confine the update to the loops: the framework and antigen follow the unmodified
    # reverse-diffusion trajectory.
    masked = torch.zeros_like(grad)
    masked[:, selection.cdr_atoms, :] = grad[:, selection.cdr_atoms, :]
    return masked


def make_cdr_potential(cfg: CDRGuidanceConfig, parameters: dict | None = None):
    """Build a Boltz `Potential` wrapping the energy above.

    Imported lazily so this module stays usable without boltz installed.
    """
    from boltz.model.potentials.potentials import Potential

    class CDRInterfacePotential(Potential):
        """Boltz-2 steering potential for CDR–epitope interface geometry."""

        def __init__(self, config: CDRGuidanceConfig, params: dict) -> None:
            super().__init__(parameters=params)
            self.config = config
            self._cache: dict[int, CDRAtomSelection] = {}

        def _selection(self, feats: dict) -> CDRAtomSelection:
            # feats is rebuilt per prediction, not per step; cache on identity.
            key = id(feats)
            if key not in self._cache:
                self._cache[key] = resolve_selection(feats, self.config)
            return self._cache[key]

        def compute(self, coords, feats, parameters):
            """Energy per sample — consumed by the Feynman-Kac resampling weights."""
            return cdr_interface_energy(coords, self._selection(feats), self.config)

        def compute_gradient(self, coords, feats, parameters):
            """dE/dx per sample — consumed by the guidance update. The backward pass."""
            return cdr_interface_gradient(coords, self._selection(feats), self.config)

        # `Potential` declares THREE abstract methods — compute_args, compute_variable,
        # and compute_function. Together they are its template-method pattern for
        # *analytic* derivatives: the base `compute`/`compute_gradient` build an index,
        # evaluate a geometric variable, then apply a flat-bottom function with a
        # hand-derived derivative.
        #
        # We bypass that pattern completely by overriding compute() and
        # compute_gradient() above, because our derivative comes from autograd. But all
        # three must still be *defined*, or Python refuses to instantiate the subclass:
        #     TypeError: Can't instantiate abstract class CDRInterfacePotential
        # Defining only compute_args is not enough.

        _BYPASS = (
            "CDRInterfacePotential overrides compute() and compute_gradient() directly "
            "and does not use Potential's analytic-derivative template. Reaching this "
            "means the Boltz base class has started calling the template methods, and "
            "this class needs updating."
        )

        def compute_args(self, feats, parameters):
            raise NotImplementedError(self._BYPASS)

        def compute_variable(self, coords, index, compute_gradient=False):
            raise NotImplementedError(self._BYPASS)

        def compute_function(self, value, *args, **kwargs):
            raise NotImplementedError(self._BYPASS)

    defaults = {
        "guidance_interval": 1,
        "guidance_weight": 0.2,
        "resampling_weight": 1.0,
    }
    defaults.update(parameters or {})
    return CDRInterfacePotential(cfg, defaults)
