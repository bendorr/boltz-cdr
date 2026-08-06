"""Tests for the differentiable guidance potential and the mask translation.

These guard the correctness of the gradient itself, not merely that the code runs: the
potential is differentiated by autograd, so the gradient is checked against central finite
differences and its confinement to CDR atoms is asserted exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boltz_cdr.cdr import annotate_vhh
from boltz_cdr.featurize import mock_feats_from_complex
from boltz_cdr.masks import atom_mask, atom_residue_labels, build_token_index
from boltz_cdr.potentials import (
    CDRGuidanceConfig,
    cdr_interface_energy,
    cdr_interface_gradient,
    resolve_selection,
)
from boltz_cdr.sampler import CDRSamplingConfig


@pytest.fixture(scope="module")
def system(native_complex):
    """Real coordinates, real CDR spans, float64 for finite differencing."""
    annotation = annotate_vhh(native_complex.antibody.seq)
    feats = mock_feats_from_complex(native_complex, dtype=torch.float64)
    config = CDRGuidanceConfig(
        antibody_chain=0,
        antigen_chain=1,
        cdr_residues=tuple(int(i) for i in annotation.all_indices),
    )
    return native_complex, annotation, feats, config, resolve_selection(feats, config)


# ------------------------------------------------------------------ mask translation

def test_token_index_handles_author_numbering(system):
    """8QF4's antigen is numbered 211-277; positions must still come out 0-based."""
    _, _, feats, _, _ = system
    index = build_token_index(feats)
    assert index.n_chains == 2
    for ordinal in (0, 1):
        positions = index.residue_position[index.chain_ordinal == ordinal]
        assert positions.min().item() == 0
        assert positions.max().item() == len(positions) - 1
        assert torch.equal(torch.sort(positions).values, torch.arange(len(positions)))


def test_atom_mask_selects_the_right_atoms(system):
    complex_, annotation, feats, _, _ = system
    mask = atom_mask(feats, 0, tuple(int(i) for i in annotation.all_indices))
    expected = complex_.antibody.atom_mask_for_residues(annotation.all_indices)
    # Antibody atoms come first in the concatenated ordering.
    assert mask[: complex_.antibody.n_atom].numpy().tolist() == expected.tolist()
    assert not mask[complex_.antibody.n_atom :].any(), "mask leaked onto the antigen"


def test_chain_masks_are_disjoint_and_complete(system):
    complex_, _, feats, _, _ = system
    ab = atom_mask(feats, 0)
    ag = atom_mask(feats, 1)
    assert not (ab & ag).any()
    assert (ab | ag).all()
    assert ab.sum().item() == complex_.antibody.n_atom
    assert ag.sum().item() == complex_.antigen.n_atom


def test_residue_labels_group_atoms_correctly(system):
    complex_, _, feats, _, _ = system
    labels = atom_residue_labels(feats)
    assert len(torch.unique(labels)) == complex_.antibody.n_res + complex_.antigen.n_res
    first_residue = complex_.antibody.residue_atom_indices(0)
    assert len(torch.unique(labels[first_residue])) == 1


# --------------------------------------------------------------------- forward pass

def test_energy_is_near_zero_on_a_crystal_structure(system):
    """A flat-bottom potential must not penalize a real, correct interface."""
    _, _, feats, config, selection = system
    energy = cdr_interface_energy(feats["coords"], selection, config)
    assert torch.isfinite(energy).all()
    assert energy.item() < 0.1


def test_energy_rises_when_the_interface_is_broken(system):
    _, complex_native, feats, config, selection = system[1], system[0], system[2], system[3], system[4]
    baseline = cdr_interface_energy(feats["coords"], selection, config).item()
    for distance in (2.0, 5.0, 10.0):
        moved = feats["coords"].clone()
        moved[:, : complex_native.antibody.n_atom, 0] += distance
        assert cdr_interface_energy(moved, selection, config).item() > baseline


def test_energy_is_batched(system):
    complex_, _, feats, config, selection = system
    batch = feats["coords"].repeat(4, 1, 1)
    # Displace the antibody only. Translating the whole complex is a rigid motion and
    # would leave the energy unchanged — which is itself worth asserting, below.
    batch[1, : complex_.antibody.n_atom, 0] += 5.0
    energy = cdr_interface_energy(batch, selection, config)

    assert energy.shape == (4,)
    assert energy[1] > energy[0]
    assert energy[0].item() == pytest.approx(energy[2].item())
    assert energy[2].item() == pytest.approx(energy[3].item())


def test_energy_is_invariant_to_rigid_motion(system):
    """Translating the whole complex must not change the energy."""
    _, _, feats, config, selection = system
    moved = feats["coords"] + torch.tensor([10.0, -4.0, 7.0], dtype=feats["coords"].dtype)
    assert cdr_interface_energy(moved, selection, config).item() == pytest.approx(
        cdr_interface_energy(feats["coords"], selection, config).item()
    )


# -------------------------------------------------------------------- backward pass

def test_gradient_matches_finite_differences(system):
    """The core correctness check: autograd against a central difference."""
    complex_, _, feats, config, selection = system
    x = feats["coords"].clone()
    x[:, : complex_.antibody.n_atom, 0] += 4.0
    grad = cdr_interface_gradient(x, selection, config)

    rng = np.random.default_rng(0)
    probes = selection.cdr_atoms[
        torch.as_tensor(rng.choice(len(selection.cdr_atoms), size=6, replace=False))
    ]
    eps = 1e-6
    for atom in probes.tolist():
        for dim in range(3):
            plus, minus = x.clone(), x.clone()
            plus[0, atom, dim] += eps
            minus[0, atom, dim] -= eps
            numeric = (
                cdr_interface_energy(plus, selection, config)
                - cdr_interface_energy(minus, selection, config)
            ).item() / (2 * eps)
            assert numeric == pytest.approx(grad[0, atom, dim].item(), abs=1e-6)


def test_gradient_is_confined_to_cdr_atoms(system):
    """Framework and antigen must receive exactly zero gradient, not merely small."""
    complex_, _, feats, config, selection = system
    x = feats["coords"].clone()
    x[:, : complex_.antibody.n_atom, 0] += 4.0
    grad = cdr_interface_gradient(x, selection, config)

    non_cdr = np.setdiff1d(np.arange(grad.shape[1]), selection.cdr_atoms.numpy())
    assert grad[:, non_cdr, :].abs().max().item() == 0.0
    assert grad[:, selection.cdr_atoms, :].abs().max().item() > 0.0


def test_gradient_is_finite_and_nonzero_when_perturbed(system):
    complex_, _, feats, config, selection = system
    x = feats["coords"].clone()
    x[:, : complex_.antibody.n_atom, 0] += 4.0
    grad = cdr_interface_gradient(x, selection, config)
    assert torch.isfinite(grad).all()
    assert grad.norm().item() > 0


def test_descent_reduces_the_energy(system):
    """Guidance must actually improve the geometry, not merely be differentiable."""
    complex_, _, feats, config, selection = system
    x = feats["coords"].clone()
    x[:, : complex_.antibody.n_atom, 0] += 4.0
    start = cdr_interface_energy(x, selection, config).item()
    for _ in range(80):
        x = x - 8.0 * cdr_interface_gradient(x, selection, config)
    end = cdr_interface_energy(x, selection, config).item()
    assert end < start * 1e-2


def test_descent_satisfies_the_contact_specification(system):
    """After descent, `n_contacts` CDR residues should be within `contact_distance`."""
    complex_, _, feats, config, selection = system
    x = feats["coords"].clone()
    x[:, : complex_.antibody.n_atom, 0] += 4.0
    for _ in range(150):
        x = x - 8.0 * cdr_interface_gradient(x, selection, config)

    distances = torch.cdist(
        x[0, selection.cdr_atoms], x[0, selection.epitope_atoms]
    ).min(dim=-1).values
    in_contact = torch.unique(
        selection.cdr_residue_labels[distances < config.contact_distance]
    ).numel()
    assert in_contact >= config.n_contacts


def test_empty_selection_is_a_no_op(system):
    """No CDRs specified must give zero energy and zero gradient, not a crash."""
    _, _, feats, _, _ = system
    config = CDRGuidanceConfig(antibody_chain=0, antigen_chain=1, cdr_residues=())
    selection = resolve_selection(feats, config)
    assert selection.is_empty
    assert cdr_interface_energy(feats["coords"], selection, config).item() == 0.0
    assert cdr_interface_gradient(feats["coords"], selection, config).abs().max().item() == 0.0


def test_epitope_restriction_changes_the_energy(system):
    """Naming an epitope the CDRs do not touch must make the native pose look bad.

    This is the epitope-directed-generation capability: with the true epitope named, the
    crystal structure is already optimal; with a decoy patch on the far side of the
    antigen named, the same coordinates score badly and guidance would pull the loops
    toward the named site instead.
    """
    complex_, _, feats, config, _ = system

    # Choose the five antigen residues furthest from the paratope, so the premise that
    # this patch is "somewhere else" is established rather than assumed.
    whole = resolve_selection(feats, config)
    coords = feats["coords"][0]
    to_paratope = torch.cdist(
        coords[whole.epitope_atoms], coords[whole.cdr_atoms]
    ).min(dim=-1).values
    antigen_residue_of_atom = torch.as_tensor(
        complex_.antigen.atom_res_index, device=coords.device
    )
    per_residue = torch.full((complex_.antigen.n_res,), float("inf"), dtype=coords.dtype)
    per_residue = per_residue.scatter_reduce(
        0, antigen_residue_of_atom, to_paratope, reduce="amin", include_self=True
    )
    decoy = tuple(int(i) for i in torch.topk(per_residue, 5).indices)

    decoy_config = CDRGuidanceConfig(
        antibody_chain=config.antibody_chain,
        antigen_chain=config.antigen_chain,
        cdr_residues=config.cdr_residues,
        epitope_residues=decoy,
    )
    narrow = resolve_selection(feats, decoy_config)
    assert narrow.meta["n_epitope_atoms"] < whole.meta["n_epitope_atoms"]

    native_energy = cdr_interface_energy(feats["coords"], whole, config).item()
    decoy_energy = cdr_interface_energy(feats["coords"], narrow, decoy_config).item()
    assert decoy_energy > native_energy
    # And the gradient must point somewhere: the decoy site is not satisfied.
    grad = cdr_interface_gradient(feats["coords"], narrow, decoy_config)
    assert grad.norm().item() > 0


# ------------------------------------------------------------------ sampler config

def test_noise_scale_applies_only_to_cdr_atoms(system):
    _, annotation, feats, _, _ = system
    mask = atom_mask(feats, 0, tuple(int(i) for i in annotation.all_indices))
    config = CDRSamplingConfig(cdr_atom_mask=mask, noise_scale=2.5)
    scale = config.atom_noise_scale(len(mask), torch.device("cpu"), torch.float32)

    assert scale.shape == (1, len(mask), 1)
    assert torch.all(scale[0, mask, 0] == 2.5)
    assert torch.all(scale[0, ~mask, 0] == 1.0)


def test_unit_noise_scale_is_identity(system):
    _, annotation, feats, _, _ = system
    mask = atom_mask(feats, 0, tuple(int(i) for i in annotation.all_indices))
    config = CDRSamplingConfig(cdr_atom_mask=mask, noise_scale=1.0)
    assert not config.modifies_noise
    scale = config.atom_noise_scale(len(mask), torch.device("cpu"), torch.float32)
    assert torch.all(scale == 1.0)


def test_partial_diffusion_requires_both_arguments():
    with pytest.raises(ValueError, match="partial diffusion"):
        CDRSamplingConfig(partial_diffusion_sigma=8.0)
    with pytest.raises(ValueError, match="partial diffusion"):
        CDRSamplingConfig(reference_coords=torch.zeros(10, 3))


def test_invalid_guidance_config_rejected():
    with pytest.raises(ValueError, match="n_contacts"):
        CDRGuidanceConfig(n_contacts=0)
    with pytest.raises(ValueError, match="clash_distance"):
        CDRGuidanceConfig(clash_distance=5.0, contact_distance=4.5)
