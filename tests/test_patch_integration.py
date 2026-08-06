"""Integration tests for the three modules that touch Boltz.

These run the vendored sampler for real against a stub Boltz (`tests/fake_boltz.py`),
which is the only way to exercise `patch.py`, `sampler.py`, and `run.py` without a GPU and
model weights. They check our wiring, not Boltz's numerics: that both `get_potentials`
bindings are rebound, that the CDR code paths execute, that shapes survive atom padding,
and that the guidance hook is actually reached.
"""

from __future__ import annotations

import pytest
import torch

from tests import fake_boltz

N_ATOM = 40
N_CDR = 12


@pytest.fixture
def boltz_stub():
    teardown = fake_boltz.install()
    import boltz_cdr.patch as patch_module

    patch_module.uninstall()
    yield
    patch_module.uninstall()
    teardown()


@pytest.fixture
def cdr_mask():
    mask = torch.zeros(N_ATOM, dtype=torch.bool)
    mask[:N_CDR] = True
    return mask


def make_feats(n_atom: int = N_ATOM, n_token: int = 8):
    """A feature dict shaped like Boltz's, with two chains."""
    atom_to_token = torch.zeros(1, n_atom, n_token)
    per = n_atom // n_token
    for a in range(n_atom):
        atom_to_token[0, a, min(a // per, n_token - 1)] = 1.0
    return {
        "asym_id": torch.tensor([[0] * (n_token // 2) + [1] * (n_token - n_token // 2)]),
        "residue_index": torch.tensor([list(range(n_token // 2)) * 2][0]).unsqueeze(0),
        "token_pad_mask": torch.ones(1, n_token, dtype=torch.bool),
        "atom_pad_mask": torch.ones(1, n_atom, dtype=torch.bool),
        "atom_to_token": atom_to_token,
    }


def steering(**overrides):
    from boltz_cdr.patch import steering_args

    args = steering_args(guidance=True, fk_steering=False, physical_guidance=False)
    args.update(overrides)
    return args


def run_sampler(model, feats, multiplicity=2, **kwargs):
    from boltz_cdr.sampler import cdr_selective_sample

    return cdr_selective_sample(
        model,
        atom_mask=torch.ones(1, model.n_atom, dtype=torch.bool),
        multiplicity=multiplicity,
        steering_args=steering(**kwargs),
        feats=feats,
    )


# ---------------------------------------------------------------- patch install/uninstall

def test_install_rebinds_both_get_potentials_bindings(boltz_stub, cdr_mask):
    """The diffusionv2 binding must be rebound too, or Arm B silently becomes baseline."""
    from boltz.model.modules import diffusionv2
    from boltz.model.potentials import potentials as boltz_potentials

    from boltz_cdr import patch
    from boltz_cdr.potentials import CDRGuidanceConfig

    before_pot = boltz_potentials.get_potentials
    before_diff = diffusionv2.get_potentials
    assert before_pot is before_diff

    status = patch.install(guidance=CDRGuidanceConfig(cdr_residues=(0, 1)))
    assert status.installed and status.sampler_patched and status.potential_registered
    assert boltz_potentials.get_potentials is not before_pot
    assert diffusionv2.get_potentials is not before_diff, (
        "diffusionv2 keeps its own import-time binding; failing to rebind it means our "
        "potential is never registered and the run is silently the baseline"
    )
    assert diffusionv2.get_potentials is boltz_potentials.get_potentials

    patch.uninstall()
    assert boltz_potentials.get_potentials is before_pot
    assert diffusionv2.get_potentials is before_diff


def test_install_swaps_the_sampler(boltz_stub):
    from boltz.model.modules import diffusionv2

    from boltz_cdr import patch
    from boltz_cdr.sampler import cdr_selective_sample

    original = diffusionv2.AtomDiffusion.sample
    patch.install()
    assert diffusionv2.AtomDiffusion.sample is cdr_selective_sample
    patch.uninstall()
    assert diffusionv2.AtomDiffusion.sample is original


def test_wrapper_appends_our_potential(boltz_stub):
    from boltz.model.potentials import potentials as boltz_potentials

    from boltz_cdr import patch
    from boltz_cdr.potentials import CDRGuidanceConfig

    patch.install(guidance=CDRGuidanceConfig(cdr_residues=(0, 1)))
    assert boltz_potentials.get_potentials({}, boltz2=False) == [], "non-boltz2 must be untouched"
    found = boltz_potentials.get_potentials({}, boltz2=True)
    assert len(found) == 1
    assert type(found[0]).__name__ == "CDRInterfacePotential"


def test_version_guard_rejects_wrong_version(cdr_mask):
    teardown = fake_boltz.install(version="9.9.9")
    try:
        from boltz_cdr import patch

        with pytest.raises(RuntimeError, match="written against boltz"):
            patch.install()
        # ...but can be overridden deliberately.
        status = patch.install(strict_version=False)
        assert status.installed
        patch.uninstall()
    finally:
        teardown()


def test_cdr_sampling_context_restores_state(boltz_stub, cdr_mask):
    from boltz_cdr import patch
    from boltz_cdr.sampler import CDRSamplingConfig

    assert patch.active_sampling_config() is None
    config = CDRSamplingConfig(cdr_atom_mask=cdr_mask, noise_scale=2.0)
    with patch.cdr_sampling(config):
        assert patch.active_sampling_config() is config
    assert patch.active_sampling_config() is None


# ------------------------------------------------------------------- the sampler runs

def test_sampler_runs_unmodified(boltz_stub):
    """With no active config the vendored copy must behave like stock."""
    model = fake_boltz.FakeAtomDiffusion(N_ATOM)
    out = run_sampler(model, make_feats())
    coords = out["sample_atom_coords"]
    assert coords.shape == (2, N_ATOM, 3)
    assert torch.isfinite(coords).all()
    assert len(model.forward_calls) > 0


def test_lambda_one_is_bitwise_identical_to_no_config(boltz_stub, cdr_mask):
    """λ = 1.0 must reproduce stock sampling exactly — this is what makes it a control."""
    from boltz_cdr import patch
    from boltz_cdr.sampler import CDRSamplingConfig

    feats = make_feats()
    torch.manual_seed(0)
    baseline = run_sampler(fake_boltz.FakeAtomDiffusion(N_ATOM), feats)["sample_atom_coords"]

    torch.manual_seed(0)
    with patch.cdr_sampling(CDRSamplingConfig(cdr_atom_mask=cdr_mask, noise_scale=1.0)):
        same = run_sampler(fake_boltz.FakeAtomDiffusion(N_ATOM), feats)["sample_atom_coords"]

    assert torch.equal(baseline, same)


def test_noise_scaling_changes_the_trajectory(boltz_stub, cdr_mask):
    from boltz_cdr import patch
    from boltz_cdr.sampler import CDRSamplingConfig

    feats = make_feats()
    torch.manual_seed(0)
    baseline = run_sampler(fake_boltz.FakeAtomDiffusion(N_ATOM), feats)["sample_atom_coords"]

    torch.manual_seed(0)
    with patch.cdr_sampling(CDRSamplingConfig(cdr_atom_mask=cdr_mask, noise_scale=4.0)):
        scaled = run_sampler(fake_boltz.FakeAtomDiffusion(N_ATOM), feats)["sample_atom_coords"]

    assert torch.isfinite(scaled).all()
    assert not torch.equal(baseline, scaled)


# ------------------------------------------------- partial diffusion, including padding

def test_partial_diffusion_truncates_the_schedule(boltz_stub, cdr_mask):
    """A lower sigma_start must mean strictly fewer denoising steps."""
    from boltz_cdr import patch
    from boltz_cdr.sampler import CDRSamplingConfig

    feats = make_feats()
    reference = torch.randn(N_ATOM, 3)

    full = fake_boltz.FakeAtomDiffusion(N_ATOM, num_sampling_steps=12)
    run_sampler(full, feats)
    n_full = len(full.forward_calls)

    partial = fake_boltz.FakeAtomDiffusion(N_ATOM, num_sampling_steps=12)
    config = CDRSamplingConfig(
        cdr_atom_mask=cdr_mask, partial_diffusion_sigma=8.0, reference_coords=reference
    )
    with patch.cdr_sampling(config):
        out = run_sampler(partial, feats)

    assert torch.isfinite(out["sample_atom_coords"]).all()
    assert len(partial.forward_calls) < n_full
    assert max(partial.forward_calls) <= 8.0 * (1 + full.gamma_0) + 1e-6


def test_partial_diffusion_survives_atom_padding(boltz_stub):
    """Boltz pads the atom dimension; a mask sized to the unpadded structure must still work.

    This is the realistic case: `build_sampling_config` derives the CDR mask and reference
    coordinates from the crystal structure, which has fewer atoms than Boltz's padded
    tensor. Broadcasting a short mask against a padded coordinate tensor is a hard error,
    so the sampler has to reconcile the lengths itself.
    """
    from boltz_cdr import patch
    from boltz_cdr.sampler import CDRSamplingConfig

    padded = N_ATOM + 17  # Boltz pads to a block size; the mask does not know that
    model = fake_boltz.FakeAtomDiffusion(padded)
    feats = make_feats(n_atom=padded)

    short_mask = torch.zeros(N_ATOM, dtype=torch.bool)
    short_mask[:N_CDR] = True
    short_reference = torch.randn(N_ATOM, 3)

    config = CDRSamplingConfig(
        cdr_atom_mask=short_mask,
        partial_diffusion_sigma=8.0,
        reference_coords=short_reference,
    )
    with patch.cdr_sampling(config):
        out = run_sampler(model, feats)

    assert out["sample_atom_coords"].shape == (2, padded, 3)
    assert torch.isfinite(out["sample_atom_coords"]).all()


def test_noise_scale_survives_atom_padding(boltz_stub):
    from boltz_cdr import patch
    from boltz_cdr.sampler import CDRSamplingConfig

    padded = N_ATOM + 17
    model = fake_boltz.FakeAtomDiffusion(padded)
    short_mask = torch.zeros(N_ATOM, dtype=torch.bool)
    short_mask[:N_CDR] = True

    with patch.cdr_sampling(CDRSamplingConfig(cdr_atom_mask=short_mask, noise_scale=3.0)):
        out = run_sampler(model, make_feats(n_atom=padded))
    assert torch.isfinite(out["sample_atom_coords"]).all()


def test_sigma_start_above_sigma_max_does_not_crash(boltz_stub, cdr_mask):
    """Degenerate configuration: nothing to truncate. Must degrade, not explode."""
    from boltz_cdr import patch
    from boltz_cdr.sampler import CDRSamplingConfig

    model = fake_boltz.FakeAtomDiffusion(N_ATOM)
    config = CDRSamplingConfig(
        cdr_atom_mask=cdr_mask,
        partial_diffusion_sigma=1e9,
        reference_coords=torch.randn(N_ATOM, 3),
    )
    with patch.cdr_sampling(config):
        out = run_sampler(model, make_feats())
    assert torch.isfinite(out["sample_atom_coords"]).all()


# ------------------------------------------------------------- guidance reaches our code

def test_guidance_hook_calls_our_potential(boltz_stub):
    """End-to-end through the real `patch.install()` path.

    Nothing is stubbed beyond Boltz itself: `install(guidance=...)` registers the
    potential, the patched sampler resolves it via `get_potentials`, and Boltz's guidance
    loop must reach `compute_gradient`. A spy on the registry records the instance the
    sampler actually obtained, so this fails if the wiring breaks anywhere along the way.
    """
    from boltz.model.potentials import potentials as boltz_potentials

    from boltz_cdr import patch
    from boltz_cdr.potentials import CDRGuidanceConfig

    guidance = CDRGuidanceConfig(
        antibody_chain=0, antigen_chain=1, cdr_residues=(0, 1), n_contacts=1
    )
    patch.install(guidance=guidance, guidance_parameters={"guidance_weight": 0.5})

    calls = {"energy": 0, "grad": 0}
    registered = boltz_potentials.get_potentials
    seen: list = []

    def spying_registry(steering_args, boltz2=False):
        found = registered(steering_args, boltz2=boltz2)
        for potential in found:
            if type(potential).__name__ != "CDRInterfacePotential" or potential in seen:
                continue
            seen.append(potential)
            real_compute, real_grad = potential.compute, potential.compute_gradient

            def compute(coords, f, p, _r=real_compute):
                calls["energy"] += 1
                return _r(coords, f, p)

            def gradient(coords, f, p, _r=real_grad):
                calls["grad"] += 1
                out = _r(coords, f, p)
                assert torch.isfinite(out).all()
                return out

            potential.compute, potential.compute_gradient = compute, gradient
        return found

    boltz_potentials.get_potentials = spying_registry
    out = run_sampler(fake_boltz.FakeAtomDiffusion(N_ATOM), make_feats(), num_gd_steps=2)

    assert seen, "patch.install(guidance=...) did not register CDRInterfacePotential"
    assert calls["grad"] > 0, "Boltz's guidance loop never reached compute_gradient"
    assert torch.isfinite(out["sample_atom_coords"]).all()


def test_guidance_is_absent_without_registration(boltz_stub):
    """The negative control: install() with no guidance must register nothing."""
    from boltz.model.potentials import potentials as boltz_potentials

    from boltz_cdr import patch

    patch.install()  # no guidance argument
    assert boltz_potentials.get_potentials({}, boltz2=True) == []


def test_potential_conforms_to_the_boltz_abc(boltz_stub):
    """Our potential must be constructible as a real `Potential` subclass."""
    from boltz.model.potentials.potentials import Potential

    from boltz_cdr.potentials import CDRGuidanceConfig, make_cdr_potential

    potential = make_cdr_potential(CDRGuidanceConfig(cdr_residues=(0, 1)))
    assert isinstance(potential, Potential)
    params = potential.compute_parameters(0.5)
    assert params["guidance_weight"] > 0
    with pytest.raises(NotImplementedError):
        potential.compute_args({}, {})


# ----------------------------------------------------------------------------- run.py

def test_run_prediction_refuses_on_patch_state_mismatch(boltz_stub, tmp_path):
    from boltz_cdr.run import run_prediction

    with pytest.raises(RuntimeError, match="patch state mismatch"):
        run_prediction(tmp_path / "in.yaml", tmp_path / "out", expect_patched=True)


def test_run_prediction_accepts_matching_state(boltz_stub, tmp_path):
    from boltz_cdr import patch
    from boltz_cdr.run import run_prediction

    patch.install()
    run_prediction(tmp_path / "in.yaml", tmp_path / "out", expect_patched=True, use_msa_server=False)
    patch.uninstall()
    run_prediction(tmp_path / "in.yaml", tmp_path / "out", expect_patched=False, use_msa_server=False)


def test_collect_predictions_pairs_structures_with_confidence(tmp_path):
    from boltz_cdr.run import collect_predictions

    predictions = tmp_path / "boltz_results_x" / "predictions" / "rec"
    predictions.mkdir(parents=True)
    for k in (0, 1, 2):
        (predictions / f"rec_model_{k}.cif").write_text("data_x\n")
    (predictions / "confidence_rec_model_0.json").write_text('{"iptm": 0.8}')
    (predictions / "confidence_rec_model_1.json").write_text('{"iptm": 0.6}')

    found = collect_predictions(tmp_path)
    assert len(found) == 3
    assert [p.model_index for p in found] == [0, 1, 2]
    assert all(p.record_id == "rec" for p in found)
    assert found[0].confidence() == {"iptm": 0.8}
    assert found[2].confidence() == {}, "a missing sidecar must give {} rather than raise"


def test_describe_environment_runs_without_boltz():
    from boltz_cdr.run import describe_environment

    info = describe_environment()
    assert "python" in info and "torch" in info


# ------------------------------------------- residue-based CDR spec (the preferred path)

def test_cdr_residues_resolve_against_boltz_features(boltz_stub):
    """Specifying CDRs by residue must produce the same mask `masks.atom_mask` would.

    This is the path that removes the atom-ordering assumption: the mask is derived from
    Boltz's own `atom_to_token` mapping rather than from an external structure.
    """
    from boltz_cdr.masks import atom_mask
    from boltz_cdr.sampler import CDRSamplingConfig

    feats = make_feats()
    config = CDRSamplingConfig(cdr_residues=(0, 1), antibody_chain=0, noise_scale=2.0)
    resolved = config.fitted_cdr_mask(N_ATOM, torch.device("cpu"), feats)

    expected = atom_mask(feats, 0, (0, 1))
    assert torch.equal(resolved, expected)
    assert resolved.any(), "residue spec resolved to an empty mask"


def test_cdr_residues_survive_padding(boltz_stub):
    from boltz_cdr.sampler import CDRSamplingConfig

    padded = N_ATOM + 17
    config = CDRSamplingConfig(cdr_residues=(0, 1), antibody_chain=0, noise_scale=2.0)
    mask = config.fitted_cdr_mask(padded, torch.device("cpu"), make_feats(n_atom=padded))
    assert len(mask) == padded
    assert mask.any()


def test_residue_spec_runs_end_to_end(boltz_stub):
    from boltz_cdr import patch
    from boltz_cdr.sampler import CDRSamplingConfig

    model = fake_boltz.FakeAtomDiffusion(N_ATOM)
    config = CDRSamplingConfig(cdr_residues=(0, 1), antibody_chain=0, noise_scale=3.0)
    with patch.cdr_sampling(config):
        out = run_sampler(model, make_feats())
    assert torch.isfinite(out["sample_atom_coords"]).all()


def test_mask_length_mismatch_warns(boltz_stub):
    """The legacy atom-mask path must complain loudly rather than silently misalign."""
    from boltz_cdr.sampler import CDRSamplingConfig

    short = torch.zeros(N_ATOM, dtype=torch.bool)
    short[:N_CDR] = True
    config = CDRSamplingConfig(cdr_atom_mask=short, noise_scale=2.0)
    with pytest.warns(UserWarning, match="may not line up with Boltz's atom ordering"):
        config.fitted_cdr_mask(N_ATOM + 17, torch.device("cpu"))


def test_noise_scaling_without_any_cdr_spec_is_rejected():
    from boltz_cdr.sampler import CDRSamplingConfig

    with pytest.raises(ValueError, match="cdr_residues"):
        CDRSamplingConfig(noise_scale=2.0)
