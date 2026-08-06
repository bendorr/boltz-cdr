"""Runtime installation of the CDR modifications into an unmodified Boltz-2.

Two patch points, both minimal and both reversible:

  1. `AtomDiffusion.sample`            -> `sampler.cdr_selective_sample`   (Arm B1, B2)
  2. `potentials.get_potentials`       -> appends our CDR interface potential (Arm B3)

Patching at runtime rather than vendoring a fork keeps this repository small, keeps the
delta against upstream explicit, and means a `pip install -U boltz` is a one-line version
bump rather than a merge. The cost is coupling to Boltz internals, which is why
`install()` hard-checks the installed version: the vendored `sample()` copy is only
guaranteed faithful for the version it was taken from, and silently running a stale copy
against a newer Boltz would be far worse than refusing to start.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from boltz_cdr.potentials import CDRGuidanceConfig, make_cdr_potential
from boltz_cdr.sampler import CDRSamplingConfig, cdr_selective_sample

SUPPORTED_BOLTZ_VERSIONS = ("2.2.1",)

# Set by `cdr_sampling(...)`; read by the patched sampler on every call. A module-level
# handle is used because `AtomDiffusion.sample` has no argument we could thread this
# through, and predictions are run one configuration at a time.
_ACTIVE: CDRSamplingConfig | None = None
_INSTALLED = False
_ORIGINALS: dict[str, object] = {}


@dataclass
class PatchStatus:
    installed: bool
    boltz_version: str
    sampler_patched: bool
    potential_registered: bool

    def __str__(self) -> str:
        return (
            f"boltz_cdr patch: installed={self.installed} boltz={self.boltz_version} "
            f"sampler={self.sampler_patched} potential={self.potential_registered}"
        )


def boltz_version() -> str:
    from importlib.metadata import version

    return version("boltz")


def active_sampling_config() -> CDRSamplingConfig | None:
    """Current CDR sampling configuration, or None for stock behavior."""
    return _ACTIVE


def install(
    guidance: CDRGuidanceConfig | None = None,
    *,
    guidance_parameters: dict | None = None,
    strict_version: bool = True,
) -> PatchStatus:
    """Patch Boltz-2 in place. Idempotent.

    Parameters
    ----------
    guidance
        If given, registers the differentiable CDR-epitope potential (Arm B3). Boltz must
        additionally be run with `contact_guidance_update` or `fk_steering` enabled for
        the sampler to call it.
    guidance_parameters
        Overrides for the potential's steering parameters (`guidance_weight`,
        `guidance_interval`, `resampling_weight`).
    strict_version
        Raise on an unsupported Boltz version instead of warning.
    """
    global _INSTALLED  # noqa: PLW0603

    version = boltz_version()
    if version not in SUPPORTED_BOLTZ_VERSIONS:
        msg = (
            f"boltz_cdr was written against boltz {'/'.join(SUPPORTED_BOLTZ_VERSIONS)}, "
            f"but boltz {version} is installed. `sampler.cdr_selective_sample` is a "
            f"line-for-line copy of that version's AtomDiffusion.sample and may no "
            f"longer be faithful. Re-derive it from the installed source before use."
        )
        if strict_version:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)

    from boltz.model.modules import diffusionv2
    from boltz.model.potentials import potentials as boltz_potentials

    if "sample" not in _ORIGINALS:
        _ORIGINALS["sample"] = diffusionv2.AtomDiffusion.sample
        _ORIGINALS["get_potentials"] = boltz_potentials.get_potentials

    diffusionv2.AtomDiffusion.sample = cdr_selective_sample

    registered = False
    if guidance is not None:
        original_get_potentials = _ORIGINALS["get_potentials"]

        def get_potentials_with_cdr(steering_args, boltz2=False):
            found = list(original_get_potentials(steering_args, boltz2=boltz2))
            if boltz2:
                found.append(make_cdr_potential(guidance, guidance_parameters))
            return found

        boltz_potentials.get_potentials = get_potentials_with_cdr
        # The sampler imports the symbol into its own module namespace at call time via
        # `from ... import get_potentials`, but diffusionv2 bound it at import time, so
        # both references have to be updated.
        diffusionv2.get_potentials = get_potentials_with_cdr
        registered = True

    _INSTALLED = True
    return PatchStatus(True, version, True, registered)


def uninstall() -> None:
    """Restore stock Boltz-2. Used by tests and by the baseline ablation arm."""
    global _INSTALLED, _ACTIVE  # noqa: PLW0603

    if not _ORIGINALS:
        return
    from boltz.model.modules import diffusionv2
    from boltz.model.potentials import potentials as boltz_potentials

    diffusionv2.AtomDiffusion.sample = _ORIGINALS["sample"]
    boltz_potentials.get_potentials = _ORIGINALS["get_potentials"]
    diffusionv2.get_potentials = _ORIGINALS["get_potentials"]
    _INSTALLED = False
    _ACTIVE = None


@contextmanager
def cdr_sampling(config: CDRSamplingConfig | None):
    """Activate a CDR sampling configuration for the duration of a prediction run."""
    global _ACTIVE  # noqa: PLW0603

    previous = _ACTIVE
    _ACTIVE = config
    try:
        yield config
    finally:
        _ACTIVE = previous


def steering_args(
    *,
    guidance: bool = True,
    fk_steering: bool = False,
    num_particles: int = 3,
    physical_guidance: bool = True,
    num_gd_steps: int = 16,
    fk_lambda: float = 4.0,
    fk_resampling_interval: int = 5,
) -> dict:
    """Build the `steering_args` dict Boltz-2's sampler expects.

    `contact_guidance_update` is the switch that makes the sampler call
    `potential.compute_gradient` at all, so it must be on for Arm B3 to do anything.
    """
    return {
        "fk_steering": fk_steering,
        "num_particles": num_particles,
        "fk_lambda": fk_lambda,
        "fk_resampling_interval": fk_resampling_interval,
        "physical_guidance_update": physical_guidance,
        "contact_guidance_update": guidance,
        "num_gd_steps": num_gd_steps,
    }


def cdr_atom_mask_from_feats(feats: dict, guidance: CDRGuidanceConfig) -> torch.Tensor:
    """(n_atom,) bool CDR mask, for building a `CDRSamplingConfig` from real features."""
    from boltz_cdr.masks import atom_mask

    return atom_mask(feats, guidance.antibody_chain, guidance.cdr_residues)
