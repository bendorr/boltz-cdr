"""CDR-selective diffusion sampling — Arm B1 (noise scaling) and B2 (partial diffusion).

`cdr_selective_sample` is a copy of `boltz.model.modules.diffusionv2.AtomDiffusion.sample`
as of **boltz 2.2.1**, with the modifications fenced between `# >>> boltz_cdr` and
`# <<< boltz_cdr` so the diff against upstream is auditable by eye. Everything outside
those fences is upstream code and must stay that way; `patch.py` refuses to install if the
installed Boltz version differs from the one this copy was taken against.

Why copy rather than wrap: the two levers act on single expressions buried in the middle
of the reverse-diffusion loop (the per-step noise draw and the initialization), and there
is no hook for either. The alternatives were monkey-patching `torch.randn` for the
duration of the call — compact but opaque, and dependent on the *order* of random draws —
or forking the whole package. A fenced copy of one method is the most reviewable option.

The two levers:

  B1 · noise scaling      `eps` is multiplied per-atom by `1 + (lambda - 1) * mask_CDR`.
                          Framework and antigen atoms keep exactly the trajectory the
                          unmodified sampler would give them; CDR atoms get lambda times
                          the stochastic churn. This is what "more sampling in the loops"
                          means mechanically, as opposed to "more seeds", which
                          re-randomizes everything equally and mostly re-derives the parts
                          the model already had right.

  B2 · partial diffusion  Start from an existing docked pose instead of pure noise, and
                          re-noise *only* the CDR atoms to an intermediate sigma. The
                          schedule is truncated to begin there. Directly analogous to
                          RFdiffusion's partial noising, restricted to the paratope: it
                          explores loop conformations around a pose that is already good
                          rather than re-solving the docking problem every sample.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CDRSamplingConfig:
    """Configuration for the CDR-selective sampler."""

    cdr_atom_mask: torch.Tensor | None = None  # (n_atom,) bool — fallback, see below
    noise_scale: float = 1.0  # B1 lambda; 1.0 reproduces stock Boltz-2
    partial_diffusion_sigma: float | None = None  # B2 sigma_start; None disables
    reference_coords: torch.Tensor | None = None  # (n_atom, 3) docked pose for B2

    # Preferred way to specify the CDRs: by residue, resolved against Boltz's own features
    # at sampling time. A precomputed `cdr_atom_mask` has to assume that some external
    # structure's atom ordering matches Boltz's featurized ordering, which is an assumption
    # we cannot check and which fails quietly — crystal structures are missing side-chain
    # atoms that Boltz predicts, so the counts genuinely differ. Resolving from `feats`
    # removes the assumption entirely.
    cdr_residues: tuple[int, ...] | None = None  # 0-based positions within the chain
    antibody_chain: int = 0  # chain ordinal, matching CDRGuidanceConfig

    def __post_init__(self) -> None:
        if self.noise_scale <= 0:
            msg = "noise_scale must be positive"
            raise ValueError(msg)
        if (self.partial_diffusion_sigma is None) != (self.reference_coords is None):
            msg = "partial diffusion needs both partial_diffusion_sigma and reference_coords"
            raise ValueError(msg)
        if self.modifies_noise and self.cdr_atom_mask is None and self.cdr_residues is None:
            msg = "noise scaling needs either cdr_residues (preferred) or cdr_atom_mask"
            raise ValueError(msg)

    @property
    def modifies_noise(self) -> bool:
        return self.noise_scale != 1.0

    @property
    def modifies_init(self) -> bool:
        return self.partial_diffusion_sigma is not None

    def fitted_cdr_mask(self, n_atom: int, device, feats: dict | None = None) -> torch.Tensor:
        """(n_atom,) bool CDR mask for the padded atom dimension.

        Resolved from `feats` when `cdr_residues` is set — the reliable path, because it
        uses Boltz's own atom/token mapping. Otherwise falls back to the precomputed
        `cdr_atom_mask`, truncated or zero-padded to `n_atom` so downstream broadcasts stay
        legal. Padded atoms are never marked as CDR, which is correct: they are masked out
        everywhere else too.
        """
        if self.cdr_residues is not None and feats is not None:
            from boltz_cdr.masks import atom_mask

            resolved = atom_mask(feats, self.antibody_chain, self.cdr_residues).to(device)
            fitted = torch.zeros(n_atom, dtype=torch.bool, device=device)
            n = min(len(resolved), n_atom)
            fitted[:n] = resolved[:n]
            return fitted

        fitted = torch.zeros(n_atom, dtype=torch.bool, device=device)
        if self.cdr_atom_mask is not None:
            mask = self.cdr_atom_mask.to(device)
            if len(mask) != n_atom:
                warnings.warn(
                    f"cdr_atom_mask has {len(mask)} entries but the padded atom dimension "
                    f"is {n_atom}. It will be truncated/zero-padded, but a length mismatch "
                    f"means the mask may not line up with Boltz's atom ordering at all. "
                    f"Prefer CDRSamplingConfig(cdr_residues=...), which is resolved from "
                    f"Boltz's own features.",
                    stacklevel=3,
                )
            n = min(len(mask), n_atom)
            fitted[:n] = mask[:n]
        return fitted

    def fitted_reference(self, n_atom: int, device, dtype) -> torch.Tensor:
        """(n_atom, 3) reference pose, zero-padded or truncated to the padded atom count.

        Unlike the mask, this cannot be resolved from `feats` — it is coordinate data, and
        it only makes sense if its atom ordering already matches Boltz's. That holds when
        the reference is a Boltz prediction read straight back from its output mmCIF, which
        is how `03_arm_b_diffusion_resample.py` obtains it.
        """
        fitted = torch.zeros(n_atom, 3, device=device, dtype=dtype)
        if self.reference_coords is not None:
            reference = self.reference_coords.to(device, dtype)
            if len(reference) > n_atom:
                warnings.warn(
                    f"reference_coords has {len(reference)} atoms but the padded atom "
                    f"dimension is only {n_atom}; it will be truncated. The reference must "
                    f"come from a Boltz prediction of this exact input for partial "
                    f"diffusion to be meaningful.",
                    stacklevel=3,
                )
            n = min(len(reference), n_atom)
            fitted[:n] = reference[:n]
        return fitted

    def atom_noise_scale(self, n_atom: int, device, dtype, feats: dict | None = None) -> torch.Tensor:
        """(1, n_atom, 1) per-atom multiplier for the injected noise."""
        scale = torch.ones(n_atom, device=device, dtype=dtype)
        if self.modifies_noise:
            mask = self.fitted_cdr_mask(n_atom, device, feats)
            scale = torch.where(mask, torch.full_like(scale, self.noise_scale), scale)
        return scale.view(1, -1, 1)


def cdr_selective_sample(
    self,
    atom_mask,
    num_sampling_steps=None,
    multiplicity=1,
    max_parallel_samples=None,
    steering_args=None,
    **network_condition_kwargs,
):
    """Drop-in replacement for `AtomDiffusion.sample` with CDR-selective sampling."""
    from boltz.model.loss.diffusionv2 import weighted_rigid_align
    from boltz.model.modules.diffusionv2 import compute_random_augmentation
    from boltz.model.potentials.potentials import get_potentials

    from boltz_cdr.patch import active_sampling_config

    def default(v, d):
        return v if v is not None else d

    if steering_args is not None and (
        steering_args["fk_steering"]
        or steering_args["physical_guidance_update"]
        or steering_args["contact_guidance_update"]
    ):
        potentials = get_potentials(steering_args, boltz2=True)

    if steering_args["fk_steering"]:
        multiplicity = multiplicity * steering_args["num_particles"]
        energy_traj = torch.empty((multiplicity, 0), device=self.device)
        resample_weights = torch.ones(multiplicity, device=self.device).reshape(
            -1, steering_args["num_particles"]
        )
    if (
        steering_args["physical_guidance_update"]
        or steering_args["contact_guidance_update"]
    ):
        scaled_guidance_update = torch.zeros(
            (multiplicity, *atom_mask.shape[1:], 3),
            dtype=torch.float32,
            device=self.device,
        )
    if max_parallel_samples is None:
        max_parallel_samples = multiplicity

    num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
    atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

    shape = (*atom_mask.shape, 3)

    # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
    sigmas = self.sample_schedule(num_sampling_steps)
    gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
    sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))
    if self.training and self.step_scale_random is not None:
        step_scale = np.random.choice(self.step_scale_random)
    else:
        step_scale = self.step_scale

    # atom position is noise at the beginning
    init_sigma = sigmas[0]
    atom_coords = init_sigma * torch.randn(shape, device=self.device)

    # >>> boltz_cdr — Arm B1/B2 setup
    cfg = active_sampling_config()
    cdr_feats = network_condition_kwargs.get("feats")
    noise_scale = (
        cfg.atom_noise_scale(shape[1], self.device, atom_coords.dtype, cdr_feats)
        if cfg is not None and cfg.modifies_noise
        else None
    )
    if cfg is not None and cfg.modifies_init:
        # B2: begin from the supplied docked pose with only the CDR atoms re-noised, and
        # truncate the schedule so denoising starts at the matching noise level.
        sigma_start = float(cfg.partial_diffusion_sigma)
        keep = [i for i, s in enumerate(sigmas.tolist()) if s <= sigma_start]
        if keep:
            start = keep[0]
            sigmas_and_gammas = list(zip(sigmas[start:-1], sigmas[start + 1 :], gammas[start + 1 :]))
            init_sigma = sigmas[start]
        # Fit both to the padded atom count before broadcasting — Boltz's atom dimension
        # is padded, while the mask and reference come from an unpadded structure.
        reference = cfg.fitted_reference(shape[1], self.device, atom_coords.dtype)
        reference = reference.unsqueeze(0).expand(shape[0], -1, -1)
        cdr = cfg.fitted_cdr_mask(shape[1], self.device, cdr_feats).view(1, -1, 1)
        atom_coords = torch.where(
            cdr, reference + init_sigma * torch.randn(shape, device=self.device), reference
        )
    # <<< boltz_cdr

    token_repr = None
    atom_coords_denoised = None

    # gradually denoise
    for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
        random_R, random_tr = compute_random_augmentation(
            multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
        )
        atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
        atom_coords = torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
        if atom_coords_denoised is not None:
            atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)
            atom_coords_denoised = (
                torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R) + random_tr
            )
        if (
            steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ) and scaled_guidance_update is not None:
            scaled_guidance_update = torch.einsum(
                "bmd,bds->bms", scaled_guidance_update, random_R
            )

        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

        t_hat = sigma_tm * (1 + gamma)
        steering_t = 1.0 - (step_idx / num_sampling_steps)
        noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
        eps = sqrt(noise_var) * torch.randn(shape, device=self.device)

        # >>> boltz_cdr — Arm B1: concentrate the stochasticity on the CDR atoms
        if noise_scale is not None:
            eps = eps * noise_scale
        # <<< boltz_cdr

        atom_coords_noisy = atom_coords + eps

        with torch.no_grad():
            atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
            sample_ids = torch.arange(multiplicity, device=atom_coords_noisy.device)
            sample_ids_chunks = sample_ids.chunk(multiplicity % max_parallel_samples + 1)

            for sample_ids_chunk in sample_ids_chunks:
                atom_coords_denoised_chunk = self.preconditioned_network_forward(
                    atom_coords_noisy[sample_ids_chunk],
                    t_hat,
                    network_condition_kwargs=dict(
                        multiplicity=sample_ids_chunk.numel(),
                        **network_condition_kwargs,
                    ),
                )
                atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk

            if steering_args["fk_steering"] and (
                (
                    step_idx % steering_args["fk_resampling_interval"] == 0
                    and noise_var > 0
                )
                or step_idx == num_sampling_steps - 1
            ):
                # Compute energy of x_0 prediction
                energy = torch.zeros(multiplicity, device=self.device)
                for potential in potentials:
                    parameters = potential.compute_parameters(steering_t)
                    if parameters["resampling_weight"] > 0:
                        component_energy = potential.compute(
                            atom_coords_denoised,
                            network_condition_kwargs["feats"],
                            parameters,
                        )
                        energy += parameters["resampling_weight"] * component_energy
                energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                # Compute log G values
                if step_idx == 0:
                    log_G = -1 * energy
                else:
                    log_G = energy_traj[:, -2] - energy_traj[:, -1]

                # Compute ll difference between guided and unguided transition distribution
                if (
                    steering_args["physical_guidance_update"]
                    or steering_args["contact_guidance_update"]
                ) and noise_var > 0:
                    ll_difference = (
                        eps**2 - (eps + scaled_guidance_update) ** 2
                    ).sum(dim=(-1, -2)) / (2 * noise_var)
                else:
                    ll_difference = torch.zeros_like(energy)

                # Compute resampling weights
                resample_weights = F.softmax(
                    (ll_difference + steering_args["fk_lambda"] * log_G).reshape(
                        -1, steering_args["num_particles"]
                    ),
                    dim=1,
                )

            # Compute guidance update to x_0 prediction
            if (
                steering_args["physical_guidance_update"]
                or steering_args["contact_guidance_update"]
            ) and step_idx < num_sampling_steps - 1:
                guidance_update = torch.zeros_like(atom_coords_denoised)
                for guidance_step in range(steering_args["num_gd_steps"]):
                    energy_gradient = torch.zeros_like(atom_coords_denoised)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        if (
                            parameters["guidance_weight"] > 0
                            and (guidance_step) % parameters["guidance_interval"] == 0
                        ):
                            energy_gradient += parameters[
                                "guidance_weight"
                            ] * potential.compute_gradient(
                                atom_coords_denoised + guidance_update,
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                    guidance_update -= energy_gradient
                atom_coords_denoised += guidance_update
                scaled_guidance_update = (
                    guidance_update * -1 * self.step_scale * (sigma_t - t_hat) / t_hat
                )

            if steering_args["fk_steering"] and (
                (
                    step_idx % steering_args["fk_resampling_interval"] == 0
                    and noise_var > 0
                )
                or step_idx == num_sampling_steps - 1
            ):
                resample_indices = (
                    torch.multinomial(
                        resample_weights,
                        resample_weights.shape[1]
                        if step_idx < num_sampling_steps - 1
                        else 1,
                        replacement=True,
                    )
                    + resample_weights.shape[1]
                    * torch.arange(
                        resample_weights.shape[0], device=resample_weights.device
                    ).unsqueeze(-1)
                ).flatten()

                atom_coords = atom_coords[resample_indices]
                atom_coords_noisy = atom_coords_noisy[resample_indices]
                atom_mask = atom_mask[resample_indices]
                if atom_coords_denoised is not None:
                    atom_coords_denoised = atom_coords_denoised[resample_indices]
                energy_traj = energy_traj[resample_indices]
                if (
                    steering_args["physical_guidance_update"]
                    or steering_args["contact_guidance_update"]
                ):
                    scaled_guidance_update = scaled_guidance_update[resample_indices]
                if token_repr is not None:
                    token_repr = token_repr[resample_indices]

        if self.alignment_reverse_diff:
            with torch.autocast("cuda", enabled=False):
                atom_coords_noisy = weighted_rigid_align(
                    atom_coords_noisy.float(),
                    atom_coords_denoised.float(),
                    atom_mask.float(),
                    atom_mask.float(),
                )

            atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
        atom_coords_next = (
            atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
        )

        atom_coords = atom_coords_next

    return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)
