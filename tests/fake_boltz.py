"""A minimal stand-in for the parts of Boltz-2 that `boltz_cdr` patches.

`patch.py`, `sampler.py`, and `run.py` are the three modules that touch Boltz, and they
are precisely the ones a laptop cannot exercise — which is how integration bugs survive
until the first expensive GPU run. This module supplies just enough of the Boltz surface
for them to execute for real:

  boltz.model.modules.diffusionv2   AtomDiffusion, compute_random_augmentation, get_potentials
  boltz.model.loss.diffusionv2      weighted_rigid_align
  boltz.model.potentials.potentials Potential (ABC), get_potentials
  boltz.main                        predict

The fake `AtomDiffusion` carries the same attributes the real one does and a denoiser that
pulls coordinates toward a fixed target, so the reverse-diffusion loop converges and the
returned coordinates are checkable. Nothing here validates Boltz's own numerics — it
validates *our* wiring: that the patch binds, that the CDR code paths run, that shapes
survive padding, and that the guidance hook is actually reached.
"""

from __future__ import annotations

import sys
import types
from abc import ABC, abstractmethod

import torch

BOLTZ_MODULES = (
    "boltz",
    "boltz.main",
    "boltz.model",
    "boltz.model.loss",
    "boltz.model.loss.diffusionv2",
    "boltz.model.modules",
    "boltz.model.modules.diffusionv2",
    "boltz.model.potentials",
    "boltz.model.potentials.potentials",
)


class FakePotential(ABC):
    """Mirror of `boltz.model.potentials.potentials.Potential`."""

    def __init__(self, parameters=None):
        self.parameters = parameters

    def compute(self, coords, feats, parameters):
        raise NotImplementedError

    def compute_gradient(self, coords, feats, parameters):
        raise NotImplementedError

    def compute_parameters(self, t):
        # The real implementation resolves ParameterSchedule objects; ours passes floats
        # straight through, which is all our potential uses.
        return dict(self.parameters or {})

    @abstractmethod
    def compute_function(self, *args, **kwargs): ...

    @abstractmethod
    def compute_variable(self, *args, **kwargs): ...

    @abstractmethod
    def compute_args(self, *args, **kwargs): ...


class FakeAtomDiffusion:
    """Stands in for `AtomDiffusion`, with the attributes `sample()` reads."""

    def __init__(self, n_atom: int, *, num_sampling_steps: int = 6, target=None):
        self.device = torch.device("cpu")
        self.num_sampling_steps = num_sampling_steps
        self.sigma_min, self.sigma_max, self.sigma_data = 0.0004, 160.0, 16.0
        self.rho = 7
        self.gamma_0, self.gamma_min = 0.8, 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5
        self.step_scale_random = None
        self.training = False
        self.alignment_reverse_diff = True
        self.n_atom = n_atom
        self.target = torch.zeros(n_atom, 3) if target is None else target
        self.forward_calls: list[float] = []

    def sample_schedule(self, num_sampling_steps=None):
        """The real Karras schedule, copied so truncation is tested against real values."""
        n = num_sampling_steps or self.num_sampling_steps
        inv_rho = 1 / self.rho
        steps = torch.arange(n, dtype=torch.float32)
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (n - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    def preconditioned_network_forward(self, noised, sigma, network_condition_kwargs):
        """A contraction toward `self.target` — enough for the loop to converge."""
        self.forward_calls.append(float(sigma) if not torch.is_tensor(sigma) else float(sigma.mean()))
        target = self.target.to(noised.device, noised.dtype)
        return target.unsqueeze(0).expand_as(noised) * 0.5 + noised * 0.5

    # Replaced by patch.install(); present so uninstall() has something to restore.
    def sample(self, *args, **kwargs):
        return {"sample_atom_coords": None, "diff_token_repr": None, "stock": True}


def compute_random_augmentation(multiplicity, device=None, dtype=None):
    """(R, t) with R a proper rotation, matching the real function's shapes."""
    q, r = torch.linalg.qr(torch.randn(multiplicity, 3, 3, device=device, dtype=dtype))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    flip = torch.where(torch.linalg.det(q) < 0, -1.0, 1.0)
    q[:, :, 0] = q[:, :, 0] * flip.unsqueeze(-1)
    return q, torch.randn(multiplicity, 1, 3, device=device, dtype=dtype)


def weighted_rigid_align(mobile, target, mask_a, mask_b):
    """Identity: alignment is Boltz's numerics, not our wiring."""
    return mobile


def fake_get_potentials(steering_args, boltz2=False):
    """Stock Boltz registers several potentials here; none of them are ours."""
    return []


def install(version: str = "2.2.1"):
    """Insert the fake `boltz` package into `sys.modules`. Returns a teardown callable."""
    import importlib.metadata

    saved = {name: sys.modules.get(name) for name in BOLTZ_MODULES}
    real_version = importlib.metadata.version

    for name in BOLTZ_MODULES:
        module = types.ModuleType(name)
        module.__path__ = []  # mark as a package so submodule imports resolve
        sys.modules[name] = module

    diff = sys.modules["boltz.model.modules.diffusionv2"]
    diff.AtomDiffusion = FakeAtomDiffusion
    diff.compute_random_augmentation = compute_random_augmentation
    # Mirrors the real module, which does `from ...potentials import get_potentials`
    # at import time — the binding our patch must also rebind.
    diff.get_potentials = fake_get_potentials

    sys.modules["boltz.model.loss.diffusionv2"].weighted_rigid_align = weighted_rigid_align

    pot = sys.modules["boltz.model.potentials.potentials"]
    pot.Potential = FakePotential
    pot.get_potentials = fake_get_potentials

    predict = types.SimpleNamespace(main=lambda args, standalone_mode=True: ("called", args))
    sys.modules["boltz.main"].predict = predict

    def _version(name, *a, **k):
        return version if name == "boltz" else real_version(name, *a, **k)

    importlib.metadata.version = _version

    def teardown():
        importlib.metadata.version = real_version
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    return teardown
