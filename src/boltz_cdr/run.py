"""Driving Boltz-2 and collecting its output.

Arm B requires the runtime patch to be live in the same Python process as the model.
Shelling out to the `boltz` CLI would start a fresh interpreter and drop the patch,
producing a run labeled Arm B that is in fact the baseline, with nothing in the output to
indicate the substitution. Predictions are therefore invoked in-process through the click
command object, and `run_prediction` verifies the patch state before starting.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

MODEL_CIF = re.compile(r"_model_(\d+)\.(cif|pdb)$")


@dataclass
class PredictionOutput:
    """One structure produced by Boltz-2, with its confidence sidecar."""

    structure_path: Path
    confidence_path: Path | None
    model_index: int
    record_id: str

    def confidence(self) -> dict:
        if self.confidence_path is None or not self.confidence_path.exists():
            return {}
        with self.confidence_path.open() as fh:
            return json.load(fh)


def run_prediction(
    yaml_path: str | Path,
    out_dir: str | Path,
    *,
    diffusion_samples: int = 5,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    seed: int | None = None,
    use_msa_server: bool = True,
    use_potentials: bool = False,
    max_parallel_samples: int | None = None,
    devices: int = 1,
    expect_patched: bool | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """Run `boltz predict` in-process. Returns the output directory.

    Parameters
    ----------
    expect_patched
        Assert the patch state before running. Pass `True` for Arm B and `False` for the
        baseline arm; a mismatch raises rather than quietly producing the wrong ablation.
    use_potentials
        Enables Boltz's own physical potentials and Feynman-Kac particle resampling. It is
        NOT required for Arm B3: `contact_guidance_update` defaults to True in
        `BoltzSteeringParams`, and that is what gates the guidance branch that calls
        `potential.compute_gradient`. Note that `fk_steering` multiplies multiplicity by
        `num_particles` (default 3), so this flag roughly triples runtime.
    """
    from boltz.main import predict

    from boltz_cdr import patch

    if expect_patched is not None:
        is_patched = patch._INSTALLED
        if is_patched != expect_patched:
            msg = (
                f"patch state mismatch: expected installed={expect_patched}, "
                f"found installed={is_patched}. Refusing to run a mislabeled arm."
            )
            raise RuntimeError(msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [
        str(yaml_path),
        "--out_dir", str(out_dir),
        "--diffusion_samples", str(diffusion_samples),
        "--recycling_steps", str(recycling_steps),
        "--sampling_steps", str(sampling_steps),
        "--output_format", "mmcif",
        "--devices", str(devices),
        "--override",
    ]
    if seed is not None:
        args += ["--seed", str(seed)]
    if use_msa_server:
        args += ["--use_msa_server"]
    if use_potentials:
        args += ["--use_potentials"]
    if max_parallel_samples is not None:
        args += ["--max_parallel_samples", str(max_parallel_samples)]
    args += extra_args or []

    predict.main(args=args, standalone_mode=False)
    return out_dir


def collect_predictions(out_dir: str | Path) -> list[PredictionOutput]:
    """Gather every predicted structure under a Boltz output directory."""
    out_dir = Path(out_dir)
    found: list[PredictionOutput] = []
    for path in sorted(out_dir.rglob("*_model_*.cif")) + sorted(out_dir.rglob("*_model_*.pdb")):
        match = MODEL_CIF.search(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        record_id = path.name[: match.start()]
        confidence = path.parent / f"confidence_{record_id}_model_{index}.json"
        found.append(
            PredictionOutput(
                structure_path=path,
                confidence_path=confidence if confidence.exists() else None,
                model_index=index,
                record_id=record_id,
            )
        )
    return found


def describe_environment() -> dict:
    """Versions and device info, recorded alongside every run for reproducibility."""
    import platform

    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        info["torch"] = None
    try:
        from importlib.metadata import version

        info["boltz"] = version("boltz")
    except Exception:
        info["boltz"] = None
    return info
