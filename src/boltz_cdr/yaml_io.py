"""Boltz-2 input YAML construction.

One choice is encoded here and is stated explicitly rather than left implicit.

**The nanobody is run single-sequence (`msa: empty`); the antigen gets a real MSA.**

An antibody's MSA is dominated by germline framework. Those homologues agree almost
perfectly on the beta-sandwich and disagree meaninglessly on the CDRs, so the profile
actively dilutes the hypervariable loop signal that determines specificity — the model is
shown a column of "anything goes" exactly where it most needs a commitment. The antigen is
an ordinary globular protein with real homologues and real coevolutionary signal, so it
benefits from an MSA in the usual way. Boltz-2 supports per-chain MSA specification, so we
take the useful half and drop the harmful half.

This is a hypothesis rather than an established result; `single_sequence_antibody=False`
restores the paired-MSA behavior for anyone wanting to test it as an ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from boltz_cdr.templates import TemplateSpec

ANTIBODY_CHAIN_ID = "A"
ANTIGEN_CHAIN_ID = "B"

# Chain ordinals, matching the order chains are written below. These are what
# `CDRGuidanceConfig.antibody_chain` / `.antigen_chain` refer to.
ANTIBODY_ORDINAL = 0
ANTIGEN_ORDINAL = 1


@dataclass
class BoltzInput:
    """A written Boltz-2 YAML plus the metadata needed to interpret its outputs."""

    path: Path
    antibody_sequence: str
    antigen_sequence: str
    templates: list[TemplateSpec] = field(default_factory=list)
    single_sequence_antibody: bool = True

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "antibody_len": len(self.antibody_sequence),
            "antigen_len": len(self.antigen_sequence),
            "templates": [t.as_dict() for t in self.templates],
            "single_sequence_antibody": self.single_sequence_antibody,
        }


def write_boltz_yaml(
    path: str | Path,
    antibody_sequence: str,
    antigen_sequence: str,
    *,
    templates: list[TemplateSpec] | None = None,
    template_force: bool = True,
    template_threshold: float = 5.0,
    single_sequence_antibody: bool = True,
    antibody_msa: str | None = None,
    antigen_msa: str | None = None,
) -> BoltzInput:
    """Write a Boltz-2 prediction input.

    Parameters
    ----------
    templates
        Structural templates. For Arm A this is the CDR-masked docked complex.
    template_force
        Enables Boltz's `TemplateReferencePotential`, which actively restrains the
        prediction toward the template during diffusion rather than merely conditioning on
        it. For Arm A this is what pins the docked frame; without it the model is free to
        drift away from the pose we are trying to hold fixed.
    template_threshold
        Angstrom tolerance for that restraint.
    single_sequence_antibody
        See the module docstring.
    """
    templates = templates or []

    antibody_entry: dict = {"id": ANTIBODY_CHAIN_ID, "sequence": antibody_sequence}
    if antibody_msa is not None:
        antibody_entry["msa"] = antibody_msa
    elif single_sequence_antibody:
        antibody_entry["msa"] = "empty"

    antigen_entry: dict = {"id": ANTIGEN_CHAIN_ID, "sequence": antigen_sequence}
    if antigen_msa is not None:
        antigen_entry["msa"] = antigen_msa
    # Otherwise leave `msa` unset so `--use_msa_server` fills it in.

    doc: dict = {
        "version": 1,
        "sequences": [{"protein": antibody_entry}, {"protein": antigen_entry}],
    }

    if templates:
        doc["templates"] = [
            {
                "cif": str(Path(t.path).resolve()),
                "chain_id": [ANTIBODY_CHAIN_ID, ANTIGEN_CHAIN_ID],
                "template_id": [t.antibody_chain_id, t.antigen_chain_id],
                **({"force": True, "threshold": template_threshold} if template_force else {}),
            }
            for t in templates
        ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)

    return BoltzInput(
        path=path,
        antibody_sequence=antibody_sequence,
        antigen_sequence=antigen_sequence,
        templates=templates,
        single_sequence_antibody=single_sequence_antibody,
    )


def boltz_predict_command(
    yaml_path: str | Path,
    out_dir: str | Path,
    *,
    diffusion_samples: int = 5,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    seed: int | None = None,
    use_msa_server: bool = True,
    use_potentials: bool = False,
    devices: int = 1,
) -> list[str]:
    """The `boltz predict` command line for a given input.

    Returned as an argv list so callers can either `subprocess.run` it or print it into a
    notebook cell.

    Note on `--use_potentials`: it does NOT gate our CDR potential. Boltz's
    `BoltzSteeringParams.contact_guidance_update` defaults to True, and that is the flag
    the sampler tests before calling `potential.compute_gradient`. `--use_potentials`
    additionally enables `fk_steering` and `physical_guidance_update` — Boltz's own
    physical potentials plus Feynman-Kac particle resampling, which multiplies the
    effective sample count by `num_particles` (3) and so roughly triples runtime.
    """
    cmd = [
        "boltz", "predict", str(yaml_path),
        "--out_dir", str(out_dir),
        "--diffusion_samples", str(diffusion_samples),
        "--recycling_steps", str(recycling_steps),
        "--sampling_steps", str(sampling_steps),
        "--output_format", "mmcif",
        "--devices", str(devices),
        "--override",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if use_msa_server:
        cmd += ["--use_msa_server"]
    if use_potentials:
        cmd += ["--use_potentials"]
    return cmd


def read_confidence(json_path: str | Path) -> dict:
    """Load one Boltz-2 `confidence_*.json` sidecar."""
    import json

    with Path(json_path).open() as fh:
        return json.load(fh)
