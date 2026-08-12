#!/usr/bin/env python
"""Arm B — CDR-selective diffusion resampling. Needs a GPU.

Three sub-arms, each isolating one lever so the ablation is interpretable:

  armB_noise    B1 alone. Per-atom noise scaling: CDR atoms get `--lambda-cdr` times the
                stochastic churn, everything else follows the stock trajectory.
  armB_partial  B2 alone. Start from a Stage-0 docked pose, re-noise only the CDR atoms to
                `--partial-sigma`, denoise from there.
  armB_guided   B1 + B3. Noise scaling plus the differentiable CDR-epitope potential, so
                the extra exploration is pulled toward productive interface geometry
                instead of wandering. This is the sub-arm that runs the backward pass
                inside the sampler.

Unlike Arm A, this arm needs the monkey-patch live in the same process as the model, so
predictions run in-process and `run_prediction` asserts the patch state before starting.

Usage:
    python scripts/03_arm_b_diffusion_resample.py --target 8QF4 --samples 8
    python scripts/03_arm_b_diffusion_resample.py --target 8QF4 --sub-arm armB_guided \
        --lambda-cdr 1.5 --guidance-weight 0.2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from _common import (
    DEFAULT_RESULTS,
    add_cdr_residues_argument,
    arm_dir,
    isolated_arm,
    load_targets,
    main_guard,
    read_json,
    require_boltz,
    warn_if_empty,
    write_json,
)
from boltz_cdr.metrics.contacts import epitope_residues
from boltz_cdr.pdb_io import (
    load_complex,
    load_predicted_complex,
    with_canonical_chain_ids,
    write_complex_cif,
)
from boltz_cdr.potentials import CDRGuidanceConfig
from boltz_cdr.sampler import CDRSamplingConfig
from boltz_cdr.yaml_io import (
    ANTIBODY_CHAIN_ID,
    ANTIBODY_ORDINAL,
    ANTIGEN_CHAIN_ID,
    ANTIGEN_ORDINAL,
    write_boltz_yaml,
)

SUB_ARMS = ("armB_noise", "armB_partial", "armB_guided")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--sub-arm", action="append", dest="sub_arms", choices=SUB_ARMS)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--lambda-cdr", type=float, default=1.5,
                        help="B1: multiplier on injected noise for CDR atoms")
    parser.add_argument("--partial-sigma", type=float, default=8.0,
                        help="B2: sigma to re-noise CDR atoms to, starting from a docked pose")
    parser.add_argument("--guidance-weight", type=float, default=0.2,
                        help="B3: weight on the CDR-epitope potential's gradient")
    parser.add_argument("--n-contacts", type=int, default=8)
    parser.add_argument("--contact-distance", type=float, default=4.5)
    parser.add_argument("--epitope-from-native", action="store_true",
                        help="restrict guidance to the crystallographic epitope. This is "
                             "an ORACLE setting — it uses the answer. Use it only to test "
                             "the epitope-directed capability, never as a benchmark arm.")
    parser.add_argument("--no-msa-server", action="store_true")
    parser.add_argument("--use-potentials", action="store_true",
                        help="additionally enable Boltz's own physical potentials and "
                             "Feynman-Kac particle resampling. NOT required for Arm B3: "
                             "contact_guidance_update is on by default and is what makes "
                             "the sampler call our potential. This flag triples runtime "
                             "because fk_steering multiplies multiplicity by num_particles.")
    add_cdr_residues_argument(parser)
    args = parser.parse_args()

    require_boltz()
    from boltz_cdr import patch
    from boltz_cdr.run import collect_predictions, describe_environment, run_prediction

    sub_arms = args.sub_arms or list(SUB_ARMS)
    targets = load_targets(
        only=None if args.all else args.targets, cdr_residues=args.cdr_residues
    )
    if not targets:
        raise SystemExit("no targets selected; pass --target ID or --all")

    print(describe_environment())

    for target in targets:
        stage0_path = arm_dir(args.results, target.id, "baseline") / "manifest.json"
        stage0 = read_json(stage0_path) if stage0_path.exists() else None

        for sub_arm in sub_arms:
            if sub_arm == "armB_partial" and stage0 is None:
                print(f"  skipping {sub_arm}: needs Stage 0 — run 01_global_dock.py first")
                continue

            out = arm_dir(args.results, target.id, sub_arm)
            print(f"\n=== {target.id} / {sub_arm} -> {out}")

            with isolated_arm(out, f"{target.id} / {sub_arm}"):

                # CDR spans for the sequence we are about to predict: the user's definition
                # if they gave one, the automatic IMGT annotation otherwise.
                annotation = target.annotation_for(target.antibody_sequence)
                epitope = (
                    tuple(int(i) for i in epitope_residues(target.native))
                    if args.epitope_from_native
                    else None
                )
                guidance = CDRGuidanceConfig(
                    antibody_chain=ANTIBODY_ORDINAL,
                    antigen_chain=ANTIGEN_ORDINAL,
                    cdr_residues=tuple(int(i) for i in annotation.all_indices),
                    epitope_residues=epitope,
                    n_contacts=args.n_contacts,
                    contact_distance=args.contact_distance,
                )

                use_guidance = sub_arm == "armB_guided"
                patch.uninstall()
                status = patch.install(
                    guidance=guidance if use_guidance else None,
                    guidance_parameters={"guidance_weight": args.guidance_weight},
                )
                print(f"    {status}")

                spec = write_boltz_yaml(
                    out / "input.yaml", target.antibody_sequence, target.antigen_sequence
                )

                sampling = build_sampling_config(sub_arm, args, target, annotation, stage0)

                with patch.cdr_sampling(sampling):
                    run_prediction(
                        spec.path,
                        out / "boltz",
                        diffusion_samples=args.samples,
                        sampling_steps=args.sampling_steps,
                        seed=args.seed,
                        use_msa_server=not args.no_msa_server,
                        use_potentials=args.use_potentials,
                        expect_patched=True,
                    )

                records = []
                for prediction in collect_predictions(out / "boltz"):
                    structure = load_predicted_complex(
                        prediction.structure_path, target.native,
                        name=f"{target.id}_{sub_arm}_{prediction.model_index}",
                    )
                    canonical = out / "structures" / f"model_{prediction.model_index}.cif"
                    write_complex_cif(
                        with_canonical_chain_ids(structure, ANTIBODY_CHAIN_ID, ANTIGEN_CHAIN_ID),
                        canonical,
                    )
                    records.append(
                        {
                            "model_index": prediction.model_index,
                            "structure": str(canonical),
                            "confidence": prediction.confidence(),
                        }
                    )

                write_json(
                    out / "manifest.json",
                    {
                        "target": target.id,
                        "arm": sub_arm,
                        "settings": {
                            "diffusion_samples": args.samples,
                            "seed": args.seed,
                            "lambda_cdr": args.lambda_cdr if sub_arm != "armB_partial" else 1.0,
                            "partial_sigma": args.partial_sigma if sub_arm == "armB_partial" else None,
                            "guidance_weight": args.guidance_weight if use_guidance else None,
                            "n_contacts": args.n_contacts,
                            "epitope_directed": bool(epitope),
                            "cdr_source": target.cdr_source,
                            "cdr_spec": target.cdr_spec.as_dict() if target.cdr_spec else None,
                            "cdr_residues": [int(i) for i in annotation.all_indices],
                        },
                        "environment": describe_environment(),
                        "models": records,
                    },
                )
                print(f"    {len(records)} structures")
                warn_if_empty(records, f"{target.id} / {sub_arm}")

            patch.uninstall()

    print("\nArm B complete.")
    return 0


def build_sampling_config(sub_arm, args, target, annotation, stage0) -> CDRSamplingConfig:
    """Assemble the sampler configuration for one sub-arm.

    The CDR set is passed as *residue positions*, not as a precomputed atom mask. The
    sampler resolves them against Boltz's own features at sampling time, so we never have
    to assume that some external structure's atom ordering matches Boltz's — an assumption
    that is in fact false (crystal structures are missing side-chain atoms Boltz predicts)
    and that fails silently rather than loudly.
    """
    cdr_residues = tuple(int(i) for i in annotation.all_indices)

    if sub_arm == "armB_partial":
        best = max(
            stage0["models"],
            key=lambda r: float((r.get("confidence") or {}).get("iptm", 0.0)),
        )
        # The reference must be in Boltz's atom order, so it is read straight back from a
        # Boltz prediction of this same input rather than from the crystal structure.
        docked = load_complex(
            best["structure"], ANTIBODY_CHAIN_ID, ANTIGEN_CHAIN_ID, name="docked"
        )
        reference = torch.as_tensor(
            np.vstack([docked.antibody.coords, docked.antigen.coords]), dtype=torch.float32
        )
        return CDRSamplingConfig(
            cdr_residues=cdr_residues,
            antibody_chain=ANTIBODY_ORDINAL,
            noise_scale=1.0,
            partial_diffusion_sigma=args.partial_sigma,
            reference_coords=reference,
        )

    return CDRSamplingConfig(
        cdr_residues=cdr_residues,
        antibody_chain=ANTIBODY_ORDINAL,
        noise_scale=args.lambda_cdr,
    )


if __name__ == "__main__":
    raise SystemExit(main_guard(main))
