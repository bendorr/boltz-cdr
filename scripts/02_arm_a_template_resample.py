#!/usr/bin/env python
"""Arm A — template-masked CDR re-prediction. Needs a GPU.

Takes the top-ranked Stage-0 poses, writes each back out as a template with the CDR
residues deleted, and re-predicts. The template pins the framework-antigen relative pose;
the deleted loops are left with no geometric constraint at all, so the sampler has to
rebuild them from scratch inside a frame that is already docked.

Also runs the `armA_control` arm, which templates the *complete* complex with nothing
deleted. If the control reproduces its input while the masked arm spreads out, the
diversity is attributable to the masking rather than to template conditioning in general.
Without that control the headline number means very little.

Usage:
    python scripts/02_arm_a_template_resample.py --target 8QF4 --top-k 2 --samples 8
"""

from __future__ import annotations

import argparse

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
from boltz_cdr.pdb_io import (
    load_complex,
    load_predicted_complex,
    with_canonical_chain_ids,
    write_complex_cif,
)
from boltz_cdr.templates import build_cdr_masked_template, build_full_template
from boltz_cdr.yaml_io import ANTIBODY_CHAIN_ID, ANTIGEN_CHAIN_ID, write_boltz_yaml


def rank_stage0(manifest: dict) -> list[dict]:
    """Order Stage-0 models by Boltz's own interface confidence."""

    def key(record: dict) -> float:
        conf = record.get("confidence") or {}
        for field in ("iptm", "confidence_score", "complex_iplddt", "ptm"):
            if isinstance(conf.get(field), (int, float)):
                return float(conf[field])
        return 0.0

    return sorted(manifest["models"], key=key, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--top-k", type=int, default=2,
                        help="how many Stage-0 poses to build templates from")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--mask-cdrs", default="cdr1,cdr2,cdr3",
                        help="comma-separated; 'cdr3' alone is a useful ablation")
    parser.add_argument("--flank", type=int, default=0,
                        help="extra residues masked either side of each loop")
    parser.add_argument("--template-threshold", type=float, default=5.0)
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--no-msa-server", action="store_true")
    parser.add_argument("--use-potentials", action="store_true",
                        help="additionally enable Boltz's physical potentials and FK "
                             "resampling; not required for template forcing, and ~3x slower")
    add_cdr_residues_argument(parser)
    args = parser.parse_args()

    require_boltz()
    from boltz_cdr import patch
    from boltz_cdr.run import collect_predictions, describe_environment, run_prediction

    patch.uninstall()  # Arm A is pure input-space; the sampler stays stock.

    mask_cdrs = tuple(c.strip() for c in args.mask_cdrs.split(",") if c.strip())
    targets = load_targets(
        only=None if args.all else args.targets, cdr_residues=args.cdr_residues
    )
    if not targets:
        raise SystemExit("no targets selected; pass --target ID or --all")

    for target in targets:
        stage0 = arm_dir(args.results, target.id, "baseline") / "manifest.json"
        if not stage0.exists():
            raise SystemExit(f"{stage0} missing — run 01_global_dock.py first")
        ranked = rank_stage0(read_json(stage0))[: args.top_k]
        print(f"\n=== {target.id}: building templates from {len(ranked)} Stage-0 poses")

        builders = [("armA", True)] + ([] if args.no_control else [("armA_control", False)])
        for arm, mask in builders:
            out = arm_dir(args.results, target.id, arm)
            records = []

            with isolated_arm(out, f"{target.id} / {arm}"):
                for rank, source in enumerate(ranked):
                    docked = load_complex(
                        source["structure"], ANTIBODY_CHAIN_ID, ANTIGEN_CHAIN_ID,
                        name=f"{target.id}_pose{rank}",
                    )
                    # A user CDR definition is authoritative and is carried across to the
                    # predicted chain; without one the annotator re-runs on the prediction's
                    # own sequence, which may differ in length from the crystal construct.
                    annotation = target.annotation_for(docked.antibody.seq)

                    template = (
                        build_cdr_masked_template(
                            docked, annotation, out / "templates" / f"pose{rank}.cif",
                            mask_cdrs=mask_cdrs, flank=args.flank,
                        )
                        if mask
                        else build_full_template(docked, out / "templates" / f"pose{rank}.cif")
                    )
                    print(
                        f"    {arm} pose{rank}: template {template.path.name}, "
                        f"{template.n_masked_residues} residues masked"
                    )

                    spec = write_boltz_yaml(
                        out / f"input_pose{rank}.yaml",
                        docked.antibody.seq,
                        docked.antigen.seq,
                        templates=[template],
                        template_force=True,
                        template_threshold=args.template_threshold,
                    )
                    run_prediction(
                        spec.path,
                        out / f"boltz_pose{rank}",
                        diffusion_samples=args.samples,
                        sampling_steps=args.sampling_steps,
                        seed=args.seed + rank,
                        use_msa_server=not args.no_msa_server,
                        # `force: true` templates are enforced by Boltz's
                        # TemplateReferencePotential, which is active whenever
                        # contact_guidance_update is set — and that defaults to True.
                        # --use_potentials is therefore not required here either.
                        use_potentials=args.use_potentials,
                        expect_patched=False,
                    )

                    for prediction in collect_predictions(out / f"boltz_pose{rank}"):
                        structure = load_predicted_complex(
                            prediction.structure_path, target.native,
                            name=f"{target.id}_{arm}_p{rank}_m{prediction.model_index}",
                        )
                        canonical = (
                            out / "structures" / f"pose{rank}_model_{prediction.model_index}.cif"
                        )
                        write_complex_cif(
                        with_canonical_chain_ids(structure, ANTIBODY_CHAIN_ID, ANTIGEN_CHAIN_ID),
                        canonical,
                    )
                        records.append(
                            {
                                "source_pose": rank,
                                "source_structure": source["structure"],
                                "model_index": prediction.model_index,
                                "structure": str(canonical),
                                "confidence": prediction.confidence(),
                                "template": template.as_dict(),
                            }
                        )

                write_json(
                    out / "manifest.json",
                    {
                        "target": target.id,
                        "arm": arm,
                        "settings": {
                            "top_k": args.top_k,
                            "diffusion_samples": args.samples,
                            "mask_cdrs": list(mask_cdrs) if mask else [],
                            "flank": args.flank,
                            "template_threshold": args.template_threshold,
                        },
                        "environment": describe_environment(),
                        "models": records,
                    },
                )
                print(f"    {arm}: {len(records)} structures")
                warn_if_empty(records, f"{target.id} / {arm}")

    print("\nArm A complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_guard(main))
