#!/usr/bin/env python
"""Stage 0 — global docking with stock Boltz-2. Needs a GPU.

Serves two purposes at once:

  * It is the **baseline ablation arm**. Everything the two modified arms do has to be
    measured against "just run unmodified Boltz-2 N times", because that is what a
    scientist would otherwise do, and the claim that it saturates is the thing under test.
  * It produces the **docked poses** that Arms A and B build on.

Optionally also runs `--steered`, which is stock Boltz-2 with its own `--use_potentials`
physical steering enabled — a second baseline that separates "steering helps" from "our
particular steering helps".

Usage:
    python scripts/01_global_dock.py --target 8QF4 --samples 8
    python scripts/01_global_dock.py --all --samples 8 --steered
"""

from __future__ import annotations

import argparse

from _common import DEFAULT_RESULTS, arm_dir, load_targets, require_boltz, write_json
from boltz_cdr.pdb_io import (
    load_predicted_complex,
    with_canonical_chain_ids,
    write_complex_cif,
)
from boltz_cdr.yaml_io import ANTIBODY_CHAIN_ID, ANTIGEN_CHAIN_ID, write_boltz_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--samples", type=int, default=8,
                        help="diffusion samples per run (Boltz's --diffusion_samples)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--steered", action="store_true",
                        help="additionally run stock Boltz-2 with --use_potentials")
    parser.add_argument("--paired-msa", action="store_true",
                        help="give the nanobody an MSA too (default: single-sequence)")
    parser.add_argument("--no-msa-server", action="store_true")
    args = parser.parse_args()

    require_boltz()
    from boltz_cdr import patch
    from boltz_cdr.run import collect_predictions, describe_environment, run_prediction

    # The baseline must run on genuinely unmodified Boltz-2.
    patch.uninstall()

    targets = load_targets(only=None if args.all else args.targets)
    if not targets:
        raise SystemExit("no targets selected; pass --target ID or --all")

    print(describe_environment())

    for target in targets:
        arms = [("baseline", False)] + ([("baseline_steered", True)] if args.steered else [])
        for arm, use_potentials in arms:
            out = arm_dir(args.results, target.id, arm)
            print(f"\n=== {target.id} / {arm} -> {out}")

            spec = write_boltz_yaml(
                out / "input.yaml",
                target.antibody_sequence,
                target.antigen_sequence,
                single_sequence_antibody=not args.paired_msa,
            )
            run_prediction(
                spec.path,
                out / "boltz",
                diffusion_samples=args.samples,
                recycling_steps=args.recycling_steps,
                sampling_steps=args.sampling_steps,
                seed=args.seed,
                use_msa_server=not args.no_msa_server,
                use_potentials=use_potentials,
                expect_patched=False,
            )

            predictions = collect_predictions(out / "boltz")
            print(f"    {len(predictions)} structures produced")

            # Re-emit each prediction with chains assigned by sequence identity, so the
            # downstream stages never have to guess which chain Boltz called what.
            records = []
            for prediction in predictions:
                structure = load_predicted_complex(
                    prediction.structure_path, target.native,
                    name=f"{target.id}_{arm}_{prediction.model_index}",
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
                        "raw_structure": str(prediction.structure_path),
                        "confidence": prediction.confidence(),
                        "chain_assignment": structure.meta.get("chain_assignment"),
                    }
                )

            write_json(
                out / "manifest.json",
                {
                    "target": target.id,
                    "arm": arm,
                    "input_yaml": str(spec.path),
                    "settings": {
                        "diffusion_samples": args.samples,
                        "seed": args.seed,
                        "recycling_steps": args.recycling_steps,
                        "sampling_steps": args.sampling_steps,
                        "use_potentials": use_potentials,
                        "single_sequence_antibody": not args.paired_msa,
                    },
                    "environment": describe_environment(),
                    "models": records,
                },
            )
    print("\nStage 0 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
