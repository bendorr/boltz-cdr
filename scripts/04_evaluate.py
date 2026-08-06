#!/usr/bin/env python
"""Stages 3-4 — score every arm's ensemble and compare selection strategies. CPU only.

Reads whatever arms exist under `results/<target>/`, evaluates every structure against the
crystal structure, and emits four tables:

  samples.csv       one row per predicted structure: all accuracy metrics, all contact
                    metrics, all interface physics, all Boltz confidence fields
  arms.csv          per-arm accuracy summary — the ablation table
  ensembles.csv     per-arm diversity and best-of-N coverage
  scorers.csv       the scorer comparison: for each selection score, its rank correlation
                    with DockQ, the DockQ its top-1 pick actually achieves, and how much
                    of the oracle-vs-random gap it captures

This stage needs no GPU and no Boltz, so results can be re-analyzed on a laptop after the
Colab session has ended.

Usage:
    python scripts/04_evaluate.py --target 8QF4
    python scripts/04_evaluate.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import (
    DEFAULT_RESULTS,
    add_cdr_residues_argument,
    load_targets,
    main_guard,
    read_json,
    write_json,
)
from boltz_cdr.ensemble import ensemble_report
from boltz_cdr.metrics import build_correspondence, evaluate, interface_report
from boltz_cdr.pdb_io import load_complex
from boltz_cdr.scoring import (
    Sample,
    add_scores,
    arm_summary,
    evaluate_scorers,
    evaluate_scorers_by_arm,
)
from boltz_cdr.yaml_io import ANTIBODY_CHAIN_ID, ANTIGEN_CHAIN_ID


def load_arm_samples(target, arm_path: Path) -> tuple[list[Sample], bool]:
    """Evaluate every structure in one arm against the crystal structure.

    Returns the samples and whether the arm came from the synthetic generator.
    """
    manifest = read_json(arm_path / "manifest.json")
    synthetic = bool(manifest.get("synthetic"))
    samples: list[Sample] = []
    for record in manifest["models"]:
        structure = load_complex(
            record["structure"], ANTIBODY_CHAIN_ID, ANTIGEN_CHAIN_ID,
            name=Path(record["structure"]).stem,
        )
        corr = build_correspondence(structure, target.native)
        # Per-CDR RMSD indexes the *prediction*, so the annotation must be expressed on
        # the predicted chain. A user definition is transferred; otherwise the annotator
        # re-runs on the prediction's own sequence.
        annotation = target.annotation_for(structure.antibody.seq)
        samples.append(
            Sample(
                sample_id=f"{manifest['arm']}:{Path(record['structure']).stem}",
                arm=manifest["arm"],
                target=target.id,
                structure=structure,
                path=record["structure"],
                confidence=record.get("confidence") or {},
                interface=interface_report(structure),
                truth=evaluate(structure, target.native, corr, annotation),
            )
        )
    return samples, synthetic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out", default=None, help="defaults to <results>/analysis")
    add_cdr_residues_argument(parser)
    args = parser.parse_args()

    results = Path(args.results)
    out_dir = Path(args.out) if args.out else results / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(
        only=None if args.all else args.targets, cdr_residues=args.cdr_residues
    )
    if not targets:
        raise SystemExit("no targets selected; pass --target ID or --all")

    all_samples: list[Sample] = []
    ensemble_rows: list[dict] = []
    any_synthetic = False

    for target in targets:
        target_dir = results / target.id
        if not target_dir.exists():
            print(f"  {target.id}: no results — run stages 01-03 first")
            continue

        for arm_path in sorted(p for p in target_dir.iterdir() if (p / "manifest.json").exists()):
            samples, synthetic = load_arm_samples(target, arm_path)
            any_synthetic |= synthetic
            if not samples:
                continue
            all_samples.extend(samples)
            print(f"  {target.id}/{arm_path.name}: {len(samples)} structures evaluated")

            annotation = target.annotation_for(samples[0].structure.antibody.seq)
            report = ensemble_report(
                arm=arm_path.name,
                structures=[s.structure for s in samples],
                annotation=annotation,
                dockq=[s.truth.dockq.dockq for s in samples],
                ligand_rmsd=[s.truth.rmsd.ligand_rmsd for s in samples],
            )
            ensemble_rows.append({"target": target.id, **report.as_dict()})

    if not all_samples:
        raise SystemExit("no evaluated structures found under " + str(results))

    samples_df = add_scores(pd.DataFrame([s.row() for s in all_samples]))
    samples_df.to_csv(out_dir / "samples.csv", index=False)

    arms_df = arm_summary(samples_df)
    arms_df.to_csv(out_dir / "arms.csv", index=False)

    ensembles_df = pd.DataFrame(ensemble_rows)
    ensembles_df.to_csv(out_dir / "ensembles.csv", index=False)

    scorers_df = evaluate_scorers(samples_df)
    scorers_df.to_csv(out_dir / "scorers.csv", index=False)
    evaluate_scorers_by_arm(samples_df).to_csv(out_dir / "scorers_by_arm.csv", index=False)

    write_json(
        out_dir / "summary.json",
        {
            "n_samples": len(samples_df),
            "targets": sorted(samples_df["target"].unique().tolist()),
            "arms": sorted(samples_df["arm"].unique().tolist()),
            "best_dockq_overall": float(samples_df["dockq"].max()),
            "best_scorer_by_top1": scorers_df.iloc[0]["scorer"] if len(scorers_df) else None,
        },
    )

    pd.set_option("display.width", 200, "display.max_columns", 50)

    print("\n" + "=" * 78)
    print("PER-ARM ACCURACY  (the ablation)")
    print("=" * 78)
    columns = [c for c in
               ("target", "arm", "dockq_count", "dockq_mean", "dockq_max",
                "ligand_rmsd_min", "interface_rmsd_min", "cdr3_rmsd_min",
                "fnat_max", "epitope_recall_max")
               if c in arms_df.columns]
    print(arms_df[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 78)
    print("ENSEMBLE DIVERSITY AND COVERAGE")
    print("=" * 78)
    columns = ["target", "arm", "n_samples", "mean_pairwise_cdr3_rmsd",
               "max_pairwise_cdr3_rmsd", "n_clusters", "best_dockq", "mean_dockq",
               "n_acceptable"]
    print(ensembles_df[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 78)
    print("SELECTION SCORERS  (which score picks the best structure?)")
    print("=" * 78)
    print(scorers_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        "\nenrichment = (top1 - mean) / (oracle - mean): 1.0 picks the best member every "
        "time, 0.0 is no better than random, negative is anti-correlated."
    )
    if any_synthetic:
        print(
            "\n*** SYNTHETIC DATA — the `conf:*` rows are MEANINGLESS here. ***\n"
            "06_synthetic_ensemble.py fabricates confidence values as noisy functions of "
            "the true DockQ, so any confidence scorer will look good and `complex_iplddt`, "
            "which is generated as a deterministic function of DockQ, will look perfect. "
            "This run validates the analysis plumbing, nothing else. The `phys:*` rows are "
            "computed from coordinates and so do carry signal even here."
        )
    print(f"\nTables written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_guard(main))
