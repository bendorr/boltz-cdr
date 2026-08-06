#!/usr/bin/env python
"""Build a synthetic ensemble by perturbing the crystal structure. CPU only, no Boltz.

Purpose: exercise the entire Stage 3-4 analysis path — evaluation, interface physics,
ensemble diversity, and the scorer comparison — without a GPU, so that everything
downstream of Boltz is debugged before a single Colab minute is spent, and so that the
analysis remains testable by anyone who cannot run the model.

The perturbations are designed to produce a *graded* spread of quality, because a scorer
comparison on an ensemble where everything is equally good measures nothing:

  near        small random displacement of CDR atoms only         -> near-native
  loop        large CDR-only perturbation, framework fixed        -> good pose, bad loops
  tilt        small rigid-body rotation of the whole nanobody     -> mid-range
  slide       rigid-body translation along the interface normal   -> degrading
  wrong       large rotation about the antigen center             -> wrong epitope

The synthetic "confidence" values are deliberately noisy functions of true quality, which
is what an imperfect confidence head looks like. They are stand-ins for plumbing purposes
only — no conclusion about real Boltz confidence can be drawn from them, and
`04_evaluate.py` run on real output is the only place those numbers mean anything.

Usage:
    python scripts/06_synthetic_ensemble.py --target 8QF4 --results results_synthetic
    python scripts/04_evaluate.py --target 8QF4 --results results_synthetic
"""

from __future__ import annotations

import argparse

import numpy as np

from _common import (
    add_cdr_residues_argument,
    arm_dir,
    load_targets,
    main_guard,
    write_json,
)
from boltz_cdr.metrics import build_correspondence
from boltz_cdr.metrics.dockq import compute_dockq
from boltz_cdr.pdb_io import Complex, with_canonical_chain_ids, write_complex_cif
from boltz_cdr.yaml_io import ANTIBODY_CHAIN_ID, ANTIGEN_CHAIN_ID


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation."""
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def perturb(
    native: Complex, cdr_atom_mask: np.ndarray, mode: str, magnitude: float, rng
) -> Complex:
    """Return a perturbed copy of `native`, renamed to the canonical chain IDs."""
    out = with_canonical_chain_ids(native, ANTIBODY_CHAIN_ID, ANTIGEN_CHAIN_ID)
    ab = out.antibody.coords

    if mode in ("near", "loop"):
        noise = rng.normal(scale=magnitude, size=ab[cdr_atom_mask].shape)
        ab[cdr_atom_mask] += noise
    elif mode == "tilt":
        center = ab.mean(axis=0)
        rot = rotation_matrix(rng.normal(size=3), magnitude)
        out.antibody.coords = (ab - center) @ rot.T + center
    elif mode == "slide":
        direction = ab[cdr_atom_mask].mean(axis=0) - out.antigen.coords.mean(axis=0)
        out.antibody.coords = ab + magnitude * direction / np.linalg.norm(direction)
    elif mode == "wrong":
        center = out.antigen.coords.mean(axis=0)
        rot = rotation_matrix(rng.normal(size=3), magnitude)
        out.antibody.coords = (ab - center) @ rot.T + center
    else:
        msg = f"unknown perturbation mode {mode!r}"
        raise ValueError(msg)
    return out


# (arm name, mode, magnitudes) — one arm per mode so the ablation table has structure.
SCHEDULE = (
    ("baseline", (("near", 0.4), ("tilt", 3.0), ("slide", 1.5), ("tilt", 8.0),
                  ("wrong", 25.0), ("slide", 4.0), ("wrong", 60.0), ("loop", 2.0))),
    ("armA", (("near", 0.3), ("loop", 1.2), ("loop", 2.0), ("loop", 3.0),
              ("near", 0.8), ("loop", 1.6), ("loop", 2.4), ("near", 0.5))),
    ("armB_noise", (("near", 0.5), ("loop", 1.5), ("loop", 2.5), ("tilt", 2.0),
                    ("loop", 3.5), ("near", 0.9), ("loop", 1.8), ("tilt", 5.0))),
    ("armB_guided", (("near", 0.3), ("near", 0.6), ("loop", 1.0), ("loop", 1.4),
                     ("loop", 2.0), ("near", 0.4), ("loop", 1.2), ("tilt", 1.5))),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--results", default="results_synthetic")
    parser.add_argument("--seed", type=int, default=0)
    add_cdr_residues_argument(parser)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    targets = load_targets(
        only=None if args.all else args.targets, cdr_residues=args.cdr_residues
    )
    if not targets:
        raise SystemExit("no targets selected; pass --target ID or --all")

    for target in targets:
        cdr_atom_mask = target.native.antibody.atom_mask_for_residues(
            target.annotation.all_indices
        )
        print(f"\n=== {target.id}: {cdr_atom_mask.sum()} CDR atoms")

        for arm, schedule in SCHEDULE:
            out = arm_dir(args.results, target.id, arm)
            records = []
            for index, (mode, magnitude) in enumerate(schedule):
                structure = perturb(target.native, cdr_atom_mask, mode, magnitude, rng)
                path = out / "structures" / f"model_{index}.cif"
                write_complex_cif(structure, path)

                truth = compute_dockq(
                    structure, target.native, build_correspondence(structure, target.native)
                )
                # An imperfect confidence head: correlated with truth, but noisily.
                noise = rng.normal(scale=0.12)
                records.append(
                    {
                        "model_index": index,
                        "structure": str(path),
                        "perturbation": {"mode": mode, "magnitude": magnitude},
                        "confidence": {
                            "confidence_score": float(np.clip(truth.dockq + noise, 0, 1)),
                            "iptm": float(np.clip(truth.dockq + rng.normal(scale=0.15), 0, 1)),
                            "ptm": float(np.clip(0.5 + 0.4 * truth.dockq + noise, 0, 1)),
                            "complex_plddt": float(np.clip(0.85 - 0.1 * rng.random(), 0, 1)),
                            "complex_iplddt": float(np.clip(truth.dockq * 0.8 + 0.15, 0, 1)),
                        },
                    }
                )
                print(f"    {arm:14s} {mode:6s} {magnitude:5.1f} -> DockQ {truth.dockq:.3f} [{truth.capri_class}]")

            write_json(
                out / "manifest.json",
                {
                    "target": target.id,
                    "arm": arm,
                    "synthetic": True,
                    "settings": {"seed": args.seed},
                    "models": records,
                },
            )

    print(
        f"\nSynthetic ensembles written to {args.results}/. "
        f"Now run:\n    python scripts/04_evaluate.py --all --results {args.results}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_guard(main))
