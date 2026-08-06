#!/usr/bin/env python
"""Download and validate the benchmark complexes. CPU only, no Boltz.

Every check here exists because a silent failure at this stage would corrupt everything
downstream. A disordered CDR3 means there is no ground truth for the loop the method is
built around; a chain break means RMSDs are computed over an inconsistent atom set; a
tiny interface means the "complex" may be a crystal packing artifact.

Usage:
    python scripts/00_fetch_targets.py
    python scripts/00_fetch_targets.py --targets data/targets.yaml --include-controls
"""

from __future__ import annotations

import argparse

from _common import DEFAULT_PDB_DIR, DEFAULT_RESULTS, DEFAULT_TARGETS, load_targets, write_json
from boltz_cdr.metrics import build_correspondence, contact_report, interface_report
from boltz_cdr.metrics.contacts import epitope_residues, paratope_residues

MIN_INTERFACE_RESIDUES = 8
MIN_CDR3_LENGTH = 8


def check_chain_continuity(chain) -> list[tuple[int, int]]:
    """Gaps in author residue numbering — i.e. unmodeled stretches."""
    nums = chain.resnums
    return [
        (int(nums[i - 1]), int(nums[i]))
        for i in range(1, len(nums))
        if nums[i] != nums[i - 1] + 1
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS))
    parser.add_argument("--out", default=str(DEFAULT_PDB_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_RESULTS / "targets_manifest.json"))
    parser.add_argument("--include-controls", action="store_true")
    args = parser.parse_args()

    targets = load_targets(args.targets, args.out, include_controls=args.include_controls)
    manifest: dict = {"targets": []}
    problems: list[str] = []

    for target in targets:
        native = target.native
        corr = build_correspondence(native, native)
        contacts = contact_report(native, native, corr)
        interface = interface_report(native)

        ab_gaps = check_chain_continuity(native.antibody)
        ag_gaps = check_chain_continuity(native.antigen)
        cdr_residues = {int(i) for i in target.annotation.all_indices}
        paratope = {int(i) for i in paratope_residues(native)}
        cdr_in_paratope = len(cdr_residues & paratope)

        print(f"\n=== {target.describe()}")
        print(f"    released {target.meta.get('released')}  resolution {target.meta.get('resolution')} A")
        print(f"    chain gaps: antibody={ab_gaps or 'none'}  antigen={ag_gaps or 'none'}")
        print(
            f"    interface: {contacts.n_native_contacts} residue contacts, "
            f"{contacts.n_native_epitope_residues} epitope residues, "
            f"{len(paratope)} paratope residues ({cdr_in_paratope} of them CDR)"
        )
        print(
            f"    physics: Sc={interface.shape_complementarity:.3f} "
            f"BSA={interface.bsa_total:.0f} A^2 "
            f"H-bonds={interface.n_hbonds} salt bridges={interface.n_salt_bridges} "
            f"clashes={interface.n_clashes}"
        )

        # --- validation ---
        if ab_gaps:
            problems.append(f"{target.id}: antibody chain has gaps {ab_gaps}")
        if ag_gaps:
            problems.append(f"{target.id}: antigen chain has gaps {ag_gaps}")
        if not target.annotation.invariants_ok:
            problems.append(f"{target.id}: VHH invariant residues failed")
        if len(target.annotation.cdr3) < MIN_CDR3_LENGTH:
            problems.append(f"{target.id}: CDR3 only {len(target.annotation.cdr3)} residues")
        if contacts.n_native_epitope_residues < MIN_INTERFACE_RESIDUES:
            problems.append(
                f"{target.id}: only {contacts.n_native_epitope_residues} epitope residues"
            )
        if cdr_in_paratope < MIN_INTERFACE_RESIDUES:
            problems.append(
                f"{target.id}: only {cdr_in_paratope} CDR residues contact the antigen — "
                "the paratope may be framework-dominated, which this method does not target"
            )

        manifest["targets"].append(
            {
                "id": target.id,
                "name": target.name,
                "released": target.meta.get("released"),
                "resolution": target.meta.get("resolution"),
                "antibody_chain": target.antibody_chain,
                "antigen_chain": target.antigen_chain,
                "antibody_sequence": target.antibody_sequence,
                "antigen_sequence": target.antigen_sequence,
                "n_antibody_residues": native.antibody.n_res,
                "n_antigen_residues": native.antigen.n_res,
                "cdr_spans": target.annotation.as_dict(),
                "cdr_sequences": target.annotation.sequences(),
                "epitope_residue_indices": [int(i) for i in epitope_residues(native)],
                "paratope_residue_indices": sorted(paratope),
                "n_cdr_residues_in_paratope": cdr_in_paratope,
                "native_contacts": contacts.as_dict(),
                "native_interface": interface.as_dict(),
                "chain_gaps": {"antibody": ab_gaps, "antigen": ag_gaps},
            }
        )

    path = write_json(args.manifest, manifest)
    print(f"\nmanifest written to {path}")

    if problems:
        print("\nVALIDATION PROBLEMS:")
        for problem in problems:
            print(f"  ! {problem}")
        return 1
    print(f"\nAll {len(targets)} targets passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
