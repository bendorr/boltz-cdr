#!/usr/bin/env python
"""Choosing which loops get resampled. CPU only, no GPU, no Boltz weights.

The CDRs are found automatically by the IMGT annotator, but that is a default, not a
constraint. This script is both the demonstration of the override and the practical tool
for using it: run it on your target, read off the author residue numbers, edit them.

It walks through:

  1. What the automatic annotator selected, printed as **author residue numbers** — the
     numbers you would read off the structure in PyMOL, and exactly what you would type.
  2. A round-trip: feeding those same numbers back in reproduces the annotation exactly.
     That is what makes step 1 a safe starting point rather than a guess.
  3. Three realistic overrides — CDR3 only, CDR3 widened to include its anchor residues,
     and a discontinuous selection — each reported as residues, atoms, and how much of
     the true paratope it covers.
  4. What happens when a specification is wrong. Author numbering is arbitrary, so a typo
     has to be a loud error rather than a silently mis-selected loop.
  5. The equivalent `targets.yaml` block, ready to paste.

Usage:
    python scripts/07_cdr_selection_demo.py
    python scripts/07_cdr_selection_demo.py --target 9HUR
    python scripts/07_cdr_selection_demo.py --target 8QF4 --cdr-residues "E:cdr3=99-117"
"""

from __future__ import annotations

import argparse
import sys
import warnings

from _common import ROOT, load_targets, main_guard  # noqa: F401  (ROOT sets sys.path)
from boltz_cdr.cdr import CDR_NAMES
from boltz_cdr.cdr_spec import CDRSpec, CDRSpecError, annotation_for_chain
from boltz_cdr.metrics.contacts import paratope_residues

RULE = "=" * 82


def banner(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def resnum_ranges(chain, indices) -> str:
    """Chain positions -> compact author residue numbers, e.g. '28-35' or '99-105+110'."""
    if len(indices) == 0:
        return "-"
    nums = sorted(int(chain.resnums[i]) for i in indices)
    runs, start, previous = [], nums[0], nums[0]
    for value in nums[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous))
        start = previous = value
    runs.append((start, previous))
    return "+".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def describe(label: str, target, annotation) -> dict:
    """One row of the comparison table: what this definition actually selects."""
    chain = target.native.antibody
    selected = set(annotation.all_indices.tolist())
    paratope = {int(i) for i in paratope_residues(target.native)}
    n_atoms = int(chain.atom_mask_for_residues(annotation.all_indices).sum()) if selected else 0
    covered = len(selected & paratope)
    return {
        "label": label,
        "n_res": len(selected),
        "n_atoms": n_atoms,
        "paratope_covered": covered,
        "paratope_total": len(paratope),
        "spans": {name: resnum_ranges(chain, annotation[name]) for name in CDR_NAMES},
    }


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'definition':<30}{'res':>5}{'atoms':>7}{'paratope':>10}   "
        f"{'CDR1':<12}{'CDR2':<12}{'CDR3'}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        coverage = f"{row['paratope_covered']}/{row['paratope_total']}"
        print(
            f"{row['label']:<30}{row['n_res']:>5}{row['n_atoms']:>7}{coverage:>10}   "
            f"{row['spans']['cdr1']:<12}{row['spans']['cdr2']:<12}{row['spans']['cdr3']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="8QF4")
    parser.add_argument("--cdr-residues", default=None, dest="cdr_residues",
                        help="an extra specification of your own to include in the comparison")
    args = parser.parse_args()

    target = load_targets(only=[args.target], include_controls=True)[0]
    chain = target.native.antibody

    # ------------------------------------------------------- 1. what the annotator chose
    banner(f"1. AUTOMATIC ANNOTATION — {target.id}, antibody chain {chain.chain_id}")
    auto = annotation_for_chain(chain)
    print(f"sequence: {chain.n_res} residues, author numbering {chain.resnums.min()}"
          f"-{chain.resnums.max()}")
    print(f"\n{auto.summary()}\n")
    print("As author residue numbers — this is what you would type:")
    for name in CDR_NAMES:
        print(f"    {name.upper()}  {resnum_ranges(chain, auto[name])}")
    positional = ",".join(resnum_ranges(chain, auto[name]) for name in CDR_NAMES)
    print(f"\n    --cdr-residues \"{chain.chain_id}:{positional}\"")

    if chain.resnums.min() != 1:
        print(
            f"\nNote the offset: this chain starts at author residue "
            f"{chain.resnums.min()}, not 1. Author numbering is arbitrary, which is why "
            f"every number you supply is validated against the structure."
        )

    # ---------------------------------------------------------------- 2. the round-trip
    banner("2. ROUND-TRIP — feeding those numbers back in must change nothing")
    echoed = CDRSpec.parse(f"{chain.chain_id}:{positional}")
    reproduced = annotation_for_chain(chain, echoed)
    identical = all(
        list(reproduced[name]) == list(auto[name]) for name in CDR_NAMES
    )
    print(f"specification: {echoed.summary()}")
    print(f"reproduces the automatic annotation exactly: {identical}")
    if not identical:
        print("  (!) mismatch — this should never happen; please report it")
        return 1

    # -------------------------------------------------------------- 3. useful overrides
    banner("3. OVERRIDES — the same target, defined three other ways")
    cdr3_lo = int(chain.resnums[auto.cdr3.min()])
    cdr3_hi = int(chain.resnums[auto.cdr3.max()])
    lo_bound, hi_bound = int(chain.resnums.min()), int(chain.resnums.max())
    wide_lo, wide_hi = max(cdr3_lo - 2, lo_bound), min(cdr3_hi + 2, hi_bound)
    mid = (cdr3_lo + cdr3_hi) // 2

    variants = [
        (
            "automatic (default)",
            None,
            "the IMGT annotator; used when nothing is specified",
        ),
        (
            "CDR3 only",
            f"{chain.chain_id}:cdr3={cdr3_lo}-{cdr3_hi}",
            "CDR3 dominates the paratope, so spending the whole sampling budget on it "
            "is a reasonable strategy",
        ),
        (
            "CDR3 + anchors",
            f"{chain.chain_id}:cdr3={wide_lo}-{wide_hi}",
            "widened by two residues either side; the anchors set the loop take-off "
            "geometry and freeing them lets the base of the loop move",
        ),
        (
            "CDR1 + CDR3, discontinuous",
            f"{chain.chain_id}:cdr1={resnum_ranges(chain, auto.cdr1)},"
            f"cdr3={cdr3_lo}-{mid}+{mid + 2}-{cdr3_hi}",
            "loops need not be contiguous — '+' joins stretches, and CDR2 is left out "
            "entirely",
        ),
    ]
    if args.cdr_residues:
        variants.append(("yours", args.cdr_residues, "the specification you passed in"))

    rows = []
    for label, text, rationale in variants:
        spec = CDRSpec.parse(text) if text else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # divergence is the point here
            annotation = annotation_for_chain(chain, spec)
        rows.append(describe(label, target, annotation))
        if text:
            print(f"  {label:<28} --cdr-residues \"{text}\"")
            print(f"  {'':<28} {rationale}")
    print()
    print_table(rows)
    print(
        "\n'paratope' counts how many of the antibody residues that actually touch the "
        "antigen in the crystal structure fall inside each definition. A definition that "
        "misses most of the paratope will not resample the geometry that matters."
    )

    # -------------------------------------------------------------- 4. bad input
    banner("4. VALIDATION — a mis-numbered specification must fail loudly")
    bad = [
        (f"{target.antigen_chain}:{cdr3_lo}-{cdr3_hi}", "names the antigen chain"),
        (f"{chain.chain_id}:{hi_bound + 50}-{hi_bound + 60}", "residues that do not exist"),
        (f"{chain.chain_id}:cdr1={cdr3_lo}-{cdr3_hi},cdr2={cdr3_lo}-{cdr3_hi}",
         "two loops claiming the same residues"),
        (f"{chain.chain_id}:{cdr3_hi}-{cdr3_lo}", "a range written backwards"),
        (f"{chain.chain_id}:not-a-number", "unparseable"),
    ]
    for text, why in bad:
        try:
            annotation_for_chain(chain, CDRSpec.parse(text))
        except CDRSpecError as exc:
            print(f"  {why}:\n    --cdr-residues \"{text}\"\n    -> {exc}\n")
        else:
            print(f"  (!) {why}: NOT rejected — this is a bug\n")
            return 1

    # ------------------------------------------------------------------ 5. yaml form
    banner("5. THE SAME THING IN targets.yaml")
    print("Per-target definitions live in data/targets.yaml and apply to every stage:\n")
    print(f"  - id: {target.id}")
    print(f"    antibody_chain: {chain.chain_id}")
    print(f"    antigen_chain: {target.antigen_chain}")
    print("    cdr_residues:")
    print(f"      chain: {chain.chain_id}")
    for name in CDR_NAMES:
        print(f'      {name}: "{resnum_ranges(chain, auto[name])}"')
    print(
        "\nPrecedence: --cdr-residues  >  cdr_residues  >  cdr_spans (legacy, 0-based)"
        "  >  automatic\n"
        "\nThe definition flows through every stage that touches CDRs: which loops are\n"
        "deleted from the Arm A template, which atoms get scaled noise and gradient\n"
        "guidance in Arm B, and which residues the per-CDR RMSD metrics report on."
    )

    banner("DONE")
    print("Try it on a real run:")
    print(f"  python scripts/05_backward_pass_demo.py --target {target.id} \\")
    print(f"      --cdr-residues \"{chain.chain_id}:cdr3={cdr3_lo}-{cdr3_hi}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main_guard(main))
