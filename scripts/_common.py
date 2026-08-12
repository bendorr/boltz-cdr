"""Shared helpers for the run scripts."""

from __future__ import annotations

import json
import sys
import textwrap
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boltz_cdr.cdr import CDRAnnotation, annotation_from_spans  # noqa: E402
from boltz_cdr.cdr_spec import (  # noqa: E402
    CDRSpec,
    annotation_for_chain,
    transfer_annotation,
)
from boltz_cdr.pdb_io import Complex, fetch_cif, load_complex  # noqa: E402

DEFAULT_TARGETS = ROOT / "data" / "targets.yaml"
DEFAULT_PDB_DIR = ROOT / "data" / "pdb"
DEFAULT_RESULTS = ROOT / "results"


@dataclass
class Target:
    """One benchmark complex, with its crystal structure and CDR annotation."""

    id: str
    name: str
    antibody_chain: str
    antigen_chain: str
    native: Complex
    annotation: CDRAnnotation
    meta: dict
    cdr_spec: CDRSpec | None = None

    @property
    def antibody_sequence(self) -> str:
        return self.native.antibody.seq

    @property
    def antigen_sequence(self) -> str:
        return self.native.antigen.seq

    @property
    def cdr_source(self) -> str:
        return "user" if self.cdr_spec is not None else "automatic IMGT"

    def annotation_for(self, seq: str) -> CDRAnnotation:
        """The CDR annotation for a *predicted* antibody chain of sequence `seq`.

        A user specification is authoritative and must be carried across rather than
        re-derived, so it is transferred by alignment when the sequences differ (they are
        usually identical — Boltz is asked to predict exactly the resolved crystal
        sequence). Without a specification, re-running the automatic annotator on the
        prediction's own sequence is more robust than transferring.
        """
        if seq == self.antibody_sequence:
            return self.annotation
        if self.cdr_spec is not None:
            return transfer_annotation(self.annotation, seq)
        from boltz_cdr.cdr import annotate_vhh

        return annotate_vhh(seq)

    def describe(self) -> str:
        return (
            f"{self.id} ({self.name}): nanobody {self.native.antibody.n_res} aa, "
            f"antigen {self.native.antigen.n_res} aa | {self.annotation.summary()} "
            f"[{self.cdr_source}]"
        )


def load_targets(
    path: str | Path = DEFAULT_TARGETS,
    pdb_dir: str | Path = DEFAULT_PDB_DIR,
    *,
    only: list[str] | None = None,
    include_controls: bool = False,
    cdr_residues: str | None = None,
) -> list[Target]:
    """Load benchmark targets, downloading structures on demand.

    CDR definitions are taken, in order of precedence:
      1. `cdr_residues` — a CLI override, author numbering, applied to every target
      2. the target's `cdr_residues` block in the YAML, also author numbering
      3. the target's `cdr_spans` block (0-based indices; kept for backward compatibility)
      4. the automatic IMGT annotator
    """
    cli_spec = CDRSpec.parse(cdr_residues) if cdr_residues else None
    with Path(path).open() as fh:
        config = yaml.safe_load(fh)

    entries = list(config.get("targets", []))
    if include_controls:
        entries += list(config.get("controls", []))

    wanted = {t.upper() for t in only} if only else None
    targets: list[Target] = []
    for entry in entries:
        if wanted and entry["id"].upper() not in wanted:
            continue
        cif = fetch_cif(entry["id"], pdb_dir)
        native = load_complex(
            cif, entry["antibody_chain"], entry["antigen_chain"], name=entry["id"]
        )
        spec = cli_spec
        if spec is None and entry.get("cdr_residues"):
            spec = CDRSpec.from_mapping(
                entry["cdr_residues"], origin=f"targets.yaml[{entry['id']}]"
            )

        if spec is not None:
            annotation = annotation_for_chain(native.antibody, spec)
        elif entry.get("cdr_spans"):
            annotation = annotation_from_spans(native.antibody.seq, entry["cdr_spans"])
        else:
            annotation = annotation_for_chain(native.antibody)

        targets.append(
            Target(
                id=entry["id"],
                name=entry.get("name", entry["id"]),
                antibody_chain=entry["antibody_chain"],
                antigen_chain=entry["antigen_chain"],
                native=native,
                annotation=annotation,
                meta=entry,
                cdr_spec=spec,
            )
        )

    if wanted and len(targets) != len(wanted):
        found = {t.id.upper() for t in targets}
        msg = f"targets not found in {path}: {sorted(wanted - found)}"
        raise KeyError(msg)
    return targets


@contextmanager
def isolated_arm(out: str | Path, label: str):
    """Run one arm; on failure record the traceback beside its outputs and carry on.

    Without this a stage dies on whichever arm fails first and takes every remaining target
    with it, leaving no record of why — which is how a run ends with one target holding four
    arms, two holding two, and nothing on disk to explain the difference. The traceback is
    both printed and written to `error.txt` in the arm's own directory, so it survives the
    notebook session that produced it.
    """
    out = Path(out)
    try:
        yield
    except Exception:  # noqa: BLE001 - the point is to let the next arm run
        out.mkdir(parents=True, exist_ok=True)
        report = traceback.format_exc()
        (out / "error.txt").write_text(report)
        print(f"    FAILED: {label}")
        print(textwrap.indent(report.rstrip(), "    "))
        print(f"    traceback written to {out / 'error.txt'}")


def warn_if_empty(records: list, label: str) -> None:
    """An arm that produces nothing has failed, even when nothing raised.

    Boltz drops a record it cannot parse and still exits cleanly, so a malformed input shows
    up as a successful run with an empty manifest rather than as an error.
    """
    if not records:
        print(f"    WARNING: {label} produced no structures — the prediction ran but "
              f"returned nothing. Check the input Boltz was given.")


def arm_dir(results: str | Path, target_id: str, arm: str) -> Path:
    path = Path(results) / target_id / arm
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def read_json(path: str | Path) -> dict:
    with Path(path).open() as fh:
        return json.load(fh)


def require_boltz() -> None:
    """Fail early and clearly when a GPU-side script is run without Boltz installed."""
    try:
        import boltz  # noqa: F401
    except ImportError as exc:
        msg = (
            "This stage needs Boltz-2. Install it with `pip install -r requirements-gpu.txt` "
            "on a GPU machine (Colab A100/L4). Local CPU development covers everything "
            "except the model itself — try scripts/05_backward_pass_demo.py."
        )
        raise SystemExit(msg) from exc


CDR_RESIDUES_HELP = (
    "Explicit CDR definition in AUTHOR residue numbering, overriding the automatic IMGT "
    "annotator, as 'CHAIN:span[,span...]'. Positional spans map to CDR1/2/3 in order, "
    "e.g. 'E:27-38,56-65,105-117'. Names work too: 'E:cdr3=105-117'. A discontinuous loop "
    "uses '+': 'E:cdr3=105-110+115-117'. Every residue number is validated against the "
    "structure. Applies to all selected targets; per-target definitions go in "
    "targets.yaml under cdr_residues."
)


def add_cdr_residues_argument(parser) -> None:
    """Attach the shared --cdr-residues flag."""
    parser.add_argument("--cdr-residues", default=None, dest="cdr_residues",
                        help=CDR_RESIDUES_HELP)


def main_guard(main_fn) -> int:
    """Run a script entry point, reporting user-input errors without a traceback.

    A malformed or mis-numbered CDR specification is a user mistake, not a crash; the
    message already says exactly what is wrong, so a stack trace only buries it.
    """
    from boltz_cdr.cdr_spec import CDRSpecError

    try:
        return main_fn()
    except CDRSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
