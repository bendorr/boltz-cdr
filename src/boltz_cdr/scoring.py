"""Selection scorers, and a head-to-head comparison of how well they rank an ensemble.

Generating a good structure and *knowing which one it is* are separate problems, and the
second is the one that decides whether any of this is usable. Best-of-N is not an
actionable number in a real campaign: nobody gets to look at the crystal structure before
choosing which model to hand to the next stage. What matters is top-1-by-some-scorer.

So this module defines two families of scorer and then measures them against each other:

  confidence   ipTM, pTM, pLDDT, interface pLDDT — read from Boltz-2's own sidecar JSON.
               Fast and free, but produced by a head trained largely on globular monomers,
               with a documented weak spot at antibody interfaces.
  physics      shape complementarity, H-bond density, salt bridges, buried surface,
               clashes, buried-unsatisfied polars — computed from coordinates alone.
               Slower, blind to the model's own beliefs, and therefore wrong in different
               places than the confidence head is.

`evaluate_scorers` reports, for each scorer: rank correlation with DockQ, the DockQ you
actually obtain by picking its top-1, and how that compares to both the oracle (best-of-N)
and to picking at random. The gap between top-1 and oracle is the value left on the table
by imperfect scoring, and it is usually large.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from boltz_cdr.metrics.dockq import FullReport
from boltz_cdr.metrics.interface import InterfaceReport
from boltz_cdr.pdb_io import Complex


@dataclass
class Sample:
    """One predicted structure with everything known about it."""

    sample_id: str
    arm: str
    target: str
    structure: Complex
    path: str = ""
    confidence: dict = field(default_factory=dict)
    interface: InterfaceReport | None = None
    truth: FullReport | None = None

    def row(self) -> dict:
        out: dict = {
            "sample_id": self.sample_id,
            "arm": self.arm,
            "target": self.target,
            "path": self.path,
        }
        out.update({f"conf_{k}": v for k, v in _flat_confidence(self.confidence).items()})
        if self.interface is not None:
            out.update(self.interface.as_dict())
        if self.truth is not None:
            out.update(self.truth.flat())
        return out


def _flat_confidence(conf: dict) -> dict[str, float]:
    """Flatten Boltz-2's confidence JSON, keeping only scalar numeric entries."""
    out: dict[str, float] = {}
    for key, value in conf.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, dict):
            for sub, sub_value in value.items():
                if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                    out[f"{key}_{sub}"] = float(sub_value)
    return out


# --- scorer definitions ------------------------------------------------------------
# Every scorer is (column, sign); sign = +1 when larger is better.

CONFIDENCE_SCORERS: dict[str, tuple[str, float]] = {
    "conf:boltz_confidence": ("conf_confidence_score", +1.0),
    "conf:iptm": ("conf_iptm", +1.0),
    "conf:ptm": ("conf_ptm", +1.0),
    "conf:complex_plddt": ("conf_complex_plddt", +1.0),
    "conf:complex_iplddt": ("conf_complex_iplddt", +1.0),
    "conf:complex_ipde": ("conf_complex_ipde", -1.0),
}

PHYSICS_SCORERS: dict[str, tuple[str, float]] = {
    "phys:shape_complementarity": ("shape_complementarity", +1.0),
    "phys:hbond_density": ("hbond_density", +1.0),
    "phys:n_hbonds": ("n_hbonds", +1.0),
    "phys:n_salt_bridges": ("n_salt_bridges", +1.0),
    "phys:bsa_total": ("bsa_total", +1.0),
    "phys:n_clashes": ("n_clashes", -1.0),
    "phys:buried_unsat_polars": ("n_buried_unsatisfied_polars", -1.0),
}

# Weights for the composite physics score. Chosen a priori from what is known to
# characterize real interfaces (packing quality first, polar satisfaction second, size
# third, penalties last) rather than fitted — with three targets, fitting them would be
# indistinguishable from memorizing the answer. Learning them on a large corpus of scored
# ensembles is the natural next step.
COMPOSITE_WEIGHTS: dict[str, float] = {
    "shape_complementarity": 1.0,
    "hbond_density": 0.6,
    "n_salt_bridges": 0.3,
    "bsa_total": 0.4,
    "n_clashes": -0.8,
    "n_buried_unsatisfied_polars": -0.4,
}


def zscore(values: np.ndarray) -> np.ndarray:
    """Z-score, returning zeros when the column is constant or all-NaN."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    if finite.sum() < 2:  # noqa: PLR2004
        return np.zeros_like(v)
    mu, sd = v[finite].mean(), v[finite].std()
    if sd == 0:
        return np.zeros_like(v)
    out = np.zeros_like(v)
    out[finite] = (v[finite] - mu) / sd
    return out


def composite_physics_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> np.ndarray:
    """Weighted sum of z-scored physics descriptors.

    Z-scoring is computed *within the frame passed in*, so this is a relative score for
    ranking one ensemble, not an absolute quantity comparable across targets.
    """
    weights = weights or COMPOSITE_WEIGHTS
    total = np.zeros(len(df))
    for column, weight in weights.items():
        if column in df.columns:
            total += weight * zscore(df[column].to_numpy())
    return total


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every scorer as a `score:<name>` column, oriented so higher is better."""
    out = df.copy()
    for name, (column, sign) in {**CONFIDENCE_SCORERS, **PHYSICS_SCORERS}.items():
        if column in out.columns:
            out[f"score:{name}"] = sign * out[column].astype(float)
    out["score:phys:composite"] = composite_physics_score(out)
    if "score:conf:iptm" in out.columns:
        # A simple hybrid: the model's interface confidence plus the structural evidence,
        # equally weighted after z-scoring.
        out["score:hybrid:iptm+physics"] = (
            zscore(out["score:conf:iptm"].to_numpy())
            + zscore(out["score:phys:composite"].to_numpy())
        )
    return out


def scorer_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("score:")]


@dataclass
class ScorerResult:
    scorer: str
    spearman: float
    pearson: float
    top1_dockq: float
    top1_capri: str
    oracle_dockq: float
    mean_dockq: float
    enrichment: float
    n_samples: int


def evaluate_scorers(
    df: pd.DataFrame, *, target: str = "dockq", scorers: list[str] | None = None
) -> pd.DataFrame:
    """Rank every scorer by how well it selects a good structure.

    `enrichment` is (top-1 DockQ - mean DockQ) / (oracle DockQ - mean DockQ): 1.0 means
    the scorer picks the best ensemble member every time, 0.0 means it does no better than
    random, and negative means it is anti-correlated.
    """
    from scipy.stats import pearsonr, spearmanr

    if target not in df.columns:
        msg = f"target column {target!r} not present; ground truth is required"
        raise KeyError(msg)

    scorers = scorers or scorer_columns(df)
    truth = df[target].to_numpy(dtype=float)
    valid_truth = np.isfinite(truth)
    if valid_truth.sum() < 2:  # noqa: PLR2004
        msg = "need at least 2 samples with ground truth"
        raise ValueError(msg)

    oracle = float(np.nanmax(truth))
    mean = float(np.nanmean(truth))
    capri = df["capri_class"].to_numpy() if "capri_class" in df.columns else None

    rows: list[ScorerResult] = []
    for column in scorers:
        score = df[column].to_numpy(dtype=float)
        ok = valid_truth & np.isfinite(score)
        if ok.sum() < 2:  # noqa: PLR2004
            continue
        rho = spearmanr(score[ok], truth[ok]).statistic if ok.sum() > 2 else float("nan")  # noqa: PLR2004
        r = pearsonr(score[ok], truth[ok]).statistic if ok.sum() > 2 else float("nan")  # noqa: PLR2004

        pick = int(np.flatnonzero(ok)[np.argmax(score[ok])])
        top1 = float(truth[pick])
        denominator = oracle - mean
        rows.append(
            ScorerResult(
                scorer=column.removeprefix("score:"),
                spearman=float(rho),
                pearson=float(r),
                top1_dockq=top1,
                top1_capri=str(capri[pick]) if capri is not None else "",
                oracle_dockq=oracle,
                mean_dockq=mean,
                enrichment=float((top1 - mean) / denominator) if denominator > 1e-9 else float("nan"),  # noqa: PLR2004
                n_samples=int(ok.sum()),
            )
        )

    result = pd.DataFrame([r.__dict__ for r in rows])
    return result.sort_values("top1_dockq", ascending=False).reset_index(drop=True)


def evaluate_scorers_by_arm(df: pd.DataFrame, *, target: str = "dockq") -> pd.DataFrame:
    """Scorer comparison run separately within each sampling arm."""
    frames = []
    for arm, group in df.groupby("arm"):
        if len(group) < 2:  # noqa: PLR2004
            continue
        sub = evaluate_scorers(group, target=target)
        sub.insert(0, "arm", arm)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_best(df: pd.DataFrame, scorer: str) -> pd.Series:
    """The row a given scorer would choose."""
    column = scorer if scorer.startswith("score:") else f"score:{scorer}"
    return df.loc[df[column].astype(float).idxmax()]


def arm_summary(df: pd.DataFrame, *, target: str = "dockq") -> pd.DataFrame:
    """Per-arm accuracy summary — the ablation table."""
    aggregations = {
        target: ["count", "mean", "max", "std"],
        "fnat": ["mean", "max"],
        "ligand_rmsd": ["mean", "min"],
        "interface_rmsd": ["mean", "min"],
        "cdr3_rmsd": ["mean", "min"],
        "epitope_recall": ["mean", "max"],
        "shape_complementarity": ["mean", "max"],
    }
    present = {k: v for k, v in aggregations.items() if k in df.columns}
    out = df.groupby(["target", "arm"]).agg(present)
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()
