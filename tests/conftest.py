"""Shared fixtures. Structures are downloaded once and cached in `data/pdb`."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from boltz_cdr.pdb_io import fetch_cif, load_complex  # noqa: E402

PDB_DIR = ROOT / "data" / "pdb"
TEST_TARGET = ("8QF4", "E", "A")


def random_rotation(rng) -> np.ndarray:
    """A uniformly-random proper rotation (det = +1) via QR."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


@pytest.fixture(scope="session")
def native_complex():
    """The 8QF4 nanobody-antigen crystal structure."""
    pdb_id, ab_chain, ag_chain = TEST_TARGET
    try:
        path = fetch_cif(pdb_id, PDB_DIR)
    except Exception as exc:
        pytest.skip(f"could not fetch {pdb_id}: {exc}")
    return load_complex(path, ab_chain, ag_chain, name=pdb_id)
