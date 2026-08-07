"""Structure I/O and a lightweight, numpy-native representation of a two-chain complex.

Everything downstream (metrics, CDR annotation, template construction) operates on the
plain-numpy `Chain`/`Complex` types defined here rather than on gemmi objects, so that
those modules stay free of structural-library coupling and are trivially unit-testable.

No torch, no Boltz — this module runs on a laptop.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

# Residues we treat as polymer. gemmi's one_letter_code returns lowercase for
# non-standard residues, which we normalize to 'X'.
_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class Chain:
    """A single polymer chain, flattened into per-residue and per-atom arrays.

    The two levels are linked by `atom_res_index`, which maps every atom to the index
    of its residue in `resnums`/`resnames`/`seq`.
    """

    chain_id: str
    resnums: np.ndarray  # (n_res,) int   author residue numbers
    resnames: list[str]  # (n_res,)       three-letter codes
    seq: str  # (n_res,)                  one-letter sequence
    atom_names: np.ndarray  # (n_atom,) <U4
    atom_elements: np.ndarray  # (n_atom,) <U2
    atom_res_index: np.ndarray  # (n_atom,) int -> index into resnums
    coords: np.ndarray  # (n_atom, 3) float64
    bfactors: np.ndarray  # (n_atom,) float64
    # Where this chain came from — the PDB model number for a multi-model entry, or a
    # file stem for a set of separate predictions. Purely descriptive: it is what labels
    # a structure in a legend, and never participates in selection or geometry.
    model_id: str = ""

    def __post_init__(self) -> None:
        n_res = len(self.resnums)
        if len(self.resnames) != n_res or len(self.seq) != n_res:
            msg = f"chain {self.chain_id}: residue arrays disagree in length"
            raise ValueError(msg)
        n_atom = len(self.coords)
        for name in ("atom_names", "atom_elements", "atom_res_index", "bfactors"):
            if len(getattr(self, name)) != n_atom:
                msg = f"chain {self.chain_id}: {name} does not match coords length"
                raise ValueError(msg)

    @property
    def n_res(self) -> int:
        return len(self.resnums)

    @property
    def n_atom(self) -> int:
        return len(self.coords)

    def ca_coords(self) -> np.ndarray:
        """(n_res, 3) CA coordinates. NaN for residues with no CA (should not happen)."""
        out = np.full((self.n_res, 3), np.nan)
        is_ca = self.atom_names == "CA"
        out[self.atom_res_index[is_ca]] = self.coords[is_ca]
        return out

    def residue_atom_indices(self, res_index: int) -> np.ndarray:
        """Atom indices belonging to residue `res_index`."""
        return np.flatnonzero(self.atom_res_index == res_index)

    def atom_mask_for_residues(self, res_indices) -> np.ndarray:
        """(n_atom,) bool mask selecting all atoms of the given residue indices."""
        wanted = np.zeros(self.n_res, dtype=bool)
        wanted[np.asarray(list(res_indices), dtype=int)] = True
        return wanted[self.atom_res_index]

    def subset_residues(self, res_indices) -> Chain:
        """A new Chain containing only the given residues, in the order supplied."""
        res_indices = np.asarray(list(res_indices), dtype=int)
        keep_atom = np.zeros(self.n_atom, dtype=bool)
        remap = {int(old): new for new, old in enumerate(res_indices)}
        for old in res_indices:
            keep_atom |= self.atom_res_index == old
        atom_idx = np.flatnonzero(keep_atom)
        new_atom_res = np.array([remap[int(i)] for i in self.atom_res_index[atom_idx]])
        # Re-sort atoms so they follow the requested residue order.
        order = np.argsort(new_atom_res, kind="stable")
        atom_idx = atom_idx[order]
        new_atom_res = new_atom_res[order]
        return Chain(
            chain_id=self.chain_id,
            resnums=self.resnums[res_indices],
            resnames=[self.resnames[i] for i in res_indices],
            seq="".join(self.seq[i] for i in res_indices),
            atom_names=self.atom_names[atom_idx],
            atom_elements=self.atom_elements[atom_idx],
            atom_res_index=new_atom_res,
            coords=self.coords[atom_idx],
            bfactors=self.bfactors[atom_idx],
            model_id=self.model_id,
        )


@dataclass
class Complex:
    """A nanobody + antigen pair."""

    name: str
    antibody: Chain
    antigen: Chain
    meta: dict = field(default_factory=dict)

    @property
    def chains(self) -> tuple[Chain, Chain]:
        return self.antibody, self.antigen


def fetch_cif(pdb_id: str, out_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Download an mmCIF from RCSB. Returns the local path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pdb_id.lower()}.cif"
    if path.exists() and not overwrite:
        return path
    url = RCSB_CIF_URL.format(pdb_id=pdb_id.upper())
    urllib.request.urlretrieve(url, path)
    return path


def _clean_structure(st: gemmi.Structure) -> gemmi.Structure:
    """Canonicalize: first model, no waters/ligands/H, single altloc."""
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    return st


def _chain_from_gemmi(ch: gemmi.Chain) -> Chain:
    resnums, resnames, seq = [], [], []
    atom_names, atom_elements, atom_res_index, coords, bfac = [], [], [], [], []
    for res_i, res in enumerate(ch):
        resnums.append(res.seqid.num)
        resnames.append(res.name)
        one = gemmi.find_tabulated_residue(res.name)
        letter = one.one_letter_code.upper() if one is not None else "X"
        seq.append(letter if letter in _STANDARD_AA else "X")
        for atom in res:
            atom_names.append(atom.name)
            atom_elements.append(atom.element.name)
            atom_res_index.append(res_i)
            coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
            bfac.append(atom.b_iso)
    return Chain(
        chain_id=ch.name,
        resnums=np.asarray(resnums, dtype=int),
        resnames=resnames,
        seq="".join(seq),
        atom_names=np.asarray(atom_names, dtype="<U4"),
        atom_elements=np.asarray(atom_elements, dtype="<U2"),
        atom_res_index=np.asarray(atom_res_index, dtype=int),
        coords=np.asarray(coords, dtype=float).reshape(-1, 3),
        bfactors=np.asarray(bfac, dtype=float),
    )


def load_chains(path: str | Path, *, min_res: int = 10) -> dict[str, Chain]:
    """Load every polymer chain of >= `min_res` residues from a CIF/PDB file."""
    st = _clean_structure(gemmi.read_structure(str(path)))
    out: dict[str, Chain] = {}
    for ch in st[0]:
        if len(ch) < min_res:
            continue
        out[ch.name] = _chain_from_gemmi(ch)
    return out


def load_models(
    path: str | Path, chain_id: str | None = None, *, min_res: int = 10
) -> list[Chain]:
    """Every model of a multi-model file, as a list of `Chain`.

    NMR depositions carry an experimentally-determined conformational ensemble in a single
    entry — typically 20 models of one molecule — which makes them a source of real loop
    flexibility for testing and demonstration, with no prediction required.

    `chain_id` selects which chain to take from each model; the longest polymer chain is
    used when it is omitted.
    """
    st = _clean_structure(gemmi.read_structure(str(path)))
    models: list[Chain] = []
    for model in st:
        candidates = [ch for ch in model if len(ch) >= min_res]
        if chain_id is not None:
            candidates = [ch for ch in candidates if ch.name == chain_id]
        if not candidates:
            continue
        chain = _chain_from_gemmi(max(candidates, key=len))
        chain.model_id = f"model {model.num}"
        models.append(chain)
    if not models:
        msg = f"{path}: no polymer chain of >= {min_res} residues found"
        raise ValueError(msg)
    return models


def load_complex(
    path: str | Path,
    antibody_chain: str,
    antigen_chain: str,
    *,
    name: str | None = None,
) -> Complex:
    """Load a two-chain complex by explicit chain IDs."""
    chains = load_chains(path)
    missing = [c for c in (antibody_chain, antigen_chain) if c not in chains]
    if missing:
        msg = f"{path}: chain(s) {missing} not found; available: {sorted(chains)}"
        raise KeyError(msg)
    complex_name = name or Path(path).stem
    for chain in (chains[antibody_chain], chains[antigen_chain]):
        chain.model_id = complex_name
    return Complex(
        name=complex_name,
        antibody=chains[antibody_chain],
        antigen=chains[antigen_chain],
        meta={"path": str(path)},
    )


def load_predicted_complex(
    path: str | Path,
    reference: Complex,
    *,
    name: str | None = None,
) -> Complex:
    """Load a Boltz-2 prediction and assign chains by sequence identity to `reference`.

    Boltz renames and renumbers chains, so we cannot rely on the original chain IDs.
    We assign each predicted chain to whichever reference chain it matches best.
    """
    chains = load_chains(path)
    if len(chains) < 2:  # noqa: PLR2004
        msg = f"{path}: expected >=2 polymer chains, found {len(chains)}"
        raise ValueError(msg)

    ref_seqs = {"antibody": reference.antibody.seq, "antigen": reference.antigen.seq}
    scores = {
        (role, cid): _identity_fraction(ref_seq, ch.seq)
        for role, ref_seq in ref_seqs.items()
        for cid, ch in chains.items()
    }
    # Greedy assignment; with two roles and few chains this is optimal in practice.
    assigned: dict[str, str] = {}
    for role, cid in sorted(scores, key=lambda k: -scores[k]):
        if role in assigned or cid in assigned.values():
            continue
        assigned[role] = cid
        if len(assigned) == 2:  # noqa: PLR2004
            break

    complex_name = name or Path(path).stem
    for role in ("antibody", "antigen"):
        chains[assigned[role]].model_id = complex_name
    return Complex(
        name=complex_name,
        antibody=chains[assigned["antibody"]],
        antigen=chains[assigned["antigen"]],
        meta={
            "path": str(path),
            "chain_assignment": assigned,
            "assignment_identity": {
                role: round(scores[(role, cid)], 3) for role, cid in assigned.items()
            },
        },
    )


def _identity_fraction(a: str, b: str) -> float:
    """Fast ungapped-prefix identity estimate, used only for chain role assignment."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / max(len(a), len(b))


def with_canonical_chain_ids(
    cx: Complex, antibody_chain_id: str = "A", antigen_chain_id: str = "B"
) -> Complex:
    """Copy of `cx` with its chains renamed to the pipeline's canonical IDs.

    Boltz names output chains however it likes, and `load_predicted_complex` assigns the
    antibody/antigen *roles* by sequence identity without touching the names. Writing such
    a Complex straight back out would preserve Boltz's names, while every later stage
    re-loads it with a hard-coded ("A", "B"). That agrees today only because Boltz happens
    to echo the input YAML's chain IDs in the order we wrote them; if it ever did not, the
    antibody and antigen would be silently transposed and every metric would be garbage
    without anything raising. Renaming here removes the coupling.
    """
    import copy

    out = copy.deepcopy(cx)
    out.antibody.chain_id = antibody_chain_id
    out.antigen.chain_id = antigen_chain_id
    return out


def write_complex_cif(cx: Complex, path: str | Path, *, name: str | None = None) -> Path:
    """Write a Complex back out as mmCIF (used to emit masked templates)."""
    st = gemmi.Structure()
    st.name = name or cx.name
    st.spacegroup_hm = "P 1"
    st.cell = gemmi.UnitCell(1, 1, 1, 90, 90, 90)
    model = gemmi.Model("1")
    for chain in cx.chains:
        gch = gemmi.Chain(chain.chain_id)
        for res_i in range(chain.n_res):
            res = gemmi.Residue()
            res.name = chain.resnames[res_i]
            res.seqid = gemmi.SeqId(int(chain.resnums[res_i]), " ")
            res.het_flag = "A"
            for atom_i in chain.residue_atom_indices(res_i):
                at = gemmi.Atom()
                at.name = str(chain.atom_names[atom_i])
                at.element = gemmi.Element(str(chain.atom_elements[atom_i]))
                at.pos = gemmi.Position(*chain.coords[atom_i])
                at.b_iso = float(chain.bfactors[atom_i])
                at.occ = 1.0
                res.add_atom(at)
            gch.add_residue(res)
        model.add_chain(gch)
    st.add_model(model)
    st.setup_entities()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    st.make_mmcif_document().write_file(str(path))
    return path


def align_sequences(query: str, target: str) -> list[tuple[int, int]]:
    """Global alignment; returns (query_index, target_index) pairs for aligned columns.

    Used to build residue correspondence between a prediction and its crystal
    structure, which generally differ in numbering and may differ in construct
    boundaries (tags, truncations).
    """
    from Bio import Align

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.substitution_matrix = _blosum62()

    aln = aligner.align(query, target)[0]
    pairs: list[tuple[int, int]] = []
    for (qs, qe), (ts, _te) in zip(aln.aligned[0], aln.aligned[1], strict=True):
        pairs.extend((qs + k, ts + k) for k in range(qe - qs))
    return pairs


_BLOSUM62 = None


def _blosum62():
    global _BLOSUM62  # noqa: PLW0603 - cheap module-level cache
    if _BLOSUM62 is None:
        from Bio.Align import substitution_matrices

        _BLOSUM62 = substitution_matrices.load("BLOSUM62")
    return _BLOSUM62
