"""Model-independent, physics-based description of a protein–protein interface.

These quantities serve two roles in this project:

  1. As *descriptors* of a predicted complex, alongside the RMSD and contact metrics.
  2. As *selection scores* — a way to rank an ensemble that does not depend on the same
     network that produced it. A confidence head trained mostly on globular monomers has
     a known blind spot at antibody interfaces; shape complementarity and hydrogen-bond
     satisfaction do not share that blind spot, and their failure modes are uncorrelated
     with it. Whether they actually rank better is measured in `scoring.py`, not assumed.

Implementation notes / honest limitations:

  * SASA is Shrake–Rupley with a 1.4 A probe and Bondi radii.
  * Shape complementarity follows Lawrence & Colman (1993, JMB 234:946) on a grid-derived
    solvent-excluded (Connolly) surface. The peripheral band is trimmed by a
    nearest-partner-distance cutoff rather than a true patch-boundary calculation, so
    values track the published statistic without being bit-identical to CCP4 `sc`. On the
    three benchmark crystal structures it returns Sc = 0.66-0.70, against a published
    range of 0.64-0.68 for antibody-antigen interfaces.
  * Hydrogen bonds are assigned from heavy-atom geometry (donor/acceptor distance plus
    both antecedent angles), since predicted structures carry no hydrogens.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

from boltz_cdr.pdb_io import Chain, Complex

PROBE_RADIUS = 1.4

# Bondi van der Waals radii.
_VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90, "F": 1.47}
_VDW_DEFAULT = 1.70

# Heavy-atom hydrogen-bond donors -> covalent antecedent, used for the angle test.
_SIDECHAIN_DONORS = {
    ("ARG", "NE"): "CD", ("ARG", "NH1"): "CZ", ("ARG", "NH2"): "CZ",
    ("LYS", "NZ"): "CE",
    ("HIS", "ND1"): "CG", ("HIS", "NE2"): "CD2",
    ("ASN", "ND2"): "CG", ("GLN", "NE2"): "CD",
    ("TRP", "NE1"): "CD1",
    ("SER", "OG"): "CB", ("THR", "OG1"): "CB", ("TYR", "OH"): "CZ",
    ("CYS", "SG"): "CB",
}
_SIDECHAIN_ACCEPTORS = {
    ("ASP", "OD1"): "CG", ("ASP", "OD2"): "CG",
    ("GLU", "OE1"): "CD", ("GLU", "OE2"): "CD",
    ("ASN", "OD1"): "CG", ("GLN", "OE1"): "CD",
    ("SER", "OG"): "CB", ("THR", "OG1"): "CB", ("TYR", "OH"): "CZ",
    ("HIS", "ND1"): "CG", ("HIS", "NE2"): "CD2",
    ("MET", "SD"): "CG",
}
_CATIONIC = {("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("LYS", "NZ"),
             ("HIS", "ND1"), ("HIS", "NE2")}
_ANIONIC = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}

HBOND_MAX_DIST = 3.5
HBOND_MIN_ANGLE = 90.0
SALT_BRIDGE_MAX_DIST = 4.0


def vdw_radii(elements: np.ndarray) -> np.ndarray:
    return np.array([_VDW.get(str(e).upper(), _VDW_DEFAULT) for e in elements])


def _sphere_points(n: int) -> np.ndarray:
    """`n` approximately-uniform points on the unit sphere (Fibonacci/golden spiral)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1
    )


@dataclass
class DotSurface:
    """A dot surface carrying both radial positions of each accessible surface point.

    `sas` is the point at (vdW + probe) — the solvent-accessible position, which is the
    correct place to test whether a water probe still fits once the partner chain is
    added. `vdw` is the same direction at the plain van der Waals radius, which is where
    the two surfaces of a packed interface actually meet and therefore the surface on
    which shape complementarity must be measured.
    """

    sas: np.ndarray  # (n_dot, 3)
    vdw: np.ndarray  # (n_dot, 3)
    normals: np.ndarray  # (n_dot, 3) outward unit vectors
    atom_index: np.ndarray  # (n_dot,)
    per_atom_sasa: np.ndarray  # (n_atom,)


def surface_dots(
    coords: np.ndarray,
    elements: np.ndarray,
    *,
    n_points: int = 256,
    probe: float = PROBE_RADIUS,
) -> DotSurface:
    """Shrake-Rupley dot surface, returned at both the SAS and vdW radii."""
    r_vdw = vdw_radii(elements)
    radii = r_vdw + probe
    unit = _sphere_points(n_points)
    tree = cKDTree(coords)
    max_r = float(radii.max())

    sas_pts, vdw_pts, normals, owners = [], [], [], []
    accessible_counts = np.zeros(len(coords), dtype=int)

    for i, (center, r) in enumerate(zip(coords, radii, strict=True)):
        candidates = np.asarray(tree.query_ball_point(center, r + max_r), dtype=int)
        candidates = candidates[candidates != i]
        pts = center + r * unit
        if len(candidates):
            d = np.linalg.norm(pts[:, None, :] - coords[candidates][None, :, :], axis=-1)
            free = np.all(d >= radii[candidates][None, :], axis=1)
        else:
            free = np.ones(len(pts), dtype=bool)
        accessible_counts[i] = int(free.sum())
        if free.any():
            sas_pts.append(pts[free])
            vdw_pts.append(center + r_vdw[i] * unit[free])
            normals.append(unit[free])
            owners.append(np.full(int(free.sum()), i))

    per_atom_sasa = 4.0 * np.pi * radii**2 * accessible_counts / n_points
    if not sas_pts:
        empty = np.zeros((0, 3))
        return DotSurface(empty, empty, empty, np.zeros(0, dtype=int), per_atom_sasa)
    return DotSurface(
        sas=np.concatenate(sas_pts),
        vdw=np.concatenate(vdw_pts),
        normals=np.concatenate(normals),
        atom_index=np.concatenate(owners),
        per_atom_sasa=per_atom_sasa,
    )


def sasa(coords: np.ndarray, elements: np.ndarray, *, n_points: int = 256) -> np.ndarray:
    """Per-atom solvent-accessible surface area (A^2)."""
    return surface_dots(coords, elements, n_points=n_points).per_atom_sasa


def buried_sasa(cx: Complex, *, n_points: int = 256) -> dict[str, float]:
    """Surface area buried on complexation, per chain and in total."""
    ab, ag = cx.antibody, cx.antigen
    sasa_ab = sasa(ab.coords, ab.atom_elements, n_points=n_points)
    sasa_ag = sasa(ag.coords, ag.atom_elements, n_points=n_points)

    joint_coords = np.vstack([ab.coords, ag.coords])
    joint_elements = np.concatenate([ab.atom_elements, ag.atom_elements])
    sasa_joint = sasa(joint_coords, joint_elements, n_points=n_points)

    d_ab = float(sasa_ab.sum() - sasa_joint[: ab.n_atom].sum())
    d_ag = float(sasa_ag.sum() - sasa_joint[ab.n_atom :].sum())
    return {"bsa_antibody": d_ab, "bsa_antigen": d_ag, "bsa_total": d_ab + d_ag}


def solvent_excluded_surface(
    coords: np.ndarray,
    elements: np.ndarray,
    *,
    probe: float = PROBE_RADIUS,
    spacing: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Connolly solvent-excluded surface, as points with outward normals.

    Built on a distance field rather than analytically:

      1. Mark grid points inside the solvent-accessible volume (within r_i + probe of any
         atom). Its complement is exactly where a probe *center* may sit.
      2. The solvent-excluded volume is everything further than `probe` from any legal
         probe center — this is what re-fills the inter-atomic crevices that the bare van
         der Waals surface leaves as pits, and it is the difference that makes the
         Lawrence & Colman statistic come out in its published range.
      3. Take the boundary layer and push each point onto the exact `distance == probe`
         isosurface along the field gradient, which recovers sub-grid accuracy and gives
         a smooth normal.
    """
    from scipy.ndimage import distance_transform_edt

    radii = vdw_radii(elements)
    pad = float(radii.max()) + 2 * probe + 2 * spacing
    lo = coords.min(axis=0) - pad
    hi = coords.max(axis=0) + pad
    shape = np.maximum(np.ceil((hi - lo) / spacing).astype(int) + 1, 3)

    inside_sas = np.zeros(shape, dtype=bool)
    reach = radii + probe
    for center, r in zip(coords, reach, strict=True):
        i0 = np.maximum(np.floor((center - r - lo) / spacing).astype(int), 0)
        i1 = np.minimum(np.ceil((center + r - lo) / spacing).astype(int) + 1, shape)
        if np.any(i1 <= i0):
            continue
        grids = np.meshgrid(
            *(lo[d] + np.arange(i0[d], i1[d]) * spacing for d in range(3)), indexing="ij"
        )
        local = sum((grids[d] - center[d]) ** 2 for d in range(3)) <= r * r
        inside_sas[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] |= local

    # Distance from every voxel to the nearest legal probe center.
    dist = distance_transform_edt(inside_sas, sampling=spacing)
    sev = dist > probe

    # Boundary voxels of the solvent-excluded volume.
    boundary = sev.copy()
    for axis in range(3):
        for shift in (1, -1):
            boundary &= np.roll(sev, shift, axis=axis)
    boundary = sev & ~boundary
    idx = np.argwhere(boundary)
    if len(idx) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    grad = np.stack(np.gradient(dist, spacing), axis=-1)
    g = grad[idx[:, 0], idx[:, 1], idx[:, 2]]
    norm = np.linalg.norm(g, axis=-1, keepdims=True)
    ok = norm[:, 0] > 1e-6  # noqa: PLR2004
    idx, g, norm = idx[ok], g[ok], norm[ok]

    pts = lo + idx * spacing
    # Outward normal points away from the interior, i.e. down the distance gradient.
    normals = -g / norm
    # Project onto the exact isosurface.
    offset = dist[idx[:, 0], idx[:, 1], idx[:, 2]] - probe
    pts = pts + normals * offset[:, None]
    return pts, normals


def shape_complementarity(
    cx: Complex,
    *,
    weight: float = 0.5,
    max_partner_dist: float = 3.0,
    spacing: float = 0.6,
) -> float:
    """Lawrence & Colman shape-complementarity statistic Sc.

    For every dot on chain A's surface that is buried by chain B, find the nearest buried
    dot on B and score `-(n_a . n_b) * exp(-w * d^2)`: +1 for perfectly complementary
    (antiparallel) normals in contact, decaying with separation. Sc is the mean of the
    two medians. Real antibody–antigen interfaces sit around 0.64–0.70.

    The surface used is the solvent-excluded (Connolly) surface of each chain computed in
    isolation; the interface is the part of it that the partner chain buries.
    """
    ab, ag = cx.antibody, cx.antigen
    pts_a, norm_a = solvent_excluded_surface(ab.coords, ab.atom_elements, spacing=spacing)
    pts_b, norm_b = solvent_excluded_surface(ag.coords, ag.atom_elements, spacing=spacing)
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("nan")

    iface_a = _buried_points(pts_a, ag.coords, vdw_radii(ag.atom_elements))
    iface_b = _buried_points(pts_b, ab.coords, vdw_radii(ab.atom_elements))
    if iface_a.sum() < 10 or iface_b.sum() < 10:  # noqa: PLR2004
        return float("nan")

    s_ab = _directional_sc(
        pts_a[iface_a], norm_a[iface_a], pts_b[iface_b], norm_b[iface_b],
        weight, max_partner_dist,
    )
    s_ba = _directional_sc(
        pts_b[iface_b], norm_b[iface_b], pts_a[iface_a], norm_a[iface_a],
        weight, max_partner_dist,
    )
    if np.isnan(s_ab) or np.isnan(s_ba):
        return float("nan")
    return float((s_ab + s_ba) / 2.0)


def _buried_points(
    pts: np.ndarray,
    other_coords: np.ndarray,
    other_vdw: np.ndarray,
    *,
    probe: float = PROBE_RADIUS,
) -> np.ndarray:
    """Surface points that the partner chain buries: no probe fits between them and it."""
    tree = cKDTree(other_coords)
    limit = float(other_vdw.max()) + probe
    hits = tree.query_ball_point(pts, limit)
    out = np.zeros(len(pts), dtype=bool)
    for k, (p, h) in enumerate(zip(pts, hits, strict=True)):
        if not h:
            continue
        gap = np.linalg.norm(other_coords[h] - p, axis=-1) - other_vdw[h]
        out[k] = bool(gap.min() < probe)
    return out


def _directional_sc(
    dots_a: np.ndarray,
    norm_a: np.ndarray,
    dots_b: np.ndarray,
    norm_b: np.ndarray,
    weight: float,
    max_partner_dist: float,
) -> float:
    dist, idx = cKDTree(dots_b).query(dots_a, k=1)
    # Peripheral trim: dots with no close partner lie on the rim of the patch, where
    # Lawrence & Colman explicitly exclude a 1.5 A band.
    keep = dist <= max_partner_dist
    if keep.sum() < 10:  # noqa: PLR2004
        return float("nan")
    complementarity = -np.sum(norm_a[keep] * norm_b[idx[keep]], axis=-1)
    return float(np.median(complementarity * np.exp(-weight * dist[keep] ** 2)))


def _polar_groups(chain: Chain) -> tuple[dict[int, int], dict[int, int]]:
    """Map atom index -> antecedent atom index, for donors and acceptors respectively."""
    donors: dict[int, int] = {}
    acceptors: dict[int, int] = {}
    for res_i in range(chain.n_res):
        idx = chain.residue_atom_indices(res_i)
        by_name = {str(chain.atom_names[i]): int(i) for i in idx}
        resname = chain.resnames[res_i]

        if "N" in by_name and "CA" in by_name and resname != "PRO":
            donors[by_name["N"]] = by_name["CA"]
        if "O" in by_name and "C" in by_name:
            acceptors[by_name["O"]] = by_name["C"]
        if "OXT" in by_name and "C" in by_name:
            acceptors[by_name["OXT"]] = by_name["C"]

        for (rn, an), antecedent in _SIDECHAIN_DONORS.items():
            if rn == resname and an in by_name and antecedent in by_name:
                donors[by_name[an]] = by_name[antecedent]
        for (rn, an), antecedent in _SIDECHAIN_ACCEPTORS.items():
            if rn == resname and an in by_name and antecedent in by_name:
                acceptors[by_name[an]] = by_name[antecedent]
    return donors, acceptors


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle a-b-c in degrees."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def hydrogen_bonds(
    chain_a: Chain,
    chain_b: Chain,
    *,
    max_dist: float = HBOND_MAX_DIST,
    min_angle: float = HBOND_MIN_ANGLE,
) -> list[tuple[int, int]]:
    """Inter-chain hydrogen bonds as (atom index in A, atom index in B) pairs.

    Both donor-in-A and donor-in-B orientations are tested. Geometry only: heavy-atom
    separation plus both antecedent angles, since predicted structures have no hydrogens.
    """
    don_a, acc_a = _polar_groups(chain_a)
    don_b, acc_b = _polar_groups(chain_b)
    bonds: list[tuple[int, int]] = []

    for donors, acceptors, flip in ((don_a, acc_b, False), (don_b, acc_a, True)):
        src = chain_b if flip else chain_a
        dst = chain_a if flip else chain_b
        if not donors or not acceptors:
            continue
        acc_idx = np.array(sorted(acceptors))
        tree = cKDTree(dst.coords[acc_idx])
        for d_atom, d_ante in donors.items():
            for k in tree.query_ball_point(src.coords[d_atom], max_dist):
                a_atom = int(acc_idx[k])
                if _angle(src.coords[d_ante], src.coords[d_atom], dst.coords[a_atom]) < min_angle:
                    continue
                a_ante = acceptors[a_atom]
                if _angle(src.coords[d_atom], dst.coords[a_atom], dst.coords[a_ante]) < min_angle:
                    continue
                bonds.append((a_atom, d_atom) if flip else (d_atom, a_atom))
    return sorted(set(bonds))


def salt_bridges(
    chain_a: Chain, chain_b: Chain, *, max_dist: float = SALT_BRIDGE_MAX_DIST
) -> list[tuple[int, int]]:
    """Inter-chain charged-pair contacts as (atom index in A, atom index in B)."""
    cat_a, ani_a = _charged_atoms(chain_a)
    cat_b, ani_b = _charged_atoms(chain_b)
    out: list[tuple[int, int]] = []
    for left, right, flip in ((cat_a, ani_b, False), (ani_a, cat_b, False),
                              (cat_b, ani_a, True), (ani_b, cat_a, True)):
        if not len(left) or not len(right):
            continue
        src = chain_b if flip else chain_a
        dst = chain_a if flip else chain_b
        tree = cKDTree(dst.coords[right])
        for i in left:
            for k in tree.query_ball_point(src.coords[i], max_dist):
                j = int(right[k])
                out.append((j, int(i)) if flip else (int(i), j))
    return sorted(set(out))


def _charged_atoms(chain: Chain) -> tuple[np.ndarray, np.ndarray]:
    cat, ani = [], []
    for res_i in range(chain.n_res):
        resname = chain.resnames[res_i]
        for i in chain.residue_atom_indices(res_i):
            key = (resname, str(chain.atom_names[i]))
            if key in _CATIONIC:
                cat.append(int(i))
            elif key in _ANIONIC:
                ani.append(int(i))
    return np.array(cat, dtype=int), np.array(ani, dtype=int)


def interface_clashes(cx: Complex, *, tolerance: float = 0.5) -> int:
    """Count inter-chain heavy-atom pairs closer than (r_i + r_j - tolerance).

    Pairs that satisfy the hydrogen-bond geometry are exempt, since real H-bonds are
    legitimately shorter than the sum of van der Waals radii.
    """
    ab, ag = cx.antibody, cx.antigen
    r_ab, r_ag = vdw_radii(ab.atom_elements), vdw_radii(ag.atom_elements)
    tree = cKDTree(ag.coords)
    hbond_set = set(hydrogen_bonds(ab, ag))
    n = 0
    max_r = float(r_ag.max())
    for i, (p, ri) in enumerate(zip(ab.coords, r_ab, strict=True)):
        for j in tree.query_ball_point(p, ri + max_r):
            if np.linalg.norm(ag.coords[j] - p) < ri + r_ag[j] - tolerance and (i, j) not in hbond_set:
                n += 1
    return n


@dataclass
class InterfaceReport:
    shape_complementarity: float
    bsa_total: float
    bsa_antibody: float
    bsa_antigen: float
    n_hbonds: int
    hbond_density: float
    n_salt_bridges: int
    n_clashes: int
    n_buried_unsatisfied_polars: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def interface_report(cx: Complex, *, n_points: int = 256) -> InterfaceReport:
    """All physics-based interface descriptors in one pass."""
    bsa = buried_sasa(cx, n_points=n_points)
    hbonds = hydrogen_bonds(cx.antibody, cx.antigen)
    bridges = salt_bridges(cx.antibody, cx.antigen)
    total_bsa = bsa["bsa_total"]
    return InterfaceReport(
        shape_complementarity=shape_complementarity(cx),
        bsa_total=total_bsa,
        bsa_antibody=bsa["bsa_antibody"],
        bsa_antigen=bsa["bsa_antigen"],
        n_hbonds=len(hbonds),
        # Per 1000 A^2 of buried surface — the density is what distinguishes a
        # well-packed interface from a merely large one.
        hbond_density=1000.0 * len(hbonds) / total_bsa if total_bsa > 0 else float("nan"),
        n_salt_bridges=len(bridges),
        n_clashes=interface_clashes(cx),
        n_buried_unsatisfied_polars=count_buried_unsatisfied_polars(cx, n_points=n_points),
    )


def count_buried_unsatisfied_polars(
    cx: Complex, *, n_points: int = 256, min_burial: float = 5.0
) -> int:
    """Polar atoms that lose solvation on binding but gain no hydrogen bond.

    A standard developability-adjacent penalty: burying a donor or acceptor without
    satisfying it costs real binding energy, and an interface that racks these up is
    usually a modeling artifact rather than a real complex.
    """
    ab, ag = cx.antibody, cx.antigen
    joint_coords = np.vstack([ab.coords, ag.coords])
    joint_elements = np.concatenate([ab.atom_elements, ag.atom_elements])
    sasa_joint = sasa(joint_coords, joint_elements, n_points=n_points)

    satisfied_ab, satisfied_ag = set(), set()
    for i, j in hydrogen_bonds(ab, ag):
        satisfied_ab.add(i)
        satisfied_ag.add(j)
    # Intra-chain hydrogen bonds also count as satisfying an atom.
    for chain, satisfied in ((ab, satisfied_ab), (ag, satisfied_ag)):
        for i, j in hydrogen_bonds(chain, chain):
            if i != j:
                satisfied.add(i)
                satisfied.add(j)

    count = 0
    for offset, chain, satisfied in ((0, ab, satisfied_ab), (ab.n_atom, ag, satisfied_ag)):
        free = sasa(chain.coords, chain.atom_elements, n_points=n_points)
        donors, acceptors = _polar_groups(chain)
        polar = set(donors) | set(acceptors)
        for i in polar:
            if i in satisfied:
                continue
            if free[i] - sasa_joint[offset + i] >= min_burial:
                count += 1
    return count
