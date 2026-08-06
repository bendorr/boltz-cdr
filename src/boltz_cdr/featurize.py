"""Boltz-2-compatible feature dicts built from a plain structure.

Boltz's real featurizer needs MSAs, CCD lookups and the full data pipeline. The guidance
potential, however, only consumes four keys — `asym_id`, `residue_index`, `atom_to_token`,
`atom_pad_mask` (plus `token_pad_mask`) — so we can build an equivalent dict directly from
a crystal structure and exercise the entire forward/backward path on real coordinates
without loading a single model weight.

That is what makes the gradient demonstration runnable on a laptop, and it is also how the
mask-translation logic in `masks.py` is unit-tested: author residue numbering (8QF4's
antigen starts at 211, not 0) deliberately exercises the within-chain ranking that the
real features also require.
"""

from __future__ import annotations

import numpy as np
import torch

from boltz_cdr.pdb_io import Complex


def mock_feats_from_complex(
    cx: Complex, *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Build a minimal Boltz-2-style feature dict from a two-chain complex.

    Chain ordinals follow the order (antibody, antigen), matching the order in which the
    run scripts write chains into the Boltz input YAML.
    """
    chains = [cx.antibody, cx.antigen]

    asym_id, residue_index = [], []
    atom_token, coords = [], []
    token_offset = 0
    for chain_i, chain in enumerate(chains):
        asym_id.extend([chain_i] * chain.n_res)
        # Author numbering on purpose: the mask code must not assume 0-based residues.
        residue_index.extend(chain.resnums.tolist())
        atom_token.extend((token_offset + chain.atom_res_index).tolist())
        coords.append(chain.coords)
        token_offset += chain.n_res

    n_token = token_offset
    n_atom = len(atom_token)

    atom_to_token = torch.zeros(1, n_atom, n_token, device=device, dtype=dtype)
    atom_to_token[0, torch.arange(n_atom), torch.as_tensor(atom_token)] = 1.0

    return {
        "asym_id": torch.as_tensor(asym_id, device=device).long().unsqueeze(0),
        "residue_index": torch.as_tensor(residue_index, device=device).long().unsqueeze(0),
        "token_pad_mask": torch.ones(1, n_token, device=device, dtype=torch.bool),
        "atom_pad_mask": torch.ones(1, n_atom, device=device, dtype=torch.bool),
        "atom_to_token": atom_to_token,
        "coords": torch.as_tensor(
            np.vstack(coords), device=device, dtype=dtype
        ).unsqueeze(0),
    }


def chain_atom_slices(cx: Complex) -> dict[str, slice]:
    """Where each chain's atoms live in the concatenated coordinate tensor."""
    n_ab = cx.antibody.n_atom
    return {"antibody": slice(0, n_ab), "antigen": slice(n_ab, n_ab + cx.antigen.n_atom)}


def coords_to_complex(coords: torch.Tensor, template: Complex) -> Complex:
    """Write a (n_atom, 3) coordinate tensor back into a Complex with `template`'s topology.

    Used to score perturbed or guidance-updated coordinates with the same metrics that
    evaluate real predictions.
    """
    import copy

    xyz = coords.detach().cpu().numpy().reshape(-1, 3)
    slices = chain_atom_slices(template)
    out = copy.deepcopy(template)
    out.antibody.coords = xyz[slices["antibody"]].astype(float)
    out.antigen.coords = xyz[slices["antigen"]].astype(float)
    return out
