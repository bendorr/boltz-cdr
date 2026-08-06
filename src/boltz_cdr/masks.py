"""Lifting (chain, residue) selections into Boltz-2 atom masks.

Everything upstream in this project speaks in terms of "residue 97 of the antibody
chain". The diffusion sampler and the guidance potential speak in terms of atom indices
into a padded `(n_atom, 3)` coordinate tensor. This module is the single place where that
translation happens.

Two robustness choices:

  * Chains are addressed by their **ordinal** (0 = first chain in the input YAML), not by
    a raw `asym_id` integer, because the numeric value of `asym_id` is an internal
    detail of Boltz's tokenizer.
  * Residues are addressed by their **ordinal position within the chain**, derived by
    ranking that chain's unique `residue_index` values, which is correct whether Boltz
    numbers residues from 0, from 1, or globally across chains.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TokenIndex:
    """Per-token chain ordinal and within-chain residue position, for one example."""

    chain_ordinal: torch.Tensor  # (n_token,) long
    residue_position: torch.Tensor  # (n_token,) long
    n_chains: int


def build_token_index(feats: dict, *, batch: int = 0) -> TokenIndex:
    """Derive chain ordinals and within-chain residue positions from Boltz features."""
    asym = feats["asym_id"][batch].long()
    res = feats["residue_index"][batch].long()

    pad = feats.get("token_pad_mask")
    valid = pad[batch].bool() if pad is not None else torch.ones_like(asym, dtype=torch.bool)

    unique_chains = torch.unique(asym[valid], sorted=True)
    chain_ordinal = torch.full_like(asym, -1)
    residue_position = torch.full_like(res, -1)

    for ordinal, chain in enumerate(unique_chains.tolist()):
        in_chain = (asym == chain) & valid
        chain_ordinal[in_chain] = ordinal
        # Rank this chain's residue indices so position is 0-based and contiguous.
        unique_res = torch.unique(res[in_chain], sorted=True)
        lookup = torch.full((int(unique_res.max().item()) + 2,), -1, dtype=torch.long,
                            device=res.device)
        lookup[unique_res] = torch.arange(len(unique_res), device=res.device)
        residue_position[in_chain] = lookup[res[in_chain]]

    return TokenIndex(chain_ordinal, residue_position, len(unique_chains))


def token_mask(
    feats: dict,
    chain_ordinal: int,
    residue_positions=None,
    *,
    batch: int = 0,
    index: TokenIndex | None = None,
) -> torch.Tensor:
    """(n_token,) bool mask for a chain, optionally restricted to residue positions."""
    idx = index if index is not None else build_token_index(feats, batch=batch)
    mask = idx.chain_ordinal == chain_ordinal
    if residue_positions is not None:
        wanted = torch.as_tensor(
            list(residue_positions), dtype=torch.long, device=idx.residue_position.device
        )
        mask &= torch.isin(idx.residue_position, wanted)
    return mask


def atom_mask_from_tokens(
    feats: dict, tok_mask: torch.Tensor, *, batch: int = 0
) -> torch.Tensor:
    """Project a token mask onto atoms via `atom_to_token`.

    `atom_to_token` is a one-hot (n_atom, n_token) matrix, so a matrix-vector product with
    the token mask marks exactly the atoms belonging to selected tokens.
    """
    a2t = feats["atom_to_token"][batch]
    selected = (a2t.float() @ tok_mask.float()) > 0.5  # noqa: PLR2004
    pad = feats.get("atom_pad_mask")
    if pad is not None:
        selected &= pad[batch].bool()
    return selected


def atom_mask(
    feats: dict,
    chain_ordinal: int,
    residue_positions=None,
    *,
    batch: int = 0,
    index: TokenIndex | None = None,
) -> torch.Tensor:
    """(n_atom,) bool mask for a chain, optionally restricted to residue positions."""
    tok = token_mask(feats, chain_ordinal, residue_positions, batch=batch, index=index)
    return atom_mask_from_tokens(feats, tok, batch=batch)


def atom_residue_labels(
    feats: dict, *, batch: int = 0, index: TokenIndex | None = None
) -> torch.Tensor:
    """(n_atom,) long: a unique id per (chain, residue), for per-residue reductions.

    Padding atoms get -1.
    """
    idx = index if index is not None else build_token_index(feats, batch=batch)
    a2t = feats["atom_to_token"][batch]
    token_of_atom = a2t.float().argmax(dim=-1)

    # Compose chain and residue into one id; the multiplier just has to exceed the
    # largest within-chain residue position.
    stride = int(idx.residue_position.max().item()) + 2
    token_label = idx.chain_ordinal * stride + idx.residue_position
    token_label = torch.where(
        idx.chain_ordinal < 0, torch.full_like(token_label, -1), token_label
    )
    labels = token_label[token_of_atom]

    pad = feats.get("atom_pad_mask")
    if pad is not None:
        labels = torch.where(pad[batch].bool(), labels, torch.full_like(labels, -1))
    return labels


def describe(feats: dict, *, batch: int = 0) -> str:
    """Human-readable chain summary — used by the run scripts to verify the mapping."""
    idx = build_token_index(feats, batch=batch)
    parts = []
    for ordinal in range(idx.n_chains):
        n_tok = int((idx.chain_ordinal == ordinal).sum().item())
        n_atom = int(atom_mask(feats, ordinal, index=idx).sum().item())
        parts.append(f"chain[{ordinal}]: {n_tok} tokens, {n_atom} atoms")
    return "; ".join(parts)
