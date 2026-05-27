
def find_atoms(mol, symbol):
    return [at for at in mol.atoms if at.symbol == symbol]

def find_xh_groups(mol, x, n_h, neighbour_symbol=None, return_indices=False):
    """[(X, [H, ...]), ...] for every X with exactly n_h H neighbours.

    neighbour_symbol: if set, keep only Xs also bonded to that symbol.
    return_indices:   return mol.atoms indices instead of Atom objects.
    """
    out = []
    for x_at in mol.atoms:
        if x_at.symbol != x:
            continue
        nbrs = [b.other_end(x_at) for b in x_at.bonds]
        hs   = [n for n in nbrs if n.symbol == 'H']
        if len(hs) != n_h:
            continue
        if neighbour_symbol is not None and not any(n.symbol == neighbour_symbol for n in nbrs):
            continue
        if return_indices:
            out.append((mol.atoms.index(x_at),
                        [mol.atoms.index(h) for h in hs]))
        else:
            out.append((x_at, hs))
    return out

def find_xh_bonds(mol, x, return_indices=False):
    """Flat [(X, H), ...] for every X-H bond where X has the given symbol."""
    out = []
    for x_at in mol.atoms:
        if x_at.symbol != x:
            continue
        for b in x_at.bonds:
            h = b.other_end(x_at)
            if h.symbol != 'H':
                continue
            if return_indices:
                out.append((mol.atoms.index(x_at), mol.atoms.index(h)))
            else:
                out.append((x_at, h))
    return out

def find_adjacent_xh_pairs(mol, x, n_h, return_indices=False):
    """[(X1, [H1,...], X2, [H2,...]), ...] for every adjacent X-X pair where
    both X atoms have exactly n_h hydrogens.
    """
    groups = find_xh_groups(mol, x, n_h)
    pairs = []
    for i, (c1, h1s) in enumerate(groups):
        nbrs1 = [b.other_end(c1) for b in c1.bonds]
        for j, (c2, h2s) in enumerate(groups):
            if j <= i or c2 not in nbrs1:
                continue
            pairs.append((c1, h1s, c2, h2s))
    if return_indices:
        return [(mol.atoms.index(c1), [mol.atoms.index(h) for h in h1s],
                 mol.atoms.index(c2), [mol.atoms.index(h) for h in h2s])
                for c1, h1s, c2, h2s in pairs]
    return pairs

def find_adjacent_xh_pair_anchored(mol, x, n_h, anchor_symbol):
    """Like find_adjacent_xh_pairs but returns the FIRST pair reordered so
    that the first X is bonded to an atom of anchor_symbol.

    Useful for choline's CH2-CH2 where the first CH2 is the one near N+.
    """
    for c1, h1s, c2, h2s in find_adjacent_xh_pairs(mol, x, n_h):
        nbrs1 = [b.other_end(c1) for b in c1.bonds]
        if any(a.symbol == anchor_symbol for a in nbrs1):
            return c1, h1s, c2, h2s
        return c2, h2s, c1, h1s
    return None

