"""
UDIM tiles: which part of UV space a triangle lives in.

Tile 1001 covers UVs 0 to 1. Each step of one in U adds 1 (up to 1010), each
step of one in V adds 10. A source mesh can spread its parts over several
tiles so that each part is sampled from its own texture.
"""

import os
import re

import numpy as np

from .errors import TransferError

FIRST = 1001
COLUMNS = 10           # tiles per row: U runs 0 to 10
NO_TILE = 0            # a triangle outside the UDIM range (U below 0 or above 10, V below 0)
STRADDLE_TOLERANCE = 1e-4


def udim(u_index, v_index):
    """UDIM number of the tile at column `u_index`, row `v_index` (0-based)."""
    return FIRST + u_index + COLUMNS * v_index


def offset(tile):
    """(u, v) of the lower left corner of a UDIM tile (or arrays of them for an array of tiles)."""
    i = np.asarray(tile) - FIRST
    return i % COLUMNS * 1.0, i // COLUMNS * 1.0


def triangle_tiles(tri_uv):
    """UDIM number of each triangle, from the tile its centre lies in.

    Args:
        tri_uv: (T, 3, 2) UVs of each triangle's corners.

    Returns:
        (T,) int64 array; `NO_TILE` where the centre is outside the UDIM range.
    """
    centre = tri_uv.mean(axis=1)
    with np.errstate(invalid="ignore"):
        cell = np.floor(centre)
    valid = np.isfinite(cell).all(axis=1) & (cell[:, 0] >= 0) & (cell[:, 0] < COLUMNS) & (cell[:, 1] >= 0)
    cell = np.where(valid[:, None], cell, 0).astype(np.int64)
    return np.where(valid, udim(cell[:, 0], cell[:, 1]), NO_TILE)


def straddling(tri_uv, tiles):
    """(T,) bool: triangles with a corner outside their own tile."""
    local = local_uv(tri_uv, tiles)
    outside = ((local < -STRADDLE_TOLERANCE) | (local > 1 + STRADDLE_TOLERANCE)).any(axis=(1, 2))
    return outside & (tiles != NO_TILE)


def local_uv(tri_uv, tiles):
    """Triangle UVs moved into 0 to 1 by subtracting each triangle's tile offset."""
    lo = np.zeros((len(tiles), 2))
    ok = tiles != NO_TILE
    lo[ok] = np.column_stack(offset(tiles[ok]))
    return tri_uv - lo[:, None, :]


def describe(tiles):
    """Readable list of tile numbers, e.g. "1001, 1002 and 1003"."""
    names = [str(t) for t in tiles]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def tile_number(tile):
    """A UDIM tile number as an int (1001 or more), from an int or text such as "1002".

    Raises:
        TransferError: it isn't one.
    """
    text = str(tile).strip()
    if isinstance(tile, bool) or not text.isdigit() or int(text) < FIRST:
        raise TransferError(f"{tile!r} isn't a UDIM tile number (1001, 1002 and so on)")
    return int(text)


_UDIM_NAME = re.compile(r"^(?P<stem>.+)\.(?P<tile>\d{4})(?P<ext>\.[A-Za-z0-9]+)$")


def split_name(path):
    """(name without the tile, tile) for a UDIM file name such as "Body_Color.1002.png", else None.

    Only the usual "name.1001.ext" form counts, so names like "Rock_2048.png"
    aren't mistaken for tiles.
    """
    m = _UDIM_NAME.match(os.path.basename(path))
    if not m or int(m["tile"]) < FIRST:
        return None
    return m["stem"] + m["ext"], int(m["tile"])
