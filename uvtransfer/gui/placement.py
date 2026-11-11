"""Where textures added to the tiled grid go: which row (output) and which tile.

Kept free of Qt so the rules can be tested on their own.

- A UDIM-named file ("Body_Color.1002.png") goes to its own tile, in the row
  named after it without the number ("Body_Color.png").
- Any other file goes to the tile it was dropped on, in the first row that
  holds the same kind of texture (colour, normal, or the same last word such
  as "SRMF") and has that tile free. Dropping Head_Basecolor, Head_Normal and
  Head_SRMF on tile 1001, then Teeth_Color and Teeth_Normal on tile 1002,
  gives three rows: colour, normal and SRMF.
"""

import os
import re
from dataclasses import dataclass, field

from .. import tiles as udims
from ..transfer import classify

_WORDS = re.compile(r"[^a-z0-9]+")


def channel(path):
    """What a texture holds, judged from its file name: "colour", "normal" or its last word.

    "Head_Basecolor.png" and "Teeth_Color.png" are both "colour";
    "Head_SRMF.png" is "srmf".
    """
    name = (udims.split_name(path) or (os.path.basename(path),))[0]
    kind = classify(name)
    words = [w for w in _WORDS.split(os.path.splitext(name)[0].lower()) if w]
    if kind == "normal":
        return "normal"
    if kind == "srgb":
        return "emissive" if any("emissi" in w for w in words) else "colour"
    return words[-1] if words else ""


@dataclass
class Slot:
    """An output row as placement sees it: its name and {tile: texture path}."""

    name: str
    tiles: dict = field(default_factory=dict)

    @property
    def channel(self):
        return channel(self.tiles[min(self.tiles)]) if self.tiles else ""


def place(slots, paths, tile):
    """Put `paths` into `slots` (changed in place; new slots are appended).

    Args:
        slots: list of `Slot` for the rows already in the grid, in order.
        paths: files to add.
        tile: the tile the files were dropped on, for files without a tile number.

    Returns:
        [(slot index, tile, path)] for every file placed. A file whose cell
        is already taken in a matching row starts a new row instead, so
        nothing is replaced.
    """
    placed = []
    for path in paths:
        split = udims.split_name(path)
        if split is not None:
            name, file_tile = split
            index = next((i for i, s in enumerate(slots) if s.name.lower() == name.lower()
                          and file_tile not in s.tiles), None)
        else:
            name, file_tile = os.path.basename(path), tile
            wanted = channel(path)
            index = next((i for i, s in enumerate(slots) if s.channel == wanted and file_tile not in s.tiles),
                         None)
        if index is None:
            slots.append(Slot(_free_name(name, slots)))
            index = len(slots) - 1
        slots[index].tiles[file_tile] = path
        placed.append((index, file_tile, path))
    return placed


def _free_name(name, slots):
    """`name`, or "name 2.png", "name 3.png" ... if a row already uses it."""
    taken = {s.name.lower() for s in slots}
    stem, ext = os.path.splitext(name)
    candidate, n = name, 1
    while candidate.lower() in taken:
        n += 1
        candidate = f"{stem} {n}{ext}"
    return candidate
