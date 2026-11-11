"""The texture list's data: one `TextureRow` per output, with an explicit status."""

import os
from dataclasses import dataclass
from enum import Enum

from ..transfer import TiledMap, output_name

PLAIN = 0     # tile key of a plain row's one texture (used on the whole mesh)


class Status(Enum):
    """Where a texture is in its life cycle. The value is the text shown in the table."""

    READY = "Ready"
    WAITING = "Waiting"
    RUNNING = "Working"
    DONE = "✓ Done"
    FAILED = "✗ Failed (hover for details)"
    CANCELLED = "Cancelled"
    SKIPPED = "Skipped"
    STALE = "Type changed: transfer again"
    CHANGED = "File changed: transfer again"
    EDITED = "Changed: transfer again"
    MISSING = "✗ File missing"

    @property
    def colour(self):
        """Progress-bar fill colour for this status."""
        return {Status.DONE: "#2e8b57", Status.FAILED: "#a33", Status.MISSING: "#a33",
                Status.RUNNING: "#2f6fbf"}.get(self, "#555")


@dataclass
class TileTexture:
    """One input texture: its file, size, and modification time when last read.

    `mtime` tells whether the file changed since it was added, reloaded or
    transferred.
    """

    path: str
    width: int
    height: int
    high_precision: bool
    mtime: float = 0.0


@dataclass
class TextureRow:
    """One output in the list, plus the widgets that show it.

    A plain row transfers one texture over the whole mesh and saves it under
    the texture's own name: `textures` is {PLAIN: texture}. A tiled row puts
    one output together from a texture per source UDIM tile ({tile: texture})
    and saves it as `name`; source parts in tiles without a texture get `fill`.

    Attributes:
        id: stable identifier (table row numbers change when rows are removed).
        textures: {tile or PLAIN: TileTexture}.
        type_combo: the row's texture-type dropdown (its data is the kind).
        status_cell: the row's `StatusCell`.
        name: output file name of a tiled row.
        fill: fill colour of a tiled row (see `TiledMap.fill`), or None.
        fill_button: the row's fill widget (tiled rows only).
        status: current `Status`.
        result_path: where this session's transfer saved the output, if it did.
        stage: latest progress message while running (e.g. "Padding edges").
    """

    id: int
    textures: dict
    type_combo: object = None
    status_cell: object = None
    name: str = ""
    fill: tuple = None
    fill_button: object = None
    status: Status = Status.READY
    result_path: str = ""
    stage: str = ""
    kind_hint: str = ""                    # kind to show while the type dropdown is being rebuilt

    @property
    def tiled(self):
        return PLAIN not in self.textures

    @property
    def kind(self):
        return self.type_combo.currentData() if self.type_combo is not None else self.kind_hint

    @property
    def first(self):
        """The texture of a plain row, or the lowest tile's texture of a tiled row."""
        return self.textures[min(self.textures)]

    @property
    def path(self):
        return self.first.path

    @property
    def high_precision(self):
        return any(t.high_precision for t in self.textures.values())

    @property
    def largest(self):
        """The largest texture (by pixel count): it sets the output size."""
        return max(self.textures.values(), key=lambda t: t.width * t.height)

    @property
    def width(self):
        return self.largest.width

    @property
    def height(self):
        return self.largest.height

    @property
    def label(self):
        """How the row is named in messages: the output name, or the plain texture's file name."""
        return self.name if self.tiled else os.path.basename(self.path)

    def output_name(self, sixteen_bit_option):
        """File name this row is saved under with the given 16-bit setting."""
        sixteen = sixteen_bit_option or self.high_precision
        return output_name(self.name if self.tiled else self.path, sixteen)

    def to_map(self):
        """This tiled row as a `TiledMap`."""
        return TiledMap(self.name, {tile: t.path for tile, t in sorted(self.textures.items())},
                        self.kind, self.fill)

    def is_plain_equivalent(self):
        """True for a tiled row that a plain row holds without loss: one texture, its own name, no fill."""
        return (len(self.textures) == 1 and self.fill is None
                and self.name == os.path.basename(self.first.path))
