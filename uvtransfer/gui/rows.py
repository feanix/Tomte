"""The texture list's data: one `TextureRow` per texture, with an explicit status."""

from dataclasses import dataclass
from enum import Enum

from ..transfer import output_name


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
    MISSING = "✗ File missing"

    @property
    def colour(self):
        """Progress-bar fill colour for this status."""
        return {Status.DONE: "#2e8b57", Status.FAILED: "#a33", Status.MISSING: "#a33",
                Status.RUNNING: "#2f6fbf"}.get(self, "#555")


@dataclass
class TextureRow:
    """One texture in the list, plus the widgets that show it.

    Attributes:
        id: stable identifier (table row numbers change when rows are removed).
        path: input texture.
        width, height: input size in pixels.
        high_precision: input has more than 8 bits per channel.
        type_combo: the row's texture-type dropdown (its data is the kind).
        status_cell: the row's `StatusCell`.
        status: current `Status`.
        result_path: where this session's transfer saved the texture, if it did.
        stage: latest progress message while running (e.g. "Padding edges").
        mtime: the file's modification time when it was last read (added,
            reloaded or transferred), to tell whether it changed since.
    """

    id: int
    path: str
    width: int
    height: int
    high_precision: bool
    type_combo: object
    status_cell: object
    status: Status = Status.READY
    result_path: str = ""
    stage: str = ""
    mtime: float = 0.0

    @property
    def kind(self):
        return self.type_combo.currentData()

    def output_name(self, sixteen_bit_option):
        """File name this texture is saved under with the given 16-bit setting."""
        return output_name(self.path, sixteen_bit_option or self.high_precision)
