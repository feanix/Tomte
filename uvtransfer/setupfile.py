"""
Setup files: a saved transfer (both meshes, the textures and their types, and
the output settings) that can be reopened in the app or run from the command
line, for bakes that are redone regularly:

    python -m uvtransfer --setup Body.tomte

A setup file is JSON. Paths are stored relative to the file's own folder
when possible, so a project folder can be moved or shared:

    {
      "tomte_setup": 1,
      "source": "meshes/Body_OldUVs.fbx",
      "target": "meshes/Body_NewUVs.fbx",
      "output": "transferred",
      "uv_channel": 0,
      "textures": [
        {"path": "textures/Body_Basecolor.png", "type": "srgb"},
        {"path": "textures/Body_Normal.png", "type": "normal"}
      ],
      "options": {"size": 0, "supersample": 2, "pad": 16,
                  "normal_convention": "auto", "sixteen_bit": false}
    }

"output" can be null: results then go to "transferred" next to the target
FBX. "type" can be left out to guess it from the file name. Missing options
take their defaults.

When the source mesh spreads its parts over UDIM tiles, "maps" puts each
output texture together from one texture per tile (see `TiledMap`). "fill"
is the colour for tiles with no texture, or null to leave them empty:

    {
      "tomte_setup": 2,
      ...
      "maps": [
        {"name": "Body_Basecolor.png", "type": "srgb",
         "tiles": {"1001": "textures/Head_Basecolor.png", "1002": "textures/Teeth_Color.png"}},
        {"name": "Body_ORM.png", "type": "linear",
         "tiles": {"1001": "textures/Head_ORM.png"}, "fill": [1.0, 0.5, 0.0]}
      ]
    }

Files without "maps" are saved as format 1, which Tomte 1.0 can open.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field, fields

from . import tiles as udims
from .errors import TransferError
from .transfer import KINDS, Options, TiledMap, check_fill, check_map_name, classify, default_output_dir

EXTENSION = ".tomte"
FORMAT_VERSION = 2          # newest format this version reads and writes
PLAIN_FORMAT = 1            # written when there are no tiled maps, so older versions can open it
VERSION_KEY = "tomte_setup"
MAP_KEYS = {"name", "type", "tiles", "fill"}
SAVED_OPTIONS = ("size", "supersample", "pad", "normal_convention", "sixteen_bit")   # not threads


@dataclass
class TextureEntry:
    """One texture in a setup: its path and type ("srgb", "normal" or "linear")."""

    path: str
    kind: str


@dataclass
class Setup:
    """Everything needed to repeat a transfer.

    Attributes:
        source, target: FBX files (absolute paths once loaded).
        textures: list of `TextureEntry`, in order.
        maps: list of `TiledMap` (one texture per source UDIM tile), in order.
            Their kinds are always set once loaded.
        output: output folder, or "" for "transferred" next to the target.
        uv_set: UV channel used on both meshes.
        options: transfer `Options` (threads is never saved).
    """

    source: str
    target: str
    textures: list = field(default_factory=list)
    maps: list = field(default_factory=list)
    output: str = ""
    uv_set: int = 0
    options: Options = field(default_factory=Options)

    def output_dir(self):
        """Where the results go."""
        return self.output or default_output_dir(self.target)

    # -- saving -------------------------------------------------------------------

    def save(self, path):
        """Write the setup to `path`, with paths relative to its folder where possible.

        Raises:
            TransferError: the file can't be written.
        """
        base = os.path.dirname(os.path.abspath(path))
        data = {
            VERSION_KEY: FORMAT_VERSION if self.maps else PLAIN_FORMAT,
            "source": _portable(self.source, base),
            "target": _portable(self.target, base),
            "output": _portable(self.output, base) if self.output else None,
            "uv_channel": self.uv_set,
            "textures": [{"path": _portable(t.path, base), "type": t.kind} for t in self.textures],
            "options": {name: getattr(self.options, name) for name in SAVED_OPTIONS},
        }
        if self.maps:
            data["maps"] = [_map_data(m, base) for m in self.maps]
        text = json.dumps(data, indent=2) + "\n"
        try:
            _write_atomically(path, text)
        except OSError as e:
            raise TransferError(f"Couldn't save {os.path.basename(path)}: {e.strerror or e}") from e

    # -- loading ------------------------------------------------------------------

    @classmethod
    def load(cls, path):
        """Read a setup file. Paths in the result are absolute.

        Raises:
            TransferError: the file can't be read, isn't a setup file, or
                holds invalid settings. The message names the problem.
        """
        label = os.path.basename(path)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except OSError as e:
            raise TransferError(f"Couldn't open {label}: {e.strerror or e}") from e
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise TransferError(f"{label} isn't a valid setup file ({e})") from e
        if not isinstance(data, dict) or VERSION_KEY not in data:
            raise TransferError(f"{label} isn't a Tomte setup file")
        version = data[VERSION_KEY]
        if isinstance(version, bool) or not isinstance(version, int) \
                or not PLAIN_FORMAT <= version <= FORMAT_VERSION:
            raise TransferError(f"{label} was made by a newer version of Tomte "
                                f"(setup format {version!r})")
        base = os.path.dirname(os.path.abspath(path))
        reader = _Reader(data, label, base)
        options = reader.options()
        setup = cls(source=reader.path("source"), target=reader.path("target"),
                    textures=reader.textures(), maps=reader.maps(),
                    output=reader.path("output", required=False),
                    uv_set=reader.integer("uv_channel", 0, minimum=0), options=options)
        reader.reject_unknown_keys()
        return setup


class _Reader:
    """Checks the values of a parsed setup file, naming the bad key in each error."""

    TOP_KEYS = {VERSION_KEY, "source", "target", "output", "uv_channel", "textures", "maps", "options"}

    def __init__(self, data, label, base):
        self.data, self.label, self.base = data, label, base

    def fail(self, problem):
        raise TransferError(f"{self.label}: {problem}")

    def path(self, key, required=True):
        value = self.data.get(key)
        if value in (None, "") and not required:
            return ""
        if not isinstance(value, str) or not value:
            self.fail(f'"{key}" must be a file path')
        return self.resolve(value)

    def resolve(self, value):
        return os.path.normpath(os.path.join(self.base, os.path.expanduser(value)))

    def integer(self, key, default, minimum=None, source=None):
        value = (self.data if source is None else source).get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            self.fail(f'"{key}" must be a whole number, not {value!r}')
        if minimum is not None and value < minimum:
            self.fail(f'"{key}" must be {minimum} or more, not {value}')
        return value

    def textures(self):
        items = self.data.get("textures", [])
        if not isinstance(items, list):
            self.fail('"textures" must be a list')
        entries = []
        for i, item in enumerate(items):
            if isinstance(item, str):                  # a bare path: guess the type
                item = {"path": item}
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
                self.fail(f"texture {i + 1} needs a \"path\"")
            path = self.resolve(item["path"])
            kind = item.get("type") or classify(path)
            if kind not in KINDS:
                self.fail(f"texture {i + 1} has type {kind!r}; use one of {', '.join(KINDS)}")
            entries.append(TextureEntry(path, kind))
        return entries

    def maps(self):
        items = self.data.get("maps", [])
        if not isinstance(items, list):
            self.fail('"maps" must be a list')
        maps, names = [], set()
        for i, item in enumerate(items):
            where = f"map {i + 1}"
            if not isinstance(item, dict):
                self.fail(f"{where} must be a set of named values")
            unknown = sorted(set(item) - MAP_KEYS)
            if unknown:
                self.fail(f"{where} has unknown setting(s) {', '.join(map(repr, unknown))}")
            name = item.get("name")
            try:
                check_map_name(name)
            except TransferError as e:
                self.fail(f"{where}: {e}")
            where = f"map {name!r}"
            if name.lower() in names:
                self.fail(f"two maps are named {name!r}; each needs its own output name")
            names.add(name.lower())
            tiles = item.get("tiles")
            if not isinstance(tiles, dict) or not tiles:
                self.fail(f'{where} needs "tiles": tile numbers with a texture each')
            resolved = {}
            for tile, path in tiles.items():
                try:
                    number = udims.tile_number(tile)
                except TransferError as e:
                    self.fail(f"{where}: {e}")
                if not isinstance(path, str) or not path:
                    self.fail(f"{where}: tile {tile} must be a file path")
                resolved[number] = self.resolve(path)
            resolved = dict(sorted(resolved.items()))
            kind = item.get("type") or classify(next(iter(resolved.values())))
            if kind not in KINDS:
                self.fail(f"{where} has type {kind!r}; use one of {', '.join(KINDS)}")
            try:
                fill = check_fill(item.get("fill"))
            except TransferError as e:
                self.fail(f"{where}: {e}")
            maps.append(TiledMap(name, resolved, kind, fill))
        return maps

    def options(self):
        values = self.data.get("options", {})
        if not isinstance(values, dict):
            self.fail('"options" must be a set of named values')
        unknown = sorted(set(values) - set(SAVED_OPTIONS))
        if unknown:
            self.fail(f"unknown option(s) {', '.join(map(repr, unknown))}; "
                      f"known: {', '.join(SAVED_OPTIONS)}")
        defaults = Options()
        kwargs = {}
        for f in fields(Options):
            if f.name not in SAVED_OPTIONS:
                continue
            default = getattr(defaults, f.name)
            if isinstance(default, bool):
                value = values.get(f.name, default)
                if not isinstance(value, bool):
                    self.fail(f'option "{f.name}" must be true or false, not {value!r}')
            elif isinstance(default, int):
                value = self.integer(f.name, default, source=values)
            else:
                value = values.get(f.name, default)
                if not isinstance(value, str):
                    self.fail(f'option "{f.name}" must be text, not {value!r}')
            kwargs[f.name] = value
        options = Options(**kwargs)
        try:
            options.validate()
        except TransferError as e:
            self.fail(str(e))
        return options

    def reject_unknown_keys(self):
        unknown = sorted(set(self.data) - self.TOP_KEYS)
        if unknown:
            self.fail(f"unknown setting(s) {', '.join(map(repr, unknown))}")


def _map_data(tiled, base):
    """A `TiledMap` as JSON data, with paths relative to `base`."""
    data = {"name": tiled.name, "type": tiled.kind or classify(next(iter(tiled.tiles.values()), "")),
            "tiles": {str(tile): _portable(path, base)
                      for tile, path in sorted(tiled.tiles.items(), key=lambda item: udims.tile_number(item[0]))}}
    if tiled.fill is not None:
        data["fill"] = list(tiled.fill)
    return data


def _portable(path, base):
    """`path` relative to `base` with forward slashes, or absolute if on another drive."""
    path = os.path.abspath(path)
    try:
        rel = os.path.relpath(path, base)
    except ValueError:                             # different drive on Windows
        return path.replace(os.sep, "/")
    return rel.replace(os.sep, "/")


def _write_atomically(path, text):
    """Write via a temporary file so a failed save never leaves half a file behind."""
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tomte-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
