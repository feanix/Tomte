"""
Command line:

    python -m uvtransfer SOURCE.fbx TARGET.fbx tex1.png [tex2.png ...] [options]
    python -m uvtransfer --setup Body.tomte [options]

SOURCE.fbx is the mesh whose UVs the textures were made for; TARGET.fbx is
the same mesh with the new UVs. Texture types are guessed from file names
("normal"/"nrm" -> normal map, "basecolor"/"albedo"/"diffuse"/"color"/
"emissive" -> sRGB colour, anything else -> linear data); override with
--as NAME=normal|srgb|linear.

--setup runs a setup file saved by the app (or by --save-setup). Options
given on the command line override the file's. --save-setup writes the
current meshes, textures and options to a setup file instead of running.

UDIM tiles: when SOURCE.fbx spreads its parts over UDIM tiles, each output
can be put together from one texture per tile. Textures named the UDIM way
("Body_Color.1001.png", "Body_Color.1002.png") become one output,
"Body_Color.png". For textures named otherwise, give each output with --map:

    --map Body_Color.png 1001=Head_Color.png 1002=Teeth_Color.png
    --map Body_ORM.png 1001=Head_ORM.png fill=1,0.5,0

Parts whose tile has no texture get the fill colour (0 to 1 per channel),
or stay empty without one. Normal maps always get a flat normal there.
"""

import argparse
import os
import sys

from . import tiles as udims
from .errors import TransferError
from .images import image_info
from .setupfile import EXTENSION, Setup, TextureEntry
from .system import available_memory
from .transfer import (KIND_LABELS, KINDS, Options, TiledMap, Transfer, check_fill, check_map_name, classify,
                       estimate_memory)

DEFAULTS = Options()


def build_parser():
    """Argument parser for the command line. Option defaults are None so that
    options left out don't override a setup file's."""
    ap = argparse.ArgumentParser(
        prog="python -m uvtransfer", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="FBX the textures were made for")
    ap.add_argument("target", nargs="?", help="the same mesh with the new UVs")
    ap.add_argument("textures", nargs="*", help="textures to transfer")
    ap.add_argument("--setup", metavar="FILE", help=f"run a saved setup file ({EXTENSION})")
    ap.add_argument("--save-setup", metavar="FILE",
                    help="save the meshes, textures and options to a setup file, then stop")
    ap.add_argument("-o", "--out", help="output folder (default: <target dir>/transferred)")
    ap.add_argument("--size", type=int,
                    help="longest side of the output in pixels (default: same as input)")
    ap.add_argument("--ss", type=int, help=f"sub-samples per pixel axis (default {DEFAULTS.supersample})")
    ap.add_argument("--pad", type=int,
                    help=f"edge padding in pixels (default {DEFAULTS.pad}; -1 fills all empty space)")
    ap.add_argument("--normal-convention", choices=["dx", "gl", "auto"],
                    help="normal map format (default auto; Unreal = dx)")
    ap.add_argument("--as", dest="as_", action="append", default=[], metavar="NAME=TYPE",
                    help="force a texture or map type (normal, srgb or linear); repeatable")
    ap.add_argument("--map", dest="maps", action="append", nargs="+", default=[],
                    metavar=("NAME", "TILE=FILE"),
                    help="put one output together from a texture per UDIM tile, with an optional "
                         "fill=COLOUR (see above); repeatable")
    ap.add_argument("--uv-set", type=int, help="UV channel to use (default 0)")
    ap.add_argument("--16bit", dest="sixteen", action="store_true", default=None,
                    help="write 16-bit PNGs (recommended for normal maps)")
    return ap


def main(argv=None):
    """Run the command line. Returns the process exit code (0 = all textures transferred)."""
    ap = build_parser()
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):       # don't crash on odd characters in old consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    try:
        setup = _setup_from_args(ap, args)
    except TransferError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.save_setup:
        try:
            setup.save(args.save_setup)
        except TransferError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"Saved {args.save_setup}")
        return 0
    return run_setup(setup)


def _setup_from_args(ap, args):
    """The `Setup` to run: a setup file with command-line overrides, or the arguments alone."""
    overrides = {}
    for item in args.as_:
        name, _, kind = item.partition("=")
        if kind not in KINDS:
            ap.error(f"--as {item}: type must be normal, srgb or linear")
        overrides[name] = kind

    if args.setup:
        if args.source or args.target or args.textures or args.maps:
            ap.error("give either --setup or the FBX files and textures, not both")
        setup = Setup.load(args.setup)
        for entry in setup.textures:              # --as also applies to a setup's textures and maps
            entry.kind = _overridden(entry.path, overrides, entry.kind)
        for tiled in setup.maps:
            tiled.kind = _overridden(tiled.name, overrides, tiled.kind)
    else:
        if not (args.source and args.target and (args.textures or args.maps)):
            ap.error("give the source FBX, the target FBX and at least one texture (or --setup FILE)")
        textures, maps = _group_udim_textures(ap, args.textures, overrides)
        maps += [_parse_map(ap, items, overrides) for items in args.maps]
        names = [m.name.lower() for m in maps]
        twice = sorted({n for n in names if names.count(n) > 1})
        if twice:
            ap.error(f"more than one map would be saved as {', '.join(twice)}")
        setup = Setup(os.path.abspath(args.source), os.path.abspath(args.target), textures, maps)

    given = {"size": args.size, "supersample": args.ss, "pad": args.pad,
             "normal_convention": args.normal_convention, "sixteen_bit": args.sixteen}
    for name, value in given.items():
        if value is not None:
            setattr(setup.options, name, value)
    if args.uv_set is not None:
        setup.uv_set = args.uv_set
    if args.out:
        setup.output = os.path.abspath(args.out)
    try:
        setup.options.validate()
    except TransferError as e:
        ap.error(str(e))
    return setup


def _overridden(path, overrides, kind):
    """`kind`, unless --as named this file (by name or by name without extension)."""
    name = os.path.basename(path)
    return overrides.get(os.path.splitext(name)[0], overrides.get(name, kind))


def _group_udim_textures(ap, paths, overrides):
    """(plain `TextureEntry`s, `TiledMap`s): files named "name.1001.ext" are grouped into maps."""
    textures, maps = [], {}
    for path in paths:
        path = os.path.abspath(path)
        split = udims.split_name(path)
        if split is None:
            textures.append(TextureEntry(path, classify(path, overrides)))
            continue
        name, tile = split
        tiled = maps.setdefault(os.path.normcase(os.path.join(os.path.dirname(path), name)),
                                TiledMap(name, {}, classify(name, overrides)))
        if tile in tiled.tiles:
            ap.error(f"two textures for tile {tile} of {name}")
        tiled.tiles[tile] = path
    for tiled in maps.values():
        tiled.tiles = dict(sorted(tiled.tiles.items()))
    return textures, list(maps.values())


def _parse_map(ap, items, overrides):
    """A `TiledMap` from --map NAME TILE=FILE [TILE=FILE ...] [fill=COLOUR]."""
    name, rest = items[0], items[1:]
    try:
        check_map_name(name)
    except TransferError as e:
        ap.error(f"--map: {e}")
    tiles, fill = {}, None
    for item in rest:
        key, sep, value = item.partition("=")
        if not sep or not value:
            ap.error(f"--map {name}: {item!r} should be TILE=FILE or fill=COLOUR")
        if key.strip().lower() == "fill":
            try:
                fill = check_fill([float(c) for c in value.split(",")])
            except (ValueError, TransferError):
                ap.error(f"--map {name}: the fill should be 1 to 4 numbers from 0 to 1, "
                         "such as fill=1,0.5,0")
            continue
        try:
            tile = udims.tile_number(key)
        except TransferError as e:
            ap.error(f"--map {name}: {e}")
        if tile in tiles:
            ap.error(f"--map {name}: tile {tile} is given twice")
        tiles[tile] = os.path.abspath(value)
    if not tiles:
        ap.error(f"--map {name}: give at least one texture as TILE=FILE, such as 1001=Head_Color.png")
    tiles = dict(sorted(tiles.items()))
    kind = _overridden(name, overrides, classify(next(iter(tiles.values())), overrides))
    return TiledMap(name, tiles, kind, fill)


def run_setup(setup):
    """Transfer every texture in `setup`, printing progress. Returns the exit code."""
    try:
        print(f"Reading {setup.source}\nReading {setup.target}")
        xfer = Transfer.from_fbx(setup.source, setup.target, setup.uv_set)
    except TransferError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"  {xfer.summary()}")
    for w in xfer.warnings:
        print(f"  warning: {w}")
    out_dir = setup.output_dir()
    failed = sum(not _transfer_one(xfer, t.path, out_dir, t.kind, setup.options)
                 for t in setup.textures)
    failed += sum(not _transfer_map(xfer, m, out_dir, setup.options) for m in setup.maps)
    return 1 if failed else 0


def _transfer_one(xfer, path, out_dir, kind, opts):
    """Transfer and report one texture. Returns True on success."""
    print(f"{os.path.basename(path)}: treated as {KIND_LABELS[kind].lower()}")
    _warn_memory([path], kind, opts)
    return _run(lambda show: xfer.transfer_file(path, out_dir, kind, opts, progress=show))


def _transfer_map(xfer, tiled, out_dir, opts):
    """Put together, save and report one tiled map. Returns True on success."""
    kind = tiled.kind or classify(next(iter(tiled.tiles.values()), ""))
    parts = ", ".join(f"{tile} {os.path.basename(path)}" for tile, path in tiled.tiles.items())
    print(f"{tiled.name}: {KIND_LABELS[kind].lower()} from tile {parts}")
    if kind == "normal":
        print("  parts with no texture: flat")
    elif tiled.fill is not None:
        print(f"  parts with no texture: fill {', '.join(f'{c:g}' for c in tiled.fill)}")
    _warn_memory(list(tiled.tiles.values()), kind, opts)
    tiled = TiledMap(tiled.name, tiled.tiles, kind, tiled.fill)
    return _run(lambda show: xfer.transfer_map(tiled, out_dir, opts, progress=show))


def _run(transfer):
    """Run transfer(progress) with a live progress line, and report the result. Returns True on success."""
    live = sys.stdout.isatty()

    def show(fraction, message):
        if live:
            sys.stdout.write(f"\r  {100 * fraction:5.1f}%  {(message or ''):<30}")
            sys.stdout.flush()

    try:
        result = transfer(show)
    except TransferError as e:
        print(("\r" if live else "") + f"  error: {e}")
        return False
    if live:
        print("\r" + " " * 45 + "\r", end="")
    for note in result.notes:
        print(f"  {note}")
    print(f"  -> {result.path}  (UV coverage {100 * result.coverage:.1f}%)")
    return True


def _warn_memory(paths, kind, opts):
    """Warn if the largest of `paths` needs most of the free memory (one is loaded at a time)."""
    sizes = []
    for path in paths:
        try:
            sizes.append(image_info(path)[:2])
        except OSError:
            pass                                  # the transfer reports unreadable files
    if not sizes:
        return
    width, height = max(sizes, key=lambda size: size[0] * size[1])
    need, free = estimate_memory(width, height, 4, kind, opts), available_memory()
    if free and need > 0.8 * free:
        print(f"  warning: needs about {need / 2**30:.1f} GB of memory; "
              f"{free / 2**30:.1f} GB is free")
