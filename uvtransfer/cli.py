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
"""

import argparse
import os
import sys

from .errors import TransferError
from .images import image_info
from .setupfile import EXTENSION, Setup, TextureEntry
from .system import available_memory
from .transfer import KIND_LABELS, KINDS, Options, Transfer, classify, estimate_memory

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
                    help="force a texture type (normal, srgb or linear); repeatable")
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
        if args.source or args.target or args.textures:
            ap.error("give either --setup or the FBX files and textures, not both")
        setup = Setup.load(args.setup)
        for entry in setup.textures:              # --as also applies to a setup's textures
            stem = os.path.splitext(os.path.basename(entry.path))[0]
            entry.kind = overrides.get(stem, overrides.get(os.path.basename(entry.path), entry.kind))
    else:
        if not (args.source and args.target and args.textures):
            ap.error("give the source FBX, the target FBX and at least one texture (or --setup FILE)")
        setup = Setup(os.path.abspath(args.source), os.path.abspath(args.target),
                      [TextureEntry(os.path.abspath(t), classify(t, overrides)) for t in args.textures])

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
    return 1 if failed else 0


def _transfer_one(xfer, path, out_dir, kind, opts):
    """Transfer and report one texture. Returns True on success."""
    print(f"{os.path.basename(path)}: treated as {KIND_LABELS[kind].lower()}")
    _warn_memory(path, kind, opts)
    live = sys.stdout.isatty()

    def show(fraction, message):
        if live:
            sys.stdout.write(f"\r  {100 * fraction:5.1f}%  {(message or ''):<30}")
            sys.stdout.flush()

    try:
        result = xfer.transfer_file(path, out_dir, kind, opts, progress=show)
    except TransferError as e:
        print(("\r" if live else "") + f"  error: {e}")
        return False
    if live:
        print("\r" + " " * 45 + "\r", end="")
    for note in result.notes:
        print(f"  {note}")
    print(f"  -> {result.path}  (UV coverage {100 * result.coverage:.1f}%)")
    return True


def _warn_memory(path, kind, opts):
    try:
        width, height, _ = image_info(path)
    except OSError:
        return                                    # transfer_file reports unreadable files
    need, free = estimate_memory(width, height, 4, kind, opts), available_memory()
    if free and need > 0.8 * free:
        print(f"  warning: needs about {need / 2**30:.1f} GB of memory; "
              f"{free / 2**30:.1f} GB is free")
