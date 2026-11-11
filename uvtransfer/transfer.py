"""
Moving textures from one UV layout to another.

For every sub-sample of the output texture, find the target triangle it lies
in, take the same barycentric point on the source triangle (same mesh, old
UVs), and sample the input texture there. Colour is averaged in linear light;
normal maps are converted into the target's tangent frames.

A source mesh can spread its parts over several UDIM tiles, each with its own
texture (`TiledMap`, `Transfer.transfer_map`): every target triangle is then
sampled from the texture of the tile its source triangle lies in.
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np

from . import normalmaps
from . import tiles as udims
from .errors import Cancelled, TransferError
from .images import (IMAGE_EXTENSIONS, image_info, linear_to_srgb, load_image, sample_bilinear, save_image,
                     srgb_to_linear)
from .mesh import Mesh, load_fbx_parts
from .padding import dilate
from .raster import BATCH_SAMPLES, TILE_RANGE, RasterPlan, coverage_counts, run_batches, sum_by_pixel
from .system import default_workers
from .tangents import corner_tangent_frames

KINDS = ("srgb", "normal", "linear")
KIND_LABELS = {"srgb": "Colour", "normal": "Normal map", "linear": "Data"}
CONVENTIONS = ("auto", "dx", "gl")
MAX_SIZE = 16384
MAX_SUPERSAMPLE = 8

OVERLAP_CHECK_SIZE = 1024   # resolution of the target-overlap check
OVERLAP_WARN = 0.005        # warn when more than 0.5% of the covered target area overlaps
POSITION_DRIFT_WARN = 1e-3  # warn when positions differ by more than 0.1% of the mesh size
_FLAT_NORMAL = (0.0, 0.0, 1.0)
_BATCH_BYTES_PER_SAMPLE = 300   # working memory per candidate sub-sample in a batch


@dataclass
class Options:
    """Settings for a transfer.

    Attributes:
        size: longest output side in pixels; 0 keeps each input's size.
        supersample: sub-samples per pixel along each axis (1-8). Higher is
            smoother where islands were shrunk, and slower.
        pad: edge padding in pixels; -1 fills all empty space.
        normal_convention: "auto", "dx" (DirectX/Unreal) or "gl" (OpenGL).
        sixteen_bit: always save 16-bit PNGs (high-precision inputs always are).
        threads: worker threads; 0 picks automatically.
    """

    size: int = 0
    supersample: int = 2
    pad: int = 16
    normal_convention: str = "auto"
    sixteen_bit: bool = False
    threads: int = 0

    def validate(self):
        """Raise TransferError listing any settings that are out of range."""
        problems = []
        if not 0 <= self.size <= MAX_SIZE:
            problems.append(f"size must be 0 (same as input) to {MAX_SIZE}, not {self.size}")
        if not 1 <= self.supersample <= MAX_SUPERSAMPLE:
            problems.append(f"supersample must be 1 to {MAX_SUPERSAMPLE}, not {self.supersample}")
        if self.pad < -1:
            problems.append(f"padding must be -1 (fill all) or more, not {self.pad}")
        if self.normal_convention not in CONVENTIONS:
            problems.append(f"normal convention must be one of {', '.join(CONVENTIONS)}, "
                            f"not {self.normal_convention!r}")
        if self.threads < 0:
            problems.append(f"threads must be 0 (automatic) or more, not {self.threads}")
        if problems:
            raise TransferError("Invalid settings: " + "; ".join(problems))


@dataclass
class Result:
    """Outcome of transferring one texture.

    Attributes:
        image: HxWxC float32 in 0..1, as saved.
        mask: HxW bool, pixels covered by the target UVs (before padding).
        kind: "srgb", "normal" or "linear".
        convention: "dx" or "gl" for normal maps, else "".
        notes: messages for the user.
        path: where the image was saved (set by `Transfer.transfer_file`).
    """

    image: np.ndarray
    mask: np.ndarray
    kind: str
    convention: str = ""
    notes: list = field(default_factory=list)
    path: str = ""

    @property
    def coverage(self):
        """Fraction of the output covered by target UVs."""
        return float(self.mask.mean())


@dataclass
class TiledMap:
    """One output texture put together from a texture per source UDIM tile.

    Attributes:
        name: output file name, e.g. "Head_Basecolor.png" (16-bit output is always PNG).
        tiles: {UDIM number: texture path}, e.g. {1001: "Head_Basecolor.png",
            1002: "Teeth_Color.png"}.
        kind: "srgb", "normal" or "linear"; None guesses it from the first
            texture's name (see `classify`).
        fill: for source triangles whose tile has no texture: a colour as
            stored in the file (1 to 4 values in 0..1, e.g. (0.5, 0.5, 0.5)),
            or None to leave them empty so edge padding covers them. Normal
            maps always get a flat normal there.
    """

    name: str
    tiles: dict
    kind: str = None
    fill: tuple = None


def check_fill(fill):
    """A fill colour as a tuple of 1 to 4 floats in 0..1, or None for no fill.

    Raises:
        TransferError: it isn't one.
    """
    if fill is None:
        return None
    if isinstance(fill, np.ndarray):
        fill = fill.tolist()
    ok = isinstance(fill, (list, tuple)) and 1 <= len(fill) <= 4 and all(
        isinstance(c, (int, float)) and not isinstance(c, bool) and 0 <= c <= 1 for c in fill)
    if not ok:
        raise TransferError(f"The fill must be 1 to 4 numbers from 0 to 1, not {fill!r}")
    return tuple(float(c) for c in fill)


def check_map_name(name):
    """Raise TransferError unless `name` is a plain image file name such as "Body_Basecolor.png"."""
    if not isinstance(name, str) or not name.strip() or os.path.basename(name) != name \
            or name in (".", "..") or "/" in name or "\\" in name:
        raise TransferError(f"A map's name must be a file name such as \"Body_Basecolor.png\", not {name!r}")
    if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
        raise TransferError(f"The map name {name!r} needs an image extension "
                            f"({', '.join(IMAGE_EXTENSIONS)})")


def classify(name, overrides=None):
    """Guess a texture type from its file name.

    Args:
        name: file name or path.
        overrides: optional {file name or stem: kind} that takes priority.

    Returns:
        "normal" for names containing normal/nrm/_n., "srgb" for
        basecolor/albedo/diffuse/color/emissive, otherwise "linear".
    """
    base = os.path.basename(name)
    for key, kind in (overrides or {}).items():
        if key.lower() in (base.lower(), os.path.splitext(base)[0].lower()):
            return kind
    low = base.lower()
    if re.search(r"normal|nrm|_n\.|_nm\.|_nor\.", low):
        return "normal"
    if re.search(r"base.?colou?r|albedo|diffuse|_col|colou?r|emissi", low):
        return "srgb"
    return "linear"


OUTPUT_FOLDER = "transferred"


def default_output_dir(target_fbx):
    """Default output folder: "transferred" next to the target FBX."""
    return os.path.join(os.path.dirname(os.path.abspath(target_fbx)), OUTPUT_FOLDER)


def output_name(path, sixteen_bit):
    """File name a transferred texture is saved under (16-bit output is always PNG)."""
    name = os.path.basename(path)
    return os.path.splitext(name)[0] + ".png" if sixteen_bit else name


def output_size(width, height, size):
    """Output (width, height) for an input of that size when the longest side should be `size`.

    `size` 0 keeps the input size; the aspect ratio is always kept.
    """
    if not size:
        return width, height
    if width >= height:
        return size, max(1, round(size * height / width))
    return max(1, round(size * width / height)), size


def estimate_memory(width, height, channels=4, kind="srgb", options=None):
    """Rough peak memory in bytes for transferring one texture of this size.

    The peak is while resampling: the loaded image, its linearised copy
    (colour only), the output accumulators and the batches in flight
    (measured at ~200-270 bytes per candidate sub-sample; 300 is used). Padding, encoding and
    saving work in row chunks and need much less.
    """
    o = options or Options()
    out_w, out_h = output_size(width, height, o.size)
    image = width * height * channels * 4
    linear_copy = image if kind == "srgb" else 0
    accumulators = out_w * out_h * (channels + 1) * 4
    batches = (o.threads or default_workers()) * BATCH_SAMPLES * _BATCH_BYTES_PER_SAMPLE
    return image + linear_copy + accumulators + batches


class _Progress:
    """Reports progress for a sub-range of the overall 0..1 and checks for cancel."""

    def __init__(self, callback=None, cancel=None, lo=0.0, hi=1.0):
        self.callback, self.cancel, self.lo, self.hi = callback, cancel, lo, hi

    def __call__(self, fraction, message=None):
        if self.cancel is not None and self.cancel():
            raise Cancelled()
        if self.callback is not None:
            fraction = min(max(fraction, 0.0), 1.0)
            self.callback(self.lo + (self.hi - self.lo) * fraction, message)

    def sub(self, lo, hi):
        """A reporter for the part lo..hi (0..1) of this reporter's range."""
        span = self.hi - self.lo
        return _Progress(self.callback, self.cancel, self.lo + span * lo, self.lo + span * hi)


class Transfer:
    """A checked source/target mesh pair, ready to transfer any number of textures.

    Tangent frames and coverage masks are computed once and reused.

    Args:
        src: mesh with the UVs the textures were made for.
        dst: the same mesh with the new UVs.
        notes: extra warnings to include in `warnings`.

    Attributes:
        warnings: messages for the user about the meshes (none are fatal).
        tiles: UDIM tiles the source UVs use, in order, e.g. [1001, 1002, 1003].

    Raises:
        TransferError: the meshes don't have the same topology.
    """

    def __init__(self, src, dst, notes=None):
        self.src, self.dst = src, dst
        _check_topology(src, dst)
        self._frames = None
        self._coverage = {}
        self._src_tri_uv = src.uv[src.tris]
        self._dst_tri_uv = dst.uv[dst.tris]
        self._src_tile = udims.triangle_tiles(self._src_tri_uv)
        self.tiles = [int(t) for t in np.unique(self._src_tile) if t != udims.NO_TILE]
        self.warnings = (list(notes or []) + _mesh_warnings(src, dst, len(self.tiles) > 1)
                         + self._tile_warnings())

    @classmethod
    def from_fbx(cls, src_path, dst_path, uv_set=0):
        """Load and check a source/target pair of FBX files.

        Meshes are paired by name when both files contain the same uniquely
        named meshes (in any order), otherwise by file order.

        Args:
            src_path: FBX the textures were made for.
            dst_path: the same mesh with the new UVs.
            uv_set: UV channel to use on both (0 = first).

        Raises:
            TransferError: a file can't be read, or the meshes don't match.
        """
        src_parts = load_fbx_parts(src_path, uv_set)
        dst_parts = load_fbx_parts(dst_path, uv_set)
        dst_parts, notes = _pair_parts(src_parts, dst_parts)
        return cls(Mesh(src_parts, src_path), Mesh(dst_parts, dst_path), notes)

    # -- info -----------------------------------------------------------------

    @property
    def n_vertices(self):
        return len(self.src.P)

    @property
    def n_triangles(self):
        return len(self.src.tris)

    @property
    def mesh_names(self):
        return [p.name for p in self.src.parts]

    def tile_shares(self):
        """{tile: fraction of the target UV area whose source triangles lie in that tile}.

        Triangles outside the UDIM range are left out.
        """
        uv = self._dst_tri_uv
        area = 0.5 * np.abs((uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
                            - (uv[:, 2, 0] - uv[:, 0, 0]) * (uv[:, 1, 1] - uv[:, 0, 1]))
        total = max(float(area.sum()), 1e-20)
        return {t: float(area[self._src_tile == t].sum()) / total for t in self.tiles}

    def _tile_warnings(self):
        if len(self.tiles) < 2:
            return []
        warnings = [f"The source UVs use tiles {udims.describe(self.tiles)}. A tiled map gives each "
                    "tile its own texture; a single texture wraps onto all of them."]
        n = int(udims.straddling(self._src_tri_uv, self._src_tile).sum())
        if n:
            warnings.append(f"{n:,} source triangle(s) cross a tile edge. They're sampled from the "
                            "tile their centre is in, so they may pick up the wrong texture at the edge.")
        n = int((self._src_tile == udims.NO_TILE).sum())
        if n:
            warnings.append(f"{n:,} source triangle(s) are outside the UDIM tiles (U 0 to 10, V 0 and "
                            "up). In a tiled transfer they get the map's fill.")
        return warnings

    def summary(self):
        """One-line description, e.g. "482 vertices, 960 triangles, 1 mesh"."""
        n = len(self.src.parts)
        return (f"{self.n_vertices:,} vertices, {self.n_triangles:,} triangles, "
                f"{n} mesh{'es' if n != 1 else ''}")

    # -- cached pieces -----------------------------------------------------------

    def frames(self):
        """((T, B, N) of the source, (T, B, N) of the target), computed on first use."""
        if self._frames is None:
            self._frames = (corner_tangent_frames(self.src), corner_tangent_frames(self.dst))
        return self._frames

    def source_coverage(self, width, height):
        """HxW bool mask of texels covered by the source UVs (cached per size)."""
        if (width, height) not in self._coverage:
            self._coverage[(width, height)] = coverage_counts(self._src_tri_uv, width, height) > 0
        return self._coverage[(width, height)]

    # -- transfer -------------------------------------------------------------

    def transfer_image(self, img, kind, options=None, progress=None, cancel=None):
        """Transfer one image held in memory.

        Args:
            img: HxWxC float32 in 0..1, as stored on disk (sRGB colour stays sRGB).
            kind: "srgb", "normal" or "linear".
            options: `Options`; defaults if None.
            progress: optional callback(fraction, message or None).
            cancel: optional callable; when it returns True the transfer stops.

        Returns:
            Result.

        Raises:
            TransferError: invalid settings, or a normal map without RGB.
            Cancelled: `cancel` returned True.
        """
        return self._transfer(img, kind, options or Options(), _Progress(progress, cancel))

    def transfer_file(self, path, out_dir, kind=None, options=None, progress=None, cancel=None):
        """Load a texture, transfer it and save it into `out_dir` under the same name.

        Args:
            path: input texture.
            out_dir: output folder (created if needed).
            kind: texture type; guessed from the file name if None (see `classify`).
            options, progress, cancel: as for `transfer_image`.

        Returns:
            Result with `path` set to the saved file.

        Raises:
            TransferError: unreadable input, invalid settings, the output would
                overwrite the input, or the output can't be written.
            Cancelled: `cancel` returned True.
        """
        o = options or Options()
        o.validate()
        report = _Progress(progress, cancel)
        report(0.0, "Loading")
        pixels, high_precision, load_notes = load_image(path)
        sixteen = o.sixteen_bit or high_precision
        out_path = os.path.join(out_dir, output_name(path, sixteen))
        if os.path.normcase(os.path.abspath(out_path)) == os.path.normcase(os.path.abspath(path)):
            raise TransferError("The output would overwrite the original texture. "
                                "Choose a different output folder.")

        result = self._transfer(pixels, kind or classify(path), o, report.sub(0.05, 0.95))
        del pixels                                # free the input before saving (matters at 8K)
        result.notes = load_notes + result.notes
        report(0.95, "Saving")
        try:
            os.makedirs(out_dir, exist_ok=True)
            save_image(out_path, result.image, sixteen)
        except OSError as e:
            raise TransferError(f"Couldn't save {out_path}: {e.strerror or e}") from e
        result.path = out_path
        report(1.0, "Done")
        return result

    def transfer_tiles(self, images, kind, options=None, fill=None, progress=None, cancel=None):
        """Transfer one map from images held in memory, one per source UDIM tile.

        Args:
            images: {UDIM number: HxWxC float32 in 0..1, as stored on disk}.
            kind: "srgb", "normal" or "linear".
            options: `Options`; defaults if None. With size 0 the output takes
                the size of the largest image.
            fill: see `TiledMap.fill`.
            progress, cancel: as for `transfer_image`.

        Returns:
            Result.

        Raises:
            TransferError: invalid settings, tiles or fill, or a normal map without RGB.
            Cancelled: `cancel` returned True.
        """
        sources = {tile: _Source(img.shape[1], img.shape[0], lambda img=img: img, f"tile {tile}")
                   for tile, img in images.items()}
        return self._transfer_tiled(sources, kind, fill, options or Options(), _Progress(progress, cancel))

    def transfer_map(self, tiled, out_dir, options=None, progress=None, cancel=None):
        """Put together a `TiledMap` from its textures and save it into `out_dir`.

        Textures are loaded one at a time, so the peak memory is about that
        of transferring the largest one alone (see `estimate_memory`).

        Args:
            tiled: `TiledMap`.
            out_dir: output folder (created if needed).
            options, progress, cancel: as for `transfer_tiles`.

        Returns:
            Result with `path` set to the saved file.

        Raises:
            TransferError: no textures, an unreadable texture, invalid settings,
                tiles or fill, the output would overwrite an input, or it can't
                be written.
            Cancelled: `cancel` returned True.
        """
        o = options or Options()
        o.validate()
        report = _Progress(progress, cancel)
        check_map_name(tiled.name)
        if not tiled.tiles:
            raise TransferError(f"{tiled.name}: no textures to transfer")
        load_notes, sources, sixteen = [], {}, o.sixteen_bit
        for tile, path in sorted(tiled.tiles.items(), key=lambda item: udims.tile_number(item[0])):
            label = os.path.basename(path)
            try:
                width, height, high_precision = image_info(path)
            except OSError as e:
                raise TransferError(f"Couldn't read {label}: {e}") from e
            sixteen = sixteen or high_precision
            sources[tile] = _Source(width, height, _loader(path, load_notes), label)
        out_path = os.path.join(out_dir, output_name(tiled.name, sixteen))
        inputs = {os.path.normcase(os.path.abspath(p)) for p in tiled.tiles.values()}
        if os.path.normcase(os.path.abspath(out_path)) in inputs:
            raise TransferError(f"The output {os.path.basename(out_path)} would overwrite one of its "
                                "textures. Choose a different name or output folder.")

        kind = tiled.kind or classify(next(iter(sources.values())).label)
        result = self._transfer_tiled(sources, kind, tiled.fill, o, report.sub(0.0, 0.95))
        result.notes = load_notes + result.notes
        report(0.95, "Saving")
        try:
            os.makedirs(out_dir, exist_ok=True)
            save_image(out_path, result.image, sixteen)
        except OSError as e:
            raise TransferError(f"Couldn't save {out_path}: {e.strerror or e}") from e
        result.path = out_path
        report(1.0, "Done")
        return result

    def _transfer(self, img, kind, o, report):
        o.validate()
        if kind not in KINDS:
            raise TransferError(f"Unknown texture type {kind!r}; use one of {', '.join(KINDS)}")
        height, width, channels = img.shape
        if kind == "normal" and channels < 3:
            raise TransferError(f"A normal map needs RGB channels; this image has {channels}. "
                                "Set its type to Data instead.")
        out_w, out_h = output_size(width, height, o.size)

        notes, convention = [], ""
        if kind == "normal":
            convention, note = self._resolve_convention(img, o.normal_convention, report)
            notes.append(note)
        work = _linearise(img, kind)
        out, mask = self._resample(work, kind, convention == "dx", out_w, out_h, o,
                                   report.sub(0.05, 0.85))
        del work
        image, _ = _finish(out, mask, kind, o.pad, report.sub(0.85, 1.0))
        report(1.0)
        return Result(image, mask, kind, convention, notes)

    def _transfer_tiled(self, sources, kind, fill, o, report):
        """Transfer one map from {tile: _Source}. See `transfer_tiles`."""
        o.validate()
        if kind not in KINDS:
            raise TransferError(f"Unknown texture type {kind!r}; use one of {', '.join(KINDS)}")
        if not sources:
            raise TransferError("No textures to transfer")
        sources = {udims.tile_number(tile): source for tile, source in sources.items()}
        fill = _fill_value(fill, kind)
        notes = []
        unused = sorted(t for t in sources if t not in self.tiles)
        if unused:
            notes.append(f"No part of the source mesh is in tile {udims.describe(unused)}, "
                         "so its texture was skipped.")
        # largest part first: with auto-detection it decides the normal map format of the result
        shares = self.tile_shares()
        textured = sorted((t for t in sources if t in shares), key=lambda t: -shares[t])
        size_from = [sources[t] for t in textured] or list(sources.values())
        largest = max(size_from, key=lambda s: s.width * s.height)
        out_w, out_h = output_size(largest.width, largest.height, o.size)

        rest = np.flatnonzero(~np.isin(self._src_tile, textured))
        weights = [shares[t] for t in textured]
        weights.append(max(0.0, 1.0 - sum(weights)) if fill is not None and rest.size else 0.0)
        total = max(sum(weights), 1e-20)
        ends = np.cumsum(weights) / total
        starts = ends - np.asarray(weights) / total

        acc, out_convention, tile_notes = None, "", {}
        for i, tile in enumerate(textured):
            source = sources[tile]
            step = report.sub(0.85 * starts[i], 0.85 * ends[i])
            step(0.0, f"Loading {source.label}")
            img = source.load()
            channels = img.shape[2]
            if kind == "normal" and channels < 3:
                raise TransferError(f"{source.label}: a normal map needs RGB channels; this image has "
                                    f"{channels}. Set the map's type to Data instead.")
            directx = directx_out = None
            if kind == "normal":
                convention, note = self._tile_convention(img, tile, o.normal_convention, step)
                out_convention = out_convention or convention
                directx, directx_out = convention == "dx", out_convention == "dx"
                if convention != out_convention:
                    note += f", converted to {_CONVENTION_NAMES[out_convention]} to match the rest"
                tile_notes[tile] = f"Tile {tile}: {note}"
            work = _linearise(img, kind)
            del img
            acc = acc.widened(channels) if acc else _Accumulator(out_w * out_h, channels)

            def sample(tri, bary, work=work, directx=directx, directx_out=directx_out, width=acc.channels):
                return _conform(self._sample_source(work, kind, directx, tri, bary, directx_out), width)

            step(0.05, "Transferring")
            self._accumulate(np.flatnonzero(self._src_tile == tile), out_w, out_h, o, sample, acc,
                             step.sub(0.05, 1.0))
            del work, sample
        notes += [tile_notes[t] for t in sorted(tile_notes)]

        if fill is not None and rest.size:
            acc = acc.widened(len(fill)) if acc else _Accumulator(out_w * out_h, len(fill))
            value = _conform(fill, acc.channels)
            step = report.sub(0.85 * starts[-1], 0.85)
            step(0.0, "Filling parts without a texture")
            self._accumulate(rest, out_w, out_h, o,
                             lambda tri, bary: np.broadcast_to(value, (len(tri), len(value))), acc, step)
        elif rest.size and textured:
            notes.append("Parts of the mesh whose tile has no texture were left empty.")
        if acc is None:
            acc = _Accumulator(out_w * out_h, 3 if kind == "normal" else 4)
        out, mask = acc.average(out_w, out_h)
        image, _ = _finish(out, mask, kind, o.pad, report.sub(0.85, 1.0))
        report(1.0)
        return Result(image, mask, kind, out_convention, notes)

    def _resolve_convention(self, img, requested, report):
        """("dx" or "gl", note) - detected from the image when requested is "auto"."""
        report(0.0, "Preparing tangent frames")
        self.frames()
        if requested != "auto":
            return requested, _CONVENTION_NAMES[requested] + " normal map"
        report(0.02, "Detecting normal map format")
        height, width = img.shape[:2]
        return normalmaps.detect_convention(img, self.source_coverage(width, height))

    def _tile_convention(self, img, tile, requested, report):
        """Like `_resolve_convention`, judged only on the part of the image that `tile` uses."""
        report(0.0, "Preparing tangent frames")
        self.frames()
        if requested != "auto":
            return requested, _CONVENTION_NAMES[requested] + " normal map"
        report(0.02, "Detecting normal map format")
        height, width = img.shape[:2]
        key = (tile, width, height)
        if key not in self._coverage:
            mine = self._src_tile == tile
            local = udims.local_uv(self._src_tri_uv[mine], self._src_tile[mine])
            self._coverage[key] = coverage_counts(local, width, height) > 0
        return normalmaps.detect_convention(img, self._coverage[key])

    def _resample(self, work, kind, directx, out_w, out_h, o, report):
        """Average source samples into an out_h x out_w image. Returns (image, covered mask)."""
        report(0.0, "Transferring")
        acc = _Accumulator(out_w * out_h, work.shape[2])
        self._accumulate(None, out_w, out_h, o,
                         lambda tri, bary: self._sample_source(work, kind, directx, tri, bary), acc, report)
        return acc.average(out_w, out_h)

    def _accumulate(self, tris, out_w, out_h, o, sample, acc, report):
        """Add sample(tri, bary) for every sub-sample of target triangles `tris` (None: all) into `acc`."""
        tri_uv = self._dst_tri_uv if tris is None else self._dst_tri_uv[tris]
        plan = RasterPlan(tri_uv, out_w, out_h, o.supersample)

        def sample_batch(k):                      # runs on worker threads
            hits = plan.batch(k)
            if hits is None:
                return None
            tri, pixel, bary = hits
            return sum_by_pixel(pixel, sample(tri if tris is None else tris[tri], bary))

        def add(k, batch_sums):                   # runs on this thread, in batch order
            if batch_sums is not None:
                pixels, n, total = batch_sums
                acc.counts[pixels] += n
                acc.sums[pixels] += total
            report(plan.done_after(k))

        run_batches(plan, sample_batch, o.threads or default_workers(), add)

    def _sample_source(self, work, kind, directx, tri, bary, directx_out=None):
        """Values of the source texture at the source-UV positions of target sub-samples."""
        src_uv = np.einsum("nk,nkj->nj", bary, self._src_tri_uv[tri])
        values = sample_bilinear(work, src_uv[:, 0], src_uv[:, 1])
        if kind == "normal":
            src_frames, dst_frames = self.frames()
            values[:, :3] = normalmaps.reorient(values[:, :3], bary, src_frames, self.src.tris[tri],
                                                dst_frames, self.dst.tris[tri], directx, directx_out)
        return values


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_CONVENTION_NAMES = {"dx": "DirectX", "gl": "OpenGL"}


@dataclass
class _Source:
    """One tile's texture: its size, and how to load it (HxWxC float32 in 0..1)."""

    width: int
    height: int
    load: object
    label: str


def _loader(path, notes):
    """A function that loads `path` and adds its load notes to `notes`."""
    def load():
        loaded = load_image(path)
        notes.extend(f"{os.path.basename(path)}: {note}" for note in loaded.notes)
        return loaded.pixels
    return load


def _fill_value(fill, kind):
    """The fill in working space (linear light for colour), flat for normal maps, or None."""
    if kind == "normal":
        return np.float32(_FLAT_NORMAL)
    fill = check_fill(fill)
    if fill is None:
        return None
    return _linearise(np.float32(fill)[None, None, :], kind)[0, 0]


def _channel_layout(channels):
    """(colour channels, has alpha) of an image with this many channels."""
    return _colour_channels(channels), channels in (2, 4)


def _conform(values, channels, one=1.0):
    """(..., C) values as (..., channels): grey is repeated into RGB, missing alpha is `one`."""
    have = values.shape[-1]
    if have == channels:
        return values
    colour_in, alpha_in = _channel_layout(have)
    colour_out, alpha_out = _channel_layout(channels)
    colour = values[..., :colour_in]
    if colour_out > colour_in:
        colour = np.repeat(colour, colour_out, axis=-1)
    parts = [colour]
    if alpha_out:
        parts.append(values[..., -1:] if alpha_in
                     else np.broadcast_to(np.asarray(one, np.float32), values.shape[:-1] + (1,)))
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


class _Accumulator:
    """Per-pixel sums and sample counts for an output image being built."""

    def __init__(self, n_pixels, channels):
        self.sums = np.zeros((n_pixels, channels), np.float32)
        self.counts = np.zeros(n_pixels, np.float32)

    @property
    def channels(self):
        return self.sums.shape[1]

    def widened(self, channels):
        """This accumulator, with room for values with `channels` channels too."""
        colour_a, alpha_a = _channel_layout(self.channels)
        colour_b, alpha_b = _channel_layout(channels)
        wanted = max(colour_a, colour_b) + (1 if alpha_a or alpha_b else 0)
        if wanted != self.channels:
            # pixels summed so far count as opaque: their alpha sum is their sample count
            self.sums = _conform(self.sums, wanted, one=self.counts[:, None])
        return self

    def average(self, width, height):
        """(HxWxC average, HxW covered mask). Reuses the sums array."""
        covered = self.counts > 0
        self.sums[covered] /= self.counts[covered][:, None]
        return self.sums.reshape(height, width, -1), covered.reshape(height, width)

def _colour_channels(channels):
    """How many leading channels hold colour (the rest is alpha)."""
    return 3 if channels >= 3 else 1


def _linearise(img, kind):
    """Working copy for averaging: sRGB colour is converted to linear light."""
    if kind != "srgb":
        return img.astype(np.float32, copy=False)
    work = img.astype(np.float32, copy=True)
    n = _colour_channels(work.shape[2])
    work[..., :n] = srgb_to_linear(work[..., :n])
    return work


def _finish(out, mask, kind, pad, report):
    """Pad island edges and encode for saving. Modifies `out`; returns (image, filled mask)."""
    if kind == "normal":
        _by_rows(out, _normalise_rgb)
        out[~mask, :3] = _FLAT_NORMAL
    filled = mask
    if pad != 0:
        report(0.0, "Padding edges")
        out, filled = dilate(out, mask, pad, progress=report)
    if kind == "normal":
        _by_rows(out, _encode_normals)
        out[~filled, :3] = (0.5, 0.5, 1.0)
    elif kind == "srgb":
        n = _colour_channels(out.shape[2])

        def encode(rows):
            rows[..., :n] = linear_to_srgb(rows[..., :n])
        _by_rows(out, encode)
    return out, filled


ROW_CHUNK = 512   # rows processed at a time when encoding, to keep temporaries small


def _by_rows(img, fn):
    """Apply an in-place function to horizontal strips of `img`."""
    for y in range(0, img.shape[0], ROW_CHUNK):
        fn(img[y:y + ROW_CHUNK])


def _normalise_rgb(img):
    n = img[..., :3]
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-20)


def _encode_normals(img):
    """Unit normals (-1..1) to stored colours (0..1), in place."""
    _normalise_rgb(img)
    img[..., :3] = img[..., :3] * 0.5 + 0.5


def _pair_parts(src_parts, dst_parts):
    """Order target parts to match the source. Returns (target parts, notes)."""
    if len(src_parts) != len(dst_parts):
        raise TransferError(
            f"The source has {len(src_parts)} mesh(es) ({', '.join(p.name for p in src_parts)}) "
            f"but the target has {len(dst_parts)} ({', '.join(p.name for p in dst_parts)}). "
            "Both files need the same meshes.")
    src_names = [p.name for p in src_parts]
    dst_names = [p.name for p in dst_parts]
    if src_names != dst_names and len(set(src_names)) == len(src_names) \
            and set(src_names) == set(dst_names):
        by_name = {p.name: p for p in dst_parts}
        return [by_name[n] for n in src_names], \
            ["The meshes are in a different order in the two files; matched them by name."]
    return dst_parts, []


def _check_topology(src, dst):
    """Raise TransferError unless every part pair has the same vertices and polygons."""
    if len(src.parts) != len(dst.parts):
        raise TransferError(f"The source has {len(src.parts)} mesh(es) but the target has "
                            f"{len(dst.parts)}. Both need the same meshes.")
    for a, b in zip(src.parts, dst.parts, strict=True):
        label = f"'{a.name}'" if a.name == b.name else f"'{a.name}' / '{b.name}'"
        if len(a.P) != len(b.P):
            raise TransferError(
                f"Mesh {label}: the source has {len(a.P):,} vertices but the target has {len(b.P):,}. "
                "Both FBX files must be the same mesh with only the UVs changed.")
        if len(a.corner_vertex) != len(b.corner_vertex) \
                or not np.array_equal(a.corner_vertex, b.corner_vertex) \
                or not np.array_equal(a.tris, b.tris):
            raise TransferError(
                f"Mesh {label}: same vertex count but the polygons differ (faces or vertex order "
                "changed). Both FBX files must be the same mesh with only the UVs changed.")


def _mesh_warnings(src, dst, tiled_source=False):
    """Non-fatal problems worth telling the user about.

    With `tiled_source`, source UVs outside 0 to 1 are expected (UDIM tiles)
    and aren't warned about here.
    """
    warnings = []
    drift = np.abs(src.P - dst.P).max()
    if drift > POSITION_DRIFT_WARN * max(1e-9, np.abs(src.P).max()):
        warnings.append(f"Vertex positions differ between the files (by up to {drift:.4g}). "
                        "This is usually harmless, e.g. a different export scale.")
    for label, mesh in (("source", src), ("target", dst)):
        if label == "source" and tiled_source:
            continue
        outside = ((mesh.uv < -1e-4) | (mesh.uv > 1 + 1e-4)).any(axis=1).mean()
        if outside > 0.001:
            warnings.append(f"{100 * outside:.1f}% of the {label} UVs are outside the 0 to 1 square. "
                            "They wrap onto the same texture (UDIM tiles aren't supported).")
        lo, hi = TILE_RANGE
        tri_uv = mesh.uv[mesh.tris]
        far = ((tri_uv < lo) | (tri_uv > hi)).any(axis=(1, 2)).sum()
        if far:
            warnings.append(f"{far:,} {label} triangle(s) have UVs more than {-lo} tiles outside "
                            "0 to 1; the parts beyond that are skipped.")
    overlap = _overlap_fraction(dst.uv[dst.tris])
    if overlap > OVERLAP_WARN:
        warnings.append(f"{100 * overlap:.1f}% of the target UV area is covered by more than one "
                        "triangle (overlapping or stacked islands). Those texels get an average "
                        "of the overlapping parts.")
    return warnings


def _overlap_fraction(tri_uv):
    # ignore centres exactly on shared edges: those are neighbours, not overlaps
    counts = coverage_counts(tri_uv, OVERLAP_CHECK_SIZE, OVERLAP_CHECK_SIZE, edge_tolerance=-1e-6)
    covered = (counts > 0).sum()
    return float((counts > 1).sum() / covered) if covered else 0.0
