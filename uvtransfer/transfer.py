"""
Moving textures from one UV layout to another.

For every sub-sample of the output texture, find the target triangle it lies
in, take the same barycentric point on the source triangle (same mesh, old
UVs), and sample the input texture there. Colour is averaged in linear light;
normal maps are converted into the target's tangent frames.
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np

from . import normalmaps
from .errors import Cancelled, TransferError
from .images import linear_to_srgb, load_image, sample_bilinear, save_image, srgb_to_linear
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

    Raises:
        TransferError: the meshes don't have the same topology.
    """

    def __init__(self, src, dst, notes=None):
        self.src, self.dst = src, dst
        _check_topology(src, dst)
        self.warnings = list(notes or []) + _mesh_warnings(src, dst)
        self._frames = None
        self._coverage = {}
        self._src_tri_uv = src.uv[src.tris]
        self._dst_tri_uv = dst.uv[dst.tris]

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

    def _resolve_convention(self, img, requested, report):
        """("dx" or "gl", note) - detected from the image when requested is "auto"."""
        report(0.0, "Preparing tangent frames")
        self.frames()
        if requested != "auto":
            return requested, {"dx": "DirectX normal map", "gl": "OpenGL normal map"}[requested]
        report(0.02, "Detecting normal map format")
        height, width = img.shape[:2]
        return normalmaps.detect_convention(img, self.source_coverage(width, height))

    def _resample(self, work, kind, directx, out_w, out_h, o, report):
        """Average source samples into an out_h x out_w image. Returns (image, covered mask)."""
        report(0.0, "Transferring")
        plan = RasterPlan(self._dst_tri_uv, out_w, out_h, o.supersample)
        channels = work.shape[2]
        sums = np.zeros((out_w * out_h, channels), np.float32)
        counts = np.zeros(out_w * out_h, np.float32)

        def sample_batch(k):                      # runs on worker threads
            hits = plan.batch(k)
            if hits is None:
                return None
            tri, pixel, bary = hits
            return sum_by_pixel(pixel, self._sample_source(work, kind, directx, tri, bary))

        def add(k, batch_sums):                   # runs on this thread, in batch order
            if batch_sums is not None:
                pixels, n, total = batch_sums
                counts[pixels] += n
                sums[pixels] += total
            report(plan.done_after(k))

        run_batches(plan, sample_batch, o.threads or default_workers(), add)
        covered = counts > 0
        sums[covered] /= counts[covered][:, None]
        return sums.reshape(out_h, out_w, channels), covered.reshape(out_h, out_w)

    def _sample_source(self, work, kind, directx, tri, bary):
        """Values of the source texture at the source-UV positions of target sub-samples."""
        src_uv = np.einsum("nk,nkj->nj", bary, self._src_tri_uv[tri])
        values = sample_bilinear(work, src_uv[:, 0], src_uv[:, 1])
        if kind == "normal":
            src_frames, dst_frames = self.frames()
            values[:, :3] = normalmaps.reorient(values[:, :3], bary, src_frames, self.src.tris[tri],
                                                dst_frames, self.dst.tris[tri], directx)
        return values


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

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


def _mesh_warnings(src, dst):
    """Non-fatal problems worth telling the user about."""
    warnings = []
    drift = np.abs(src.P - dst.P).max()
    if drift > POSITION_DRIFT_WARN * max(1e-9, np.abs(src.P).max()):
        warnings.append(f"Vertex positions differ between the files (by up to {drift:.4g}). "
                        "This is usually harmless, e.g. a different export scale.")
    for label, mesh in (("source", src), ("target", dst)):
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
