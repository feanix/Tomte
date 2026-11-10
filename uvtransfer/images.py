"""
Image loading, saving, colour conversion and sampling.

Images are handled as float32 HxWxC arrays in 0..1 (C = 1-4 channels).
High-bit-depth colour images (16-bit PNG/TIFF) are read with OpenCV when it is
installed, because Pillow reduces them to 8 bits.
"""

import os
import struct
import zlib
from typing import NamedTuple

import numpy as np
from PIL import Image

from .errors import TransferError

try:
    import cv2
except ImportError:      # optional dependency
    cv2 = None

IMAGE_EXTENSIONS = (".png", ".tga", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_TIFF_BITS_PER_SAMPLE = 258
_CV2_HINT = "install opencv-python-headless to read it at full precision"


class LoadedImage(NamedTuple):
    """Result of `load_image`."""

    pixels: np.ndarray        # HxWxC float32, 0..1
    high_precision: bool      # source had more than 8 bits per channel
    notes: list               # messages for the user (e.g. precision lost)


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------

def srgb_to_linear(c):
    """sRGB-encoded values (0..1) to linear light."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c):
    """Linear values to sRGB encoding, clipped to 0..1."""
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def _png_header(path):
    """(bit depth, colour type) of a PNG file, or None for other files."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
    except OSError:
        return None
    if len(head) == 26 and head.startswith(_PNG_SIGNATURE) and head[12:16] == b"IHDR":
        return head[24], head[25]
    return None


def bit_depth(path):
    """Bits per channel of an image file (8 when it can't be determined)."""
    hdr = _png_header(path)
    if hdr:
        return hdr[0]
    try:
        with Image.open(path) as im:
            if im.format == "TIFF":
                bits = im.tag_v2.get(_TIFF_BITS_PER_SAMPLE)
                if bits:
                    return max(bits) if isinstance(bits, tuple) else int(bits)
            if im.mode.startswith("I;16"):
                return 16
            if im.mode in ("I", "F"):
                return 32
    except OSError:
        pass
    return 8


def image_info(path):
    """Cheap header read: (width, height, high_precision)."""
    with Image.open(path) as im:
        w, h = im.size
    return w, h, bit_depth(path) > 8


def load_image(path):
    """Load an image as float32 HxWxC in 0..1.

    Returns:
        LoadedImage(pixels, high_precision, notes).

    Raises:
        TransferError: the file can't be read.
    """
    depth = bit_depth(path)
    if depth > 8 and cv2 is not None:
        pixels = _load_with_cv2(path)
        if pixels is not None:
            return LoadedImage(pixels, True, [])

    try:
        im = Image.open(path)
        im.load()
    except OSError as e:
        raise TransferError(f"Couldn't read {os.path.basename(path)}: {e}") from e
    pixels, pillow_precise = _pillow_to_array(im)
    notes = []
    if depth > 8 and not pillow_precise:
        notes.append(f"{depth}-bit image was read at 8-bit precision ({_CV2_HINT})")
    return LoadedImage(pixels, depth > 8, notes)


def _load_with_cv2(path):
    """Read with OpenCV (keeps 16-bit/float colour). None if OpenCV can't decode it."""
    a = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        return None
    if a.ndim == 2:
        a = a[..., None]
    elif a.shape[2] >= 3:
        a = a[..., [2, 1, 0] + list(range(3, a.shape[2]))]       # BGR(A) -> RGB(A)
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    return a.astype(np.float32)                                   # float TIFF: as stored


def _pillow_to_array(im):
    """(pixels, full_precision) for a loaded Pillow image."""
    mode = im.mode
    if mode.startswith("I;16") or mode == "I":
        return np.asarray(im).astype(np.float32)[..., None] / 65535.0, True
    if mode == "F":
        return np.asarray(im, np.float32)[..., None], True
    if mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    elif mode == "1":
        im = im.convert("L")
    elif mode not in ("RGB", "RGBA", "L", "LA"):
        im = im.convert("RGBA" if "A" in mode else "RGB")
    arr = np.asarray(im)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0, True
    return arr.astype(np.float32) / 255.0, False


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def save_image(path, pixels, sixteen_bit):
    """Save float 0..1 pixels. 16-bit output is always written as PNG data.

    JPEG can't store alpha, so the alpha channel is dropped for .jpg/.jpeg.

    Raises:
        OSError: the file can't be written.
    """
    ch = pixels.shape[2]
    if sixteen_bit:
        _write_png16(path, _quantise(pixels, 65535, ">u2"))
        return
    if os.path.splitext(path)[1].lower() in (".jpg", ".jpeg") and ch in (2, 4):
        pixels = pixels[..., :ch - 1]
        ch -= 1
    a = _quantise(pixels, 255, np.uint8)
    mode = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[ch]
    Image.fromarray(a[..., 0] if ch == 1 else a, mode).save(path)


def _quantise(pixels, scale, dtype, chunk_rows=512):
    """Clip to 0..1 and round to integers, in row chunks so no full-size float copy is made."""
    out = np.empty(pixels.shape, dtype)
    for y in range(0, pixels.shape[0], chunk_rows):
        out[y:y + chunk_rows] = np.clip(pixels[y:y + chunk_rows], 0, 1) * scale + 0.5
    return out


def _write_png16(path, a):
    """Minimal 16-bit PNG writer (Pillow can't write 16-bit RGB/RGBA). `a` is HxWxC big-endian u2."""
    h, w, ch = a.shape
    color_type = {1: 0, 2: 4, 3: 2, 4: 6}[ch]     # grey, grey+alpha, RGB, RGBA
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))   # filter type 0 per row

    def chunk(kind, payload):
        body = struct.pack(">I", len(payload)) + kind + payload
        return body + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(_PNG_SIGNATURE)
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, color_type, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def sample_bilinear(img, u, v):
    """Bilinearly sample an HxWxC image at UVs (v up, wrapping). Returns (n, C) float32."""
    H, W = img.shape[:2]
    x = u * W - 0.5
    y = (1.0 - v) * H - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None].astype(np.float32)
    fy = (y - y0)[:, None].astype(np.float32)
    x0 %= W
    y0 %= H
    x1 = (x0 + 1) % W
    y1 = (y0 + 1) % H
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy
