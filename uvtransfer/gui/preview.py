"""Preview images and UV wireframes, prepared off the UI thread."""

import os
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image
from PySide6.QtGui import QImage, QPainterPath

PREVIEW_SIZE = 1024       # longest side of preview images
CACHE_ENTRIES = 16


def load_preview(path, max_side=PREVIEW_SIZE):
    """Downscaled RGB QImage of an image file, or None if it can't be read.

    Safe to call from any thread (QImage, unlike QPixmap, is thread-safe).
    """
    try:
        with Image.open(path) as im:
            im.draft("RGB", (max_side, max_side))         # fast JPEG downscale on decode
            if im.mode in ("I;16", "I;16B", "I;16L", "I"):
                im = Image.fromarray((np.asarray(im, np.float32) / 257).clip(0, 255).astype(np.uint8))
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
            data = im.tobytes()
            return QImage(data, im.width, im.height, 3 * im.width, QImage.Format.Format_RGB888).copy()
    except (OSError, ValueError):
        return None


class PreviewCache:
    """Small thread-safe LRU cache of preview images, keyed by path and modification time."""

    def __init__(self, entries=CACHE_ENTRIES):
        self._items = OrderedDict()
        self._entries = entries
        self._lock = threading.Lock()

    def clear(self):
        """Forget every preview (they're reloaded when next shown)."""
        with self._lock:
            self._items.clear()

    def get(self, path):
        """Preview of `path`, loading it if it isn't cached (or changed on disk)."""
        if not path:
            return None
        try:
            key = (os.path.normcase(os.path.abspath(path)), os.path.getmtime(path))
        except OSError:
            return None
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        image = load_preview(path)
        if image is not None:
            with self._lock:
                self._items[key] = image
                while len(self._items) > self._entries:
                    self._items.popitem(last=False)
        return image


def uv_wireframe(edges):
    """QPainterPath of UV edges in UV units (0..1, v up). Build once, draw scaled.

    Args:
        edges: (E, 2, 2) array from `Mesh.uv_edges()`.
    """
    path = QPainterPath()
    for (u0, v0), (u1, v1) in edges.tolist():
        path.moveTo(u0, v0)
        path.lineTo(u1, v1)
    return path
