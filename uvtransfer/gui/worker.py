"""Background work for the window: loading meshes and running transfers.

Results are delivered through `Bridge` signals, which Qt queues onto the UI thread.
"""

import threading
import time
import traceback
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from .. import tiles as udims
from ..errors import Cancelled, TransferError
from ..transfer import Transfer
from .preview import uv_wireframe

PROGRESS_INTERVAL = 0.08  # seconds between progress updates sent to the UI


class Bridge(QObject):
    """Signals emitted from worker threads and handled on the UI thread."""

    meshLoaded = Signal(object, object)            # token, MeshLoad
    progress = Signal(int, float, str)             # row id, fraction, stage message
    rowDone = Signal(int, object, float)           # row id, Result, seconds
    rowFailed = Signal(int, str, bool)             # row id, message, cancelled
    batchDone = Signal(bool)                       # cancelled
    previewReady = Signal(int, object, object)     # request number, before QImage, after QImage


@dataclass
class MeshLoad:
    """Outcome of loading a source/target pair: a Transfer and wireframes, or an error.

    `tile_wireframes` holds the source UVs of each UDIM tile, moved into 0 to 1,
    when the source uses more than one tile.
    """

    transfer: object = None
    error: str = ""
    src_wireframe: object = None
    dst_wireframe: object = None
    tile_wireframes: dict = None


def load_meshes(bridge, token, src_path, dst_path, uv_set):
    """Load and check a mesh pair on a background thread; emits bridge.meshLoaded(token, MeshLoad)."""
    def work():
        try:
            xfer = Transfer.from_fbx(src_path, dst_path, uv_set)
            src_edges = xfer.src.uv_edges()
            result = MeshLoad(xfer, "", uv_wireframe(src_edges), uv_wireframe(xfer.dst.uv_edges()),
                              tile_wireframes(src_edges) if len(xfer.tiles) > 1 else {})
        except TransferError as e:
            result = MeshLoad(error=str(e))
        except Exception as e:                    # keep the UI alive; show what happened
            traceback.print_exc()
            result = MeshLoad(error=f"Unexpected error reading the meshes: {e!r}")
        bridge.meshLoaded.emit(token, result)

    threading.Thread(target=work, daemon=True, name="mesh-load").start()


def tile_wireframes(edges):
    """{tile: wireframe of the UV edges in that tile, moved into 0 to 1}."""
    tiles = udims.triangle_tiles(edges)           # works on edges too: the tile of each midpoint
    local = udims.local_uv(edges, tiles)
    return {int(t): uv_wireframe(local[tiles == t]) for t in set(tiles.tolist()) if t != udims.NO_TILE}


@dataclass
class Job:
    """One output to make: its table row, and a texture (path and type) or a `TiledMap`."""

    row_id: int
    path: str = ""
    kind: str = ""
    tiled: object = None


class TransferWorker(threading.Thread):
    """Transfers a list of textures one after another, reporting through `bridge`."""

    def __init__(self, bridge, transfer, jobs, out_dir, options):
        super().__init__(daemon=True, name="transfer")
        self.bridge, self.transfer, self.jobs = bridge, transfer, jobs
        self.out_dir, self.options = out_dir, options
        self._cancel = threading.Event()

    def cancel(self):
        """Ask the worker to stop at the next checkpoint."""
        self._cancel.set()

    def run(self):
        cancelled = False
        for job in self.jobs:
            if self._cancel.is_set() or not self._run_job(job):
                cancelled = True
                break
        self.bridge.batchDone.emit(cancelled)

    def _run_job(self, job):
        """Transfer one texture. Returns False if the batch was cancelled."""
        last_sent = [0.0]

        def progress(fraction, message):
            now = time.monotonic()
            if message or fraction >= 1.0 or now - last_sent[0] > PROGRESS_INTERVAL:
                last_sent[0] = now
                self.bridge.progress.emit(job.row_id, fraction, message or "")

        started = time.monotonic()
        try:
            if job.tiled is not None:
                result = self.transfer.transfer_map(job.tiled, self.out_dir, self.options,
                                                    progress=progress, cancel=self._cancel.is_set)
            else:
                result = self.transfer.transfer_file(job.path, self.out_dir, job.kind, self.options,
                                                     progress=progress, cancel=self._cancel.is_set)
        except Cancelled:
            self.bridge.rowFailed.emit(job.row_id, "Cancelled", True)
            return False
        except TransferError as e:
            self.bridge.rowFailed.emit(job.row_id, str(e), False)
        except MemoryError:
            self.bridge.rowFailed.emit(job.row_id, "Ran out of memory. Try a smaller output size "
                                                   "or a lower quality setting.", False)
        except Exception as e:                    # report, then carry on with the next texture
            traceback.print_exc()
            self.bridge.rowFailed.emit(job.row_id, f"Unexpected error: {e!r}", False)
        else:
            self.bridge.rowDone.emit(job.row_id, result, time.monotonic() - started)
        return True
