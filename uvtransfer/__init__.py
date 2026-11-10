"""
uvtransfer: the engine behind Tomte. Moves textures between two versions of
a mesh that differ only in their UVs. Colour is resampled in linear light;
tangent-space normal maps are re-oriented so rotated, mirrored or rescaled
islands come out right.

    from uvtransfer import Transfer, Options
    xfer = Transfer.from_fbx("old_uvs.fbx", "new_uvs.fbx")
    result = xfer.transfer_file("Normal.png", "out", options=Options(pad=16))

Command line: python -m uvtransfer --help
Desktop app:  python -m uvtransfer.gui
Saved setups: Setup.load("Body.tomte"), python -m uvtransfer --setup Body.tomte
"""

from .errors import Cancelled, TransferError
from .images import IMAGE_EXTENSIONS, image_info, load_image, save_image
from .mesh import Mesh, MeshPart, load_fbx_parts
from .setupfile import Setup, TextureEntry
from .system import available_memory, default_workers
from .transfer import (KIND_LABELS, KINDS, Options, Result, Transfer, classify,
                       default_output_dir, estimate_memory, output_name, output_size)

__version__ = "1.0.1"

__all__ = [
    "Cancelled", "TransferError", "IMAGE_EXTENSIONS", "image_info", "load_image", "save_image",
    "Mesh", "MeshPart", "load_fbx_parts", "Setup", "TextureEntry", "available_memory",
    "default_workers", "KIND_LABELS", "KINDS", "Options", "Result", "Transfer", "classify",
    "default_output_dir", "estimate_memory", "output_name", "output_size",
]
