"""
Meshes read from FBX files.

Data is kept per polygon *corner* (FBX "polygon vertex"): a vertex shared by
four quads has four corners, each with its own UV and normal. Polygons are
fan-triangulated; triangles refer to corners, not vertices.
"""

import os
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .errors import TransferError
from .fbx import as_str, read_fbx


@dataclass
class MeshPart:
    """One polygon mesh from a file. C = corners, V = vertices, T = triangles."""

    name: str
    P: np.ndarray               # (V, 3) vertex positions
    corner_vertex: np.ndarray   # (C,)   vertex index of each corner
    next_corner: np.ndarray     # (C,)   next corner around the same polygon (for edges)
    tris: np.ndarray            # (T, 3) corner indices of each triangle
    uv: np.ndarray              # (C, 2)
    N: np.ndarray               # (C, 3) normals, from the file or computed


class Mesh:
    """Triangulated mesh with per-corner data, merged from one or more `MeshPart`s.

    Attributes mirror `MeshPart` (P, corner_vertex, next_corner, tris, uv, N)
    with indices offset so all parts live in one set of arrays. Normals are
    normalised.
    """

    def __init__(self, parts, path=""):
        self.parts = parts
        self.path = path
        v_off = c_off = 0
        P, cv, nxt, tris, uv, N = [], [], [], [], [], []
        for p in parts:
            P.append(p.P)
            cv.append(p.corner_vertex + v_off)
            nxt.append(p.next_corner + c_off)
            tris.append(p.tris + c_off)
            uv.append(p.uv)
            N.append(p.N)
            v_off += len(p.P)
            c_off += len(p.uv)
        self.P = np.concatenate(P)
        self.corner_vertex = np.concatenate(cv)
        self.next_corner = np.concatenate(nxt)
        self.tris = np.concatenate(tris)
        self.uv = np.concatenate(uv)
        self.N = _normalised(np.concatenate(N))

    @classmethod
    def from_fbx(cls, path, uv_set=0):
        """Load every mesh in an FBX file. See `load_fbx_parts`."""
        return cls(load_fbx_parts(path, uv_set), path)

    def uv_edges(self):
        """Polygon edges in UV space as an (E, 2, 2) array: [edge, end, (u, v)].

        An edge shared by two polygons with matching UVs is returned once; on a
        UV seam each side has its own UVs, so both are returned.
        """
        a, b = self.uv, self.uv[self.next_corner]
        ka = np.round(a * 1e6).astype(np.int64)
        kb = np.round(b * 1e6).astype(np.int64)
        swap = (ka[:, 0] > kb[:, 0]) | ((ka[:, 0] == kb[:, 0]) & (ka[:, 1] > kb[:, 1]))
        lo = np.where(swap[:, None], kb, ka)
        hi = np.where(swap[:, None], ka, kb)
        _, first = np.unique(np.column_stack([lo, hi]), axis=0, return_index=True)
        first.sort()
        return np.stack([a[first], b[first]], axis=1)


def load_fbx_parts(path, uv_set=0):
    """Read every polygon mesh in an FBX file, in file order.

    Args:
        path: FBX file (ASCII 6.x/7.x or binary).
        uv_set: which UV channel to read (0 = first).

    Returns:
        List of `MeshPart`, named after their FBX models where possible.

    Raises:
        TransferError: the file can't be read, is damaged, has no meshes, or a
            mesh lacks polygons or the requested UV channel.
    """
    label = os.path.basename(path)
    try:
        return _read_parts(read_fbx(path), uv_set, label)
    except OSError as e:
        raise TransferError(f"Couldn't open {label}: {e.strerror or e}") from e
    except MemoryError as e:
        raise TransferError(f"{label} is too large to read with the memory available") from e
    except _MALFORMED as e:
        raise TransferError(f"{label} isn't a readable FBX file ({e})") from e


# Ways a damaged or hostile file can make parsing fail (TransferError passes through).
_MALFORMED = (ValueError, IndexError, KeyError, TypeError, OverflowError, RecursionError,
              UnicodeDecodeError, struct.error, zlib.error)


def _read_parts(root, uv_set, label):
    """The `MeshPart`s in a parsed FBX tree (see `load_fbx_parts`)."""
    objects = root.find("Objects")
    if objects is None:
        raise TransferError(f"{label} has no objects (is it an FBX file?)")

    names = _geometry_names(root, objects)
    parts = []
    for node in objects.children:
        # FBX 7 keeps mesh data in Geometry records; FBX 6 puts it straight in Model records.
        if node.name not in ("Geometry", "Model") or not node.props \
                or as_str(node.props[-1]) != "Mesh" or node.find("Vertices") is None:
            continue
        name = names.get(id(node)) or f"Mesh {len(parts) + 1}"
        parts.append(_read_geometry(node, name, uv_set, label))
    if not parts:
        raise TransferError(f"{label} contains no polygon meshes")
    return parts


def _object_name(prop):
    """Strip the class from an FBX object name ("Name\\x00\\x01Class" or "Class::Name")."""
    s = as_str(prop)
    if "\x00\x01" in s:
        return s.split("\x00\x01", 1)[0]
    if "::" in s:
        return s.split("::", 1)[1]
    return s


def _geometry_names(root, objects):
    """Map id(geometry node) -> display name.

    Geometry names are often blank, so prefer the name of the Model the
    geometry is connected to.
    """
    model_names = {m.props[0]: _object_name(m.props[1])
                   for m in objects.find_all("Model") if len(m.props) >= 3 and _is_id(m.props[0])}
    parent = {}
    conns = root.find("Connections")
    if conns is not None:
        for c in conns.find_all("C"):
            if len(c.props) < 3 or as_str(c.props[0]) != "OO":
                continue
            if _is_id(c.props[1]) and _is_id(c.props[2]):
                parent.setdefault(c.props[1], c.props[2])
    names = {}
    for node in objects.children:
        if node.name == "Geometry" and len(node.props) >= 3 and _is_id(node.props[0]):
            names[id(node)] = model_names.get(parent.get(node.props[0]), "") \
                or _object_name(node.props[1])
        elif node.name == "Model" and len(node.props) >= 2:
            names[id(node)] = _object_name(node.props[-2])
    return names


def _is_id(value):
    """True for a value usable as an object ID (damaged files can put arrays there)."""
    return isinstance(value, (int, str, bytes))


@dataclass
class _Polygons:
    corner_vertex: np.ndarray   # (C,)
    starts: np.ndarray          # (F,) first corner of each polygon
    sizes: np.ndarray           # (F,) corners per polygon
    corner_poly: np.ndarray     # (C,) polygon of each corner
    next_corner: np.ndarray     # (C,)


def _polygons(pvi):
    """Decode FBX PolygonVertexIndex (last corner of each polygon stored as ~index)."""
    ends = pvi < 0
    corner_vertex = np.where(ends, ~pvi, pvi)
    ends[-1] = True                               # tolerate a missing final marker
    end_pos = np.flatnonzero(ends)
    starts = np.concatenate([[0], end_pos[:-1] + 1])
    sizes = end_pos + 1 - starts
    next_corner = np.arange(len(pvi)) + 1
    next_corner[end_pos] = starts
    corner_poly = np.repeat(np.arange(len(sizes)), sizes)
    return _Polygons(corner_vertex, starts, sizes, corner_poly, next_corner)


def _fan_triangulate(starts, sizes):
    """Triangles (s, s+k, s+k+1) for k = 1 .. size-2 of every polygon, as corner indices."""
    n_tri = np.maximum(sizes - 2, 0)
    tri_poly = np.repeat(np.arange(len(sizes)), n_tri)
    k = np.arange(n_tri.sum()) - np.repeat(np.cumsum(n_tri) - n_tri, n_tri) + 1
    s0 = starts[tri_poly]
    return np.column_stack([s0, s0 + k, s0 + k + 1]).astype(np.int64)


def _read_geometry(g, name, uv_set, label):
    P = g.array("Vertices")
    pvi = g.array("PolygonVertexIndex")
    if P is None or pvi is None or P.size < 3 or pvi.size < 3:
        raise TransferError(f"{label}: mesh '{name}' has no polygons")
    P = P.astype(np.float64)[: P.size // 3 * 3].reshape(-1, 3)
    if not np.isfinite(P).all():
        raise TransferError(f"{label}: mesh '{name}' has vertex positions that are invalid numbers")
    if pvi.dtype.kind == "f" and not (np.isfinite(pvi).all() and np.abs(pvi).max() < 2**31):
        raise TransferError(f"{label}: mesh '{name}' has invalid polygon data")
    polys = _polygons(pvi.astype(np.int64))
    if polys.corner_vertex.max() >= len(P):
        raise TransferError(f"{label}: mesh '{name}' references missing vertices")
    tris = _fan_triangulate(polys.starts, polys.sizes)
    uv = _read_uvs(g, polys, uv_set, f"{label}: mesh '{name}'")
    N = _read_normals(g, polys)
    if N is None:
        N = _smooth_normals(P, polys.corner_vertex, tris)
    return MeshPart(name, P, polys.corner_vertex, polys.next_corner, tris, uv, N)


def _read_uvs(g, polys, uv_set, where):
    layers = g.find_all("LayerElementUV")
    layers.sort(key=lambda layer: layer.props[0] if layer.props and isinstance(layer.props[0], int) else 0)
    if not layers:
        raise TransferError(f"{where} has no UVs")
    if uv_set >= len(layers):
        raise TransferError(f"{where} has {len(layers)} UV channel(s), "
                            f"but UV channel {uv_set} was requested")
    uv = _expand_layer(layers[uv_set], "UV", "UVIndex", polys, 2)
    if uv is None:
        raise TransferError(f"{where}: UV channel {uv_set} is empty")
    if not np.isfinite(uv).all():
        raise TransferError(f"{where}: UV channel {uv_set} has UVs that are invalid numbers")
    return uv


def _read_normals(g, polys):
    """Per-corner normals from the file, or None if missing or unusable."""
    layer = g.find("LayerElementNormal")
    if layer is None:
        return None
    try:
        N = _expand_layer(layer, "Normals", "NormalsIndex", polys, 3)
    except TransferError:
        return None                               # unusual mapping: caller computes normals
    if N is None or not np.isfinite(N).all():
        return None
    return N


def _expand_layer(layer, data_name, index_name, polys, n_cols):
    """Expand an FBX LayerElement to one value per corner (None if it has no data).

    Raises:
        TransferError: unsupported mapping, or an index list that is too short.
    """
    vals = layer.array(data_name)
    if vals is None or vals.size < n_cols:
        return None
    vals = vals.astype(np.float64)[: vals.size // n_cols * n_cols].reshape(-1, n_cols)
    mapping = layer.string("MappingInformationType")
    ref = layer.string("ReferenceInformationType")
    n_corners = len(polys.corner_vertex)
    if mapping == "ByPolygonVertex":
        sel = np.arange(n_corners)
    elif mapping in ("ByVertex", "ByVertice", "ByControlPoint"):
        sel = polys.corner_vertex
    elif mapping == "ByPolygon":
        sel = polys.corner_poly
    elif mapping == "AllSame":
        sel = np.zeros(n_corners, np.int64)
    else:
        raise TransferError(f"unsupported {data_name} mapping {mapping!r}")
    idx = layer.array(index_name)
    if ref in ("IndexToDirect", "Index") and idx is not None:
        if sel.max(initial=0) >= len(idx):
            raise TransferError(f"{data_name} index list is too short")
        sel = idx.astype(np.int64)[sel]
    # some exporters write -1 for "unassigned"; clamp rather than crash
    return vals[np.clip(sel, 0, len(vals) - 1)]


def _smooth_normals(P, corner_vertex, tris):
    """Area-weighted vertex normals, expanded to corners."""
    acc = np.zeros_like(P)
    a, b, c = (P[corner_vertex[tris[:, k]]] for k in range(3))
    face_n = np.cross(b - a, c - a)               # length = 2 x area, so this weights by area
    for k in range(3):
        np.add.at(acc, corner_vertex[tris[:, k]], face_n)
    return _normalised(acc[corner_vertex])


def _normalised(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-20)
