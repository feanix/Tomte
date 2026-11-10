"""
Per-corner tangent frames (T, B, N), modelled on MikkTSpace.

Like MikkTSpace, each corner's tangent is the angle-weighted average of the
UV-derived tangents of the triangles around it, projected onto the normal.
Corners are only averaged together when they share position, UV, normal and
UV handedness, so frames split at UV seams and mirrored islands.

Known differences from MikkTSpace (measured against Blender on the sample
mesh with identical triangles: mean 0.26 degrees, but ~0.5% of corners differ
by more than 5 degrees):
  * corners are grouped by equal (rounded) values rather than by walking
    edge-connected triangle fans;
  * corner angles use the raw triangle edges, not edges projected onto the
    tangent plane;
  * degenerate triangles are skipped instead of borrowing from neighbours;
  * quads are fan-triangulated (MikkTSpace splits along the shorter UV diagonal).
"""

import numpy as np

UV_AREA_EPS = 1e-14       # triangles with less UV area than this have no usable tangent
GROUP_UV_SCALE = 1e6      # UVs equal to ~1e-6 count as the same when grouping corners
GROUP_NORMAL_SCALE = 1e4  # normals equal to ~1e-4 count as the same


def corner_tangent_frames(mesh):
    """Tangent frames for every corner of `mesh`.

    Args:
        mesh: a `uvtransfer.mesh.Mesh`.

    Returns:
        (T, B, N): three (C, 3) arrays. T and N are unit length and
        perpendicular; B = cross(N, T) * handedness.
    """
    tri_t, tri_b, handedness, valid = _triangle_tangents(mesh)
    weights = _corner_angles(mesh) * valid[:, None]
    group = _smoothing_groups(mesh, handedness)

    # angle-weighted sums of each triangle's tangent (projected onto the corner's
    # normal) and bitangent, per group of corners that should share a frame
    n_groups = group.max() + 1
    sum_t = np.zeros((n_groups, 3))
    sum_b = np.zeros((n_groups, 3))
    for k in range(3):
        corner = mesh.tris[:, k]
        n = mesh.N[corner]
        t_proj = _normalised(tri_t - n * _dot(tri_t, n)[:, None])
        np.add.at(sum_t, group[corner], t_proj * weights[:, k:k + 1])
        np.add.at(sum_b, group[corner], tri_b * weights[:, k:k + 1])

    N = mesh.N
    T = sum_t[group]
    T = T - N * _dot(T, N)[:, None]
    _fill_missing_tangents(T, N)
    T /= np.linalg.norm(T, axis=1)[:, None]
    sign = np.sign(_dot(np.cross(N, T), sum_b[group]))
    sign[sign == 0] = 1
    B = np.cross(N, T) * sign[:, None]
    return T, B, N


def _triangle_tangents(mesh):
    """Unit tangent and bitangent of each triangle (directions of +u and +v on the surface).

    Returns:
        (t, b, handedness, valid): t and b are (T, 3); handedness is +1/-1 for
        normal/mirrored UVs; valid is False where the triangle has no UV area.
    """
    p = mesh.P[mesh.corner_vertex[mesh.tris]]     # (T, 3 corners, xyz)
    uv = mesh.uv[mesh.tris]                       # (T, 3 corners, uv)
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]   # 2 x signed UV area
    valid = np.abs(det) > UV_AREA_EPS
    inv = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)[:, None]
    t = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * inv
    b = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * inv
    return _normalised(t), _normalised(b), np.sign(det), valid


def _corner_angles(mesh):
    """(T, 3) interior angle at each triangle corner, in radians."""
    p = mesh.P[mesh.corner_vertex[mesh.tris]]
    angles = np.zeros((len(mesh.tris), 3))
    for k in range(3):
        a = p[:, (k + 1) % 3] - p[:, k]
        c = p[:, (k + 2) % 3] - p[:, k]
        cos = _dot(a, c) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1), 1e-20)
        angles[:, k] = np.arccos(np.clip(cos, -1, 1))
    return angles


def _smoothing_groups(mesh, handedness):
    """Group id per corner: corners with the same vertex, UV, normal and handedness share one."""
    corner_handedness = np.zeros(len(mesh.uv))
    for k in range(3):
        # a corner shared by several triangles of one polygon takes the last one's value;
        # they agree unless the polygon is folded in UV space
        corner_handedness[mesh.tris[:, k]] = handedness
    key = np.column_stack([
        mesh.corner_vertex,
        np.round(mesh.uv * GROUP_UV_SCALE).astype(np.int64),
        np.round(mesh.N * GROUP_NORMAL_SCALE).astype(np.int64),
        corner_handedness.astype(np.int64),
    ])
    _, group = np.unique(key, axis=0, return_inverse=True)
    return group.ravel()


def _fill_missing_tangents(T, N):
    """Where no triangle gave a usable tangent, pick any direction perpendicular to N (in place)."""
    bad = np.linalg.norm(T, axis=1) < 1e-12
    if bad.any():
        helper = np.where(np.abs(N[bad, 0:1]) < 0.9, [[1, 0, 0]], [[0, 1, 0]])
        T[bad] = np.cross(N[bad], helper)


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _normalised(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-20)
