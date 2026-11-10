"""
Tangent-space normal map helpers: detecting DirectX vs OpenGL encoding and
re-expressing normals from one set of tangent frames in another.

Pixels store n * 0.5 + 0.5. OpenGL ("Y+") maps have green pointing along +V,
DirectX ("Y-", Unreal) maps along -V.
"""

import numpy as np

# Detection: how much lower the curl energy of one reading must be to call it
DETECT_RATIO = 1.15
_MIN_Z = 0.2             # clamp nz when converting normals to slopes (avoids huge slopes)
_FLAT_ENOUGH = 0.3       # ignore steep texels (nz below this) when measuring curl
_EDGE_MARGIN_PX = 3      # stay this far inside UV islands so seams don't count


def curl_energies(nmap, mask):
    """Curl energy of a normal map read as OpenGL and as DirectX.

    A normal map baked from a height field is a gradient field, so its curl is
    ~0 - but only when green is read with the right sign. The reading with
    much less curl is the right one.

    Args:
        nmap: HxWx3+ float array, 0..1 encoded normals.
        mask: HxW bool, texels covered by UVs.

    Returns:
        (energy_gl, energy_dx)
    """
    n = nmap[..., :3] * 2 - 1
    z = np.clip(n[..., 2], _MIN_Z, None)
    gx = -n[..., 0] / z
    gy_gl = -n[..., 1] / z
    m = mask.copy()
    for _ in range(_EDGE_MARGIN_PX):
        m[1:, :] &= m[:-1, :]
        m[:-1, :] &= m[1:, :]
        m[:, 1:] &= m[:, :-1]
        m[:, :-1] &= m[:, 1:]
    m &= n[..., 2] > _FLAT_ENOUGH

    def energy(gy):
        dgy_dcol = gy[1:-1, 2:] - gy[1:-1, :-2]
        dgx_drow = gx[2:, 1:-1] - gx[:-2, 1:-1]
        curl = dgy_dcol + dgx_drow                # rows run down the image (-v), hence +
        return float(np.sum((curl ** 2)[m[1:-1, 1:-1]]))

    return energy(gy_gl), energy(-gy_gl)


def detect_convention(nmap, mask):
    """Guess whether a normal map is DirectX or OpenGL.

    Returns:
        ("dx" or "gl", note for the user). Falls back to "dx" when unsure.
    """
    e_gl, e_dx = curl_energies(nmap, mask)
    ratio = e_dx / max(e_gl, 1e-20)
    if ratio > DETECT_RATIO:
        return "gl", "OpenGL normal map (auto-detected)"
    if ratio < 1 / DETECT_RATIO:
        return "dx", "DirectX normal map (auto-detected)"
    return "dx", "Normal format couldn't be detected; assumed DirectX (Unreal)"


def reorient(rgb, bary, src_frames, src_corners, dst_frames, dst_corners, directx):
    """Re-express encoded tangent-space normals from source frames in target frames.

    Each sample lies inside one triangle; its frame is the barycentric blend of
    the frames at that triangle's three corners, as a renderer would interpolate.

    Args:
        rgb: (n, 3) encoded normals (0..1) sampled from the source texture.
        bary: (n, 3) barycentric weights of each sample in its triangle.
        src_frames, dst_frames: (T, B, N) per-corner frames of each mesh.
        src_corners, dst_corners: (n, 3) corner indices of each sample's triangle.
        directx: True if the texture uses the DirectX (green-down) convention.

    Returns:
        (n, 3) float32 unit normals in -1..1 (not yet encoded).
    """
    def blend(F, corners):
        return np.einsum("nk,nkj->nj", bary, F[corners])

    n = rgb.astype(np.float64) * 2 - 1
    if directx:
        n[:, 1] = -n[:, 1]
    sT, sB, sN = src_frames
    world = n[:, 0:1] * blend(sT, src_corners) + n[:, 1:2] * blend(sB, src_corners) \
        + n[:, 2:3] * blend(sN, src_corners)

    # Solve [T B N] x = world. Rows of the inverse are cross products / det; the
    # result is normalised afterwards, so only the sign of det matters.
    dT, dB, dN = (blend(F, dst_corners) for F in dst_frames)
    rows = (np.cross(dB, dN), np.cross(dN, dT), np.cross(dT, dB))
    det = np.einsum("ij,ij->i", dT, rows[0])
    nt = np.column_stack([np.einsum("ij,ij->i", r, world) for r in rows])
    nt *= np.where(det < 0, -1.0, 1.0)[:, None]
    degenerate = np.abs(det) < 1e-8
    nt[degenerate] = n[degenerate]                # no usable frame: keep the normal as it was
    nt /= np.maximum(np.linalg.norm(nt, axis=1, keepdims=True), 1e-20)
    if directx:
        nt[:, 1] = -nt[:, 1]
    return nt.astype(np.float32)
