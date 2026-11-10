"""
Rasterising triangles in texture space.

Every output pixel is split into ss x ss sub-samples. A sub-sample belongs to
a triangle when its centre lies inside it (shared edges count for both sides).
Work is split into independent batches that can run on a thread pool.
"""

import numpy as np

BATCH_SAMPLES = 262_144   # candidate sub-samples per batch: big enough to vectorise well,
                          # small enough that several batches fit in memory at once
TILE_RANGE = (-4, 5)      # UVs further than this many tiles outside 0..1 are clipped away
EDGE_EPS = 1e-7           # barycentric tolerance so samples on shared edges aren't lost


class RasterPlan:
    """Which sub-samples to test against which triangle, grouped into batches.

    Each triangle contributes its bounding box of sub-samples ("box"). Very
    tall boxes are cut into row bands so no single box exceeds a batch. Boxes
    are sorted by row so each batch touches a compact band of the image.

    Args:
        tri_uv: (T, 3, 2) UVs of each triangle's corners.
        W, H: output size in pixels.
        ss: sub-samples per pixel along each axis.
        batch_samples: target number of candidate sub-samples per batch.
        edge_tolerance: barycentric slack; positive counts samples on an edge
            for both triangles, negative counts them for neither.
    """

    def __init__(self, tri_uv, W, H, ss, batch_samples=BATCH_SAMPLES, edge_tolerance=EDGE_EPS):
        self.W, self.H = W, H
        self.edge_tolerance = edge_tolerance
        self.ss = ss = max(1, int(ss))
        self.bounds = np.zeros(1, np.int64)       # batch k = boxes bounds[k]:bounds[k+1]
        x, y, finite = self._sample_coords(tri_uv)
        i0, i1, j0, j1 = self._sample_boxes(x, y)

        # twice the signed area; ~0 means a degenerate (zero-area) triangle
        area2 = (y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2])
        keep = finite & (i1 >= i0) & (j1 >= j0) & (np.abs(area2) > 1e-12)
        self.tri = tri = np.flatnonzero(keep)
        if tri.size == 0:
            return
        self.coef = self._barycentric_coefficients(x[tri], y[tri], area2[tri])
        self._make_batches(i0[tri], i1[tri], j0[tri], j1[tri], batch_samples)

    def _sample_coords(self, tri_uv):
        """Triangle corners in sub-sample units (x right, y down), and which are finite."""
        x = tri_uv[..., 0] * (self.W * self.ss)
        y = (1.0 - tri_uv[..., 1]) * (self.H * self.ss)
        finite = np.isfinite(x).all(1) & np.isfinite(y).all(1)
        return np.where(finite[:, None], x, 0.0), np.where(finite[:, None], y, 0.0), finite

    def _sample_boxes(self, x, y):
        """Inclusive sub-sample index ranges [i0, i1] x [j0, j1] around each triangle.

        Sub-sample i has its centre at i + 0.5.
        """
        GW, GH = self.W * self.ss, self.H * self.ss
        lo, hi = TILE_RANGE
        i0 = np.clip(np.ceil(x.min(1) - 0.5), lo * GW, hi * GW).astype(np.int64)
        i1 = np.clip(np.floor(x.max(1) - 0.5), lo * GW - 1, hi * GW - 1).astype(np.int64)
        j0 = np.clip(np.ceil(y.min(1) - 0.5), lo * GH, hi * GH).astype(np.int64)
        j1 = np.clip(np.floor(y.max(1) - 0.5), lo * GH - 1, hi * GH - 1).astype(np.int64)
        return i0, i1, j0, j1

    @staticmethod
    def _barycentric_coefficients(x, y, area2):
        """(n, 6) rows [a0, b0, a1, b1, x2, y2] so that, with X = px - x2 and Y = py - y2,
        weight0 = a0*X + b0*Y, weight1 = a1*X + b1*Y and weight2 = 1 - weight0 - weight1."""
        return np.column_stack([
            (y[:, 1] - y[:, 2]) / area2, (x[:, 2] - x[:, 1]) / area2,
            (y[:, 2] - y[:, 0]) / area2, (x[:, 0] - x[:, 2]) / area2,
            x[:, 2], y[:, 2],
        ])

    def _make_batches(self, i0, i1, j0, j1, batch_samples):
        w = i1 - i0 + 1
        h = j1 - j0 + 1
        # cut tall boxes into bands of at most batch_samples sub-samples
        rows = np.maximum(1, batch_samples // w)
        n_bands = -(-h // rows)
        box_tri = np.repeat(np.arange(len(w)), n_bands)
        band = np.arange(n_bands.sum()) - np.repeat(np.cumsum(n_bands) - n_bands, n_bands)
        box_j0 = j0[box_tri] + band * rows[box_tri]
        box_h = np.minimum(rows[box_tri], j1[box_tri] + 1 - box_j0)

        order = np.argsort(box_j0, kind="stable")
        self.box_tri = box_tri[order]             # index into self.tri / self.coef
        self.box_j0 = box_j0[order]
        self.box_w = w[self.box_tri]
        self.box_i0 = i0[self.box_tri]
        self.box_count = self.box_w * box_h[order]
        self.box_start = np.cumsum(self.box_count) - self.box_count
        self.total = float(self.box_count.sum())
        bucket = self.box_start // batch_samples
        cuts = np.flatnonzero(np.diff(bucket)) + 1
        self.bounds = np.concatenate([[0], cuts, [len(self.box_count)]])

    def __len__(self):
        return len(self.bounds) - 1

    def done_after(self, k):
        """Fraction of all work finished once batches 0..k are done."""
        last = self.bounds[k + 1] - 1
        return float(self.box_start[last] + self.box_count[last]) / self.total

    def batch(self, k):
        """Sub-samples of batch k that fall inside their triangle.

        Returns:
            (tri_index, pixel_index, bary) - arrays of length n: triangle index
            into the input `tri_uv`, flat output pixel index (y * W + x, UVs
            wrapped into 0..1), and (n, 3) barycentric weights. None if empty.
        """
        a, b = self.bounds[k], self.bounds[k + 1]
        counts = self.box_count[a:b]
        box = np.repeat(np.arange(b - a), counts)
        local = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
        width = self.box_w[a:b][box]
        sx = self.box_i0[a:b][box] + local % width
        sy = self.box_j0[a:b][box] + local // width
        tri = self.box_tri[a:b][box]
        c = self.coef[tri]
        X = sx + 0.5 - c[:, 4]
        Y = sy + 0.5 - c[:, 5]
        w0 = c[:, 0] * X + c[:, 1] * Y
        w1 = c[:, 2] * X + c[:, 3] * Y
        w2 = 1.0 - w0 - w1
        tol = self.edge_tolerance
        inside = (w0 >= -tol) & (w1 >= -tol) & (w2 >= -tol)
        if not inside.any():
            return None
        px = (sx[inside] // self.ss) % self.W
        py = (sy[inside] // self.ss) % self.H
        bary = np.column_stack([w0[inside], w1[inside], w2[inside]])
        return self.tri[tri[inside]], py * self.W + px, bary


def sum_by_pixel(pixel, values):
    """Add up per-sample values per output pixel.

    Memory scales with the number of samples, not with the spread of pixel
    indices (a batch can touch both the top and bottom rows when UVs wrap).

    Args:
        pixel: (n,) flat pixel index per sample.
        values: (n, C) values per sample.

    Returns:
        (pixels, counts, sums): distinct pixel indices (k,), samples per pixel
        (k,) and per-channel sums (k, C).
    """
    lo, hi = int(pixel.min()), int(pixel.max()) + 1
    if hi - lo <= 4 * len(pixel):                 # compact: index directly
        local = pixel - lo
        counts = np.bincount(local, minlength=hi - lo)
        used = np.flatnonzero(counts)
        sums = np.column_stack([np.bincount(local, weights=values[:, c], minlength=hi - lo)[used]
                                for c in range(values.shape[1])])
        return used + lo, counts[used], sums
    pixels, local = np.unique(pixel, return_inverse=True)
    local = local.ravel()
    counts = np.bincount(local)
    sums = np.column_stack([np.bincount(local, weights=values[:, c], minlength=len(pixels))
                            for c in range(values.shape[1])])
    return pixels, counts, sums


def run_batches(plan, fn, workers, on_result):
    """Run fn(k) for every batch on a thread pool and pass results to on_result(k, result) in order.

    At most 2 x workers batches are queued at once to bound memory. If
    on_result raises (e.g. Cancelled), queued batches are dropped and the
    exception propagates after running batches finish.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    if workers <= 1 or len(plan) <= 1:
        for k in range(len(plan)):
            on_result(k, fn(k))
        return
    with ThreadPoolExecutor(workers) as pool:
        pending = deque()
        k_next = 0
        try:
            while k_next < len(plan) or pending:
                while k_next < len(plan) and len(pending) < 2 * workers:
                    pending.append((k_next, pool.submit(fn, k_next)))
                    k_next += 1
                k, future = pending.popleft()
                on_result(k, future.result())
        finally:
            for _, future in pending:
                future.cancel()


def coverage_counts(tri_uv, W, H, edge_tolerance=EDGE_EPS):
    """Number of triangles covering each pixel centre, as an HxW int array.

    With a negative `edge_tolerance`, centres exactly on a shared edge count
    for neither triangle, so only true overlaps give counts above 1.
    """
    counts = np.zeros(H * W, np.int64)
    plan = RasterPlan(tri_uv, W, H, 1, batch_samples=1_000_000, edge_tolerance=edge_tolerance)
    for k in range(len(plan)):
        r = plan.batch(k)
        if r is not None:
            np.add.at(counts, r[1], 1)
    return counts.reshape(H, W)


def coverage_mask(tri_uv, W, H):
    """HxW bool: pixel centres covered by at least one triangle."""
    return coverage_counts(tri_uv, W, H) > 0
