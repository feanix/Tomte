"""Edge padding: grow texture islands outward so mipmaps and filtering don't pick up the background."""

import numpy as np

_NEIGHBOURS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def dilate(img, mask, iterations, progress=None):
    """Grow covered pixels outward by one ring per iteration.

    Each new pixel becomes the average of its already-covered 8 neighbours.
    Only the ring of pixels on the edge is touched each step, so the cost
    scales with the padding width, not the image size.

    Args:
        img: HxWxC float array. Modified in place and returned.
        mask: HxW bool, pixels that already have values.
        iterations: rings to grow; negative means until the image is full.
        progress: optional callback(fraction 0..1); may raise to cancel.

    Returns:
        (img, filled_mask)
    """
    H, W, ch = img.shape
    flat = img.reshape(-1, ch)                    # a view when img is contiguous
    filled = mask.ravel().copy()
    front = _initial_front(mask)
    total = iterations if iterations > 0 else max(1, int((~mask).sum()))
    grown_total = 0
    it = 0
    while front.size and (iterations < 0 or it < iterations):
        it += 1
        y, x = np.divmod(front, W)
        acc = np.zeros((front.size, ch), np.float32)
        cnt = np.zeros(front.size, np.float32)
        neighbours = []
        for dy, dx in _NEIGHBOURS:
            ny, nx = y + dy, x + dx
            inside = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            idx = np.where(inside, ny * W + nx, 0)
            w = (inside & filled[idx]).astype(np.float32)
            acc += flat[idx] * w[:, None]
            cnt += w
            neighbours.append(idx[inside])
        grow = cnt > 0
        flat[front[grow]] = acc[grow] / cnt[grow][:, None]
        filled[front[grow]] = True
        grown_total += int(grow.sum())
        # next ring: unfilled neighbours of this ring
        candidates = np.unique(np.concatenate(neighbours))
        front = candidates[~filled[candidates]]
        if progress:
            progress(it / total if iterations > 0 else grown_total / total)
    img = flat.reshape(H, W, ch)
    return img, filled.reshape(H, W)


def _initial_front(mask):
    """Flat indices of unfilled pixels touching a filled one."""
    H, W = mask.shape
    near = np.zeros_like(mask)
    for dy, dx in _NEIGHBOURS:
        near[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] |= \
            mask[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return np.flatnonzero(near & ~mask)
