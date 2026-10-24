"""Discretised hazard grid: priority map, hazard map, no-fly zones, targets."""
import numpy as np
from scipy.spatial import cKDTree


def _blobs(xs, ys, centres, sigmas, weights):
    z = np.zeros_like(xs, dtype=float)
    for (cx, cy), s, w in zip(centres, sigmas, weights):
        z += w * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * s ** 2))
    return z / z.max()


class Environment:
    """Grid index convention: array[y, x]; flat cell index = y * n + x."""

    def __init__(self, cfg):
        self.cfg = cfg
        n = self.n = cfg.grid_size
        rng = np.random.default_rng(cfg.env_seed)
        ys, xs = np.mgrid[0:n, 0:n]
        self.cell_x = xs.ravel().astype(np.float32)
        self.cell_y = ys.ravel().astype(np.float32)
        self.cell_xy = np.column_stack([self.cell_x, self.cell_y])
        self.base = np.array([1.0, 1.0])

        # High-priority disaster zones (e.g. collapsed buildings, flood pockets)
        pc = rng.uniform(8, n - 8, size=(4, 2))
        self.priority = _blobs(xs, ys, pc, rng.uniform(4, 6, 4), rng.uniform(0.6, 1.0, 4))
        # Hazards (wind / radiation). One sits next to a priority zone -> real trade-off.
        hc = np.vstack([pc[0] + [6, 4], rng.uniform(8, n - 8, size=(2, 2))])
        self.hazard = _blobs(xs, ys, hc, rng.uniform(5, 7, 3), rng.uniform(0.7, 1.0, 3))

        # No-fly zones as rectangles (fractions of the grid)
        self.nofly = np.zeros((n, n), dtype=bool)
        for x0, y0, x1, y1 in [(.40, .30, .52, .48), (.65, .70, .80, .82), (.15, .55, .25, .65)]:
            self.nofly[int(y0 * n):int(y1 * n), int(x0 * n):int(x1 * n)] = True

        # Hidden targets, more likely inside high-priority cells, never in no-fly zones
        w = (self.priority ** 2) * (~self.nofly)
        w = w.ravel() / w.sum()
        idx = rng.choice(n * n, size=cfg.n_targets, replace=False, p=w)
        self.targets = self.cell_xy[idx].astype(np.float32)

        free = ~self.nofly.ravel()
        self._free_xy = self.cell_xy[free]
        self._free_tree = cKDTree(self._free_xy)
        self.priority_flat = self.priority.ravel().astype(np.float32)
        self.total_priority = float(self.priority_flat[free].sum())

    # ---- helpers ----
    def cell_index(self, pts):
        ix = np.clip(np.rint(pts[:, 0]), 0, self.n - 1).astype(int)
        iy = np.clip(np.rint(pts[:, 1]), 0, self.n - 1).astype(int)
        return ix, iy

    def in_nofly(self, pts):
        ix, iy = self.cell_index(pts)
        return self.nofly[iy, ix]

    def nearest_free(self, pts):
        """Repair mechanism: snap points to the nearest allowed cell."""
        _, i = self._free_tree.query(pts)
        return self._free_xy[i]
