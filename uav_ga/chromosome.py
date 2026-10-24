"""Chromosome = real-valued tensor  genes[drone, waypoint, 4]
   gene = [x, y, speed, sensor_power]   (leg that ENDS at this waypoint uses
   its speed & power; the return-to-base leg re-uses the last waypoint profile)."""
import numpy as np


class Chromosome:
    __slots__ = ("genes", "fitness", "metrics")

    def __init__(self, genes):
        self.genes = genes
        self.fitness = None
        self.metrics = None

    def copy(self):
        c = Chromosome(self.genes.copy())
        c.fitness, c.metrics = self.fitness, self.metrics
        return c

    # ---------- construction ----------
    @staticmethod
    def random(cfg, env, rng, informed=False):
        D, K, n = cfg.n_uav, cfg.n_waypoints, cfg.grid_size
        g = np.empty((D, K, 4))
        if informed:   # waypoints sampled where priority is high, then ordered by angle
            p = env.priority_flat / env.priority_flat.sum()
            idx = rng.choice(len(p), size=(D, K), p=p)
            g[..., 0] = env.cell_x[idx] + rng.normal(0, 1, (D, K))
            g[..., 1] = env.cell_y[idx] + rng.normal(0, 1, (D, K))
            for d in range(D):
                ang = np.arctan2(g[d, :, 1] - env.base[1], g[d, :, 0] - env.base[0])
                g[d] = g[d][np.argsort(ang)]
        else:          # pure uniform
            g[..., 0] = rng.uniform(0, n - 1, (D, K))
            g[..., 1] = rng.uniform(0, n - 1, (D, K))
        g[..., 2] = rng.uniform(cfg.v_min, cfg.v_max, (D, K))
        g[..., 3] = rng.uniform(cfg.p_min, cfg.p_max, (D, K))
        c = Chromosome(g)
        c.repair(cfg, env)
        return c

    # ---------- constraint handling (repair) ----------
    def repair(self, cfg, env):
        g = self.genes
        g[..., 0:2] = np.clip(g[..., 0:2], 0, cfg.grid_size - 1)
        g[..., 2] = np.clip(g[..., 2], cfg.v_min, cfg.v_max)
        g[..., 3] = np.clip(g[..., 3], cfg.p_min, cfg.p_max)
        flat = g[..., 0:2].reshape(-1, 2)
        bad = env.in_nofly(flat)
        if bad.any():                       # push waypoints out of no-fly zones
            flat[bad] = env.nearest_free(flat[bad])
            g[..., 0:2] = flat.reshape(g.shape[0], g.shape[1], 2)
        self.fitness = None
        self.metrics = None
        return self
