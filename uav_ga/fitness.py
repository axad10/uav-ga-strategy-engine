"""Multi-objective fitness (weighted-sum scalarisation) + constraint penalties."""
import numpy as np


class FitnessEvaluator:
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.n_evals = 0
        self.t_ref = cfg.max_range / 1.25          # nominal mission time
        self.e_ref = cfg.n_uav * cfg.battery       # fleet battery

    # ---------- physics: turn one drone's genes into sampled flight path ----------
    def _trace_drone(self, g):
        c, env = self.cfg, self.env
        way = np.vstack([env.base, g[:, :2], env.base])
        v = np.concatenate([g[:, 2], g[-1:, 2]])
        p = np.concatenate([g[:, 3], g[-1:, 3]])
        a, b = way[:-1], way[1:]
        d = np.linalg.norm(b - a, axis=1)
        n = np.maximum(1, np.ceil(d / c.sample_step).astype(int))
        leg = np.repeat(np.arange(len(d)), n)
        t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 0.5) / np.repeat(n, n)
        pts = a[leg] + (b - a)[leg] * t[:, None]
        leg_time = d / v
        return dict(
            way=way, pts=pts,
            r=c.sense_r_min + (c.sense_r_max - c.sense_r_min) * p[leg],
            dt=(leg_time / n)[leg],
            length=d.sum(), time=leg_time.sum(),
            energy=(d * (c.e_move + c.e_speed * v ** 2) + c.e_sensor * p ** 2 * leg_time).sum())

    def analyze(self, chrom):
        """Full evaluation -> metrics dict (also used for plotting)."""
        c, env = self.cfg, self.env
        traces = [self._trace_drone(g) for g in chrom.genes]
        pts = np.vstack([t["pts"] for t in traces]).astype(np.float32)
        r2 = (np.concatenate([t["r"] for t in traces]) ** 2).astype(np.float32)
        dt = np.concatenate([t["dt"] for t in traces])

        # coverage of cells: n_samples x n_cells distance test
        dx = pts[:, 0:1] - env.cell_x[None, :]
        dy = pts[:, 1:2] - env.cell_y[None, :]
        cover_count = ((dx * dx + dy * dy) <= r2[:, None]).sum(0)
        covered = cover_count > 0
        free = ~env.nofly.ravel()
        coverage = float((env.priority_flat * (covered & free)).sum() / env.total_priority)

        # target detection
        tx = pts[:, 0:1] - env.targets[None, :, 0]
        ty = pts[:, 1:2] - env.targets[None, :, 1]
        detected = ((tx * tx + ty * ty) <= r2[:, None]).any(0)
        detection = float(detected.mean())

        # hazard exposure (time-weighted mean hazard along the paths)
        ix, iy = env.cell_index(pts)
        hazard = float((env.hazard[iy, ix] * dt).sum() / dt.sum())
        viol_samples = int(env.nofly[iy, ix].sum())

        energy = sum(t["energy"] for t in traces)
        mission_time = max(t["time"] for t in traces)
        e_norm, t_norm = energy / self.e_ref, mission_time / self.t_ref

        score = (c.w_cov * coverage + c.w_det * detection
                 - c.w_energy * e_norm - c.w_time * t_norm - c.w_hazard * hazard)

        # ---- hard constraints -> penalties ----
        pen = 0.0
        if viol_samples:
            pen += c.pen_nofly_base + c.pen_nofly_per_sample * viol_samples
        for t in traces:
            ex_r = max(0.0, t["length"] - c.max_range) / c.max_range
            ex_e = max(0.0, t["energy"] - c.battery) / c.battery
            for ex in (ex_r, ex_e):
                if ex > 0:
                    pen += c.pen_limit_base + c.pen_limit_scale * ex
        return dict(fitness=score - pen, score=score, penalty=pen, coverage=coverage,
                    detection=detection, energy=energy, time=mission_time, hazard=hazard,
                    nofly_samples=viol_samples, feasible=pen == 0.0,
                    lengths=[t["length"] for t in traces], traces=traces,
                    cover_count=cover_count.reshape(env.n, env.n), detected=detected)

    def evaluate(self, chrom):
        if chrom.fitness is None:
            m = self.analyze(chrom)
            chrom.fitness = m["fitness"]
            chrom.metrics = {k: m[k] for k in ("coverage", "detection", "energy", "time",
                                                "hazard", "penalty", "feasible")}
            self.n_evals += 1
        return chrom.fitness
