"""Random-search baseline with the SAME evaluation budget and the SAME
initialisation generator as GA generation 0 (fair comparison)."""
import numpy as np
from .chromosome import Chromosome


def random_search(env, cfg, evaluator, n_evals, seed=0):
    rng = np.random.default_rng(seed)
    best, best_curve, avg_curve, batch = None, [], [], []
    n_inf = int(cfg.pop_size * cfg.informed_fraction)
    for i in range(n_evals):
        c = Chromosome.random(cfg, env, rng, informed=(i % cfg.pop_size) < n_inf)
        evaluator.evaluate(c)
        batch.append(c.fitness)
        if best is None or c.fitness > best.fitness:
            best = c
        if (i + 1) % cfg.pop_size == 0:           # one "generation" = pop_size samples
            best_curve.append(best.fitness)
            avg_curve.append(float(np.mean(batch)))
            batch = []
    return best, dict(best=best_curve, avg=avg_curve)
