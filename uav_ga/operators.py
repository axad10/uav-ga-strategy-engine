"""Selection, crossover and mutation operators tailored to spatial genes."""
import numpy as np
from .chromosome import Chromosome


# ---------------- selection ----------------
def tournament_select(pop, k, rng):
    """Larger k  =>  higher selection pressure."""
    idx = rng.integers(0, len(pop), size=k)
    return max((pop[i] for i in idx), key=lambda c: c.fitness)


# ---------------- crossover ----------------
def segment_swap_crossover(a, b, rng):
    """Per drone: one-point crossover on the waypoint sequence (route-segment swap)."""
    D, K, _ = a.genes.shape
    c1, c2 = a.genes.copy(), b.genes.copy()
    for d in range(D):
        cut = rng.integers(1, K)
        c1[d, cut:], c2[d, cut:] = b.genes[d, cut:].copy(), a.genes[d, cut:].copy()
    return Chromosome(c1), Chromosome(c2)


def geometric_crossover(a, b, rng):
    """Blend (arithmetic) crossover: children lie on the line between the parents'
    waypoints -> geometrically meaningful for x,y,speed,power."""
    alpha = rng.uniform(0, 1, size=(a.genes.shape[0], a.genes.shape[1], 1))
    return (Chromosome(alpha * a.genes + (1 - alpha) * b.genes),
            Chromosome((1 - alpha) * a.genes + alpha * b.genes))


def crossover(a, b, rng):
    return (segment_swap_crossover if rng.random() < 0.5 else geometric_crossover)(a, b, rng)


# ---------------- mutation ----------------
def mutate(chrom, cfg, rng, sigma, rate):
    g = chrom.genes
    D, K, _ = g.shape
    hit = rng.random((D, K)) < rate
    for d, k in zip(*np.nonzero(hit)):
        if rng.random() < cfg.reset_prob:                    # exploration: teleport waypoint
            g[d, k, 0:2] = rng.uniform(0, cfg.grid_size - 1, 2)
        else:                                                # exploitation: Gaussian jitter
            g[d, k, 0:2] += rng.normal(0, sigma, 2)
        g[d, k, 2] += rng.normal(0, 0.15 * (cfg.v_max - cfg.v_min))
        g[d, k, 3] += rng.normal(0, 0.15 * (cfg.p_max - cfg.p_min))
    for d in range(D):                                       # order mutations
        if rng.random() < rate:
            i, j = rng.choice(K, 2, replace=False)
            g[d, [i, j]] = g[d, [j, i]]                      # swap two waypoints
        if rng.random() < rate / 2:
            i, j = sorted(rng.choice(K, 2, replace=False))
            g[d, i:j + 1] = g[d, i:j + 1][::-1].copy()       # reverse a sub-route (2-opt)
    chrom.fitness = None
    return chrom
