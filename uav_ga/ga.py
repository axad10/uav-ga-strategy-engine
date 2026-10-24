"""Object-oriented Genetic Algorithm engine."""
import numpy as np
from .chromosome import Chromosome
from .operators import tournament_select, crossover, mutate


class GeneticAlgorithm:
    def __init__(self, env, cfg, evaluator):
        self.env, self.cfg, self.ev = env, cfg, evaluator

    def init_population(self, rng):
        cfg = self.cfg
        n_inf = int(cfg.pop_size * cfg.informed_fraction)
        pop = [Chromosome.random(cfg, self.env, rng, informed=(i < n_inf))
               for i in range(cfg.pop_size)]
        for c in pop:
            self.ev.evaluate(c)
        return pop

    @staticmethod
    def diversity(pop):
        xy = np.stack([c.genes[..., :2] for c in pop])
        return float(xy.std(axis=0).mean())

    def run(self, seed=0, verbose=True):
        cfg, rng = self.cfg, np.random.default_rng(seed)
        pop = self.init_population(rng)
        hist = dict(best=[], avg=[], diversity=[], feasible=[], evals=[])
        snaps, best_ever, stall = {}, None, 0
        snap_at = {0, cfg.generations // 4, cfg.generations // 2}

        def log(g):
            f = np.array([c.fitness for c in pop])
            hist["best"].append(f.max()); hist["avg"].append(f.mean())
            hist["diversity"].append(self.diversity(pop))
            hist["feasible"].append(np.mean([c.metrics["feasible"] for c in pop]))
            hist["evals"].append(self.ev.n_evals)
            if g in snap_at:
                snaps[g] = max(pop, key=lambda c: c.fitness).copy()

        log(0)
        best_ever = max(pop, key=lambda c: c.fitness).copy()
        for g in range(1, cfg.generations + 1):
            frac = g / cfg.generations
            sigma = cfg.sigma_start + (cfg.sigma_end - cfg.sigma_start) * frac
            rate = cfg.mutation_rate * (2.0 if hist["diversity"][-1] < cfg.min_diversity else 1.0)
            pop.sort(key=lambda c: c.fitness, reverse=True)
            new = [c.copy() for c in pop[:cfg.elite]]                  # elitism
            while len(new) < cfg.pop_size:
                p1 = tournament_select(pop, cfg.tournament_k, rng)
                p2 = tournament_select(pop, cfg.tournament_k, rng)
                if rng.random() < cfg.crossover_rate:
                    c1, c2 = crossover(p1, p2, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                for c in (c1, c2):
                    mutate(c, cfg, rng, sigma, rate).repair(cfg, self.env)
                    new.append(c)
            pop = new[:cfg.pop_size]
            for c in pop:
                self.ev.evaluate(c)
            log(g)
            cur = max(pop, key=lambda c: c.fitness)
            if cur.fitness > best_ever.fitness + 1e-6:
                best_ever, stall = cur.copy(), 0
            else:
                stall += 1
            if verbose and (g % 10 == 0 or g == 1):
                m = cur.metrics
                print(f"gen {g:4d} | best {cur.fitness:+.4f} avg {hist['avg'][-1]:+.4f} "
                      f"div {hist['diversity'][-1]:5.2f} | cov {m['coverage']:.2f} "
                      f"det {m['detection']:.2f} haz {m['hazard']:.2f} feas {m['feasible']}")
            if stall >= cfg.patience:                                   # termination: stagnation
                if verbose:
                    print(f"early stop at generation {g} (no improvement for {stall} gens)")
                break
        snaps["final"] = best_ever
        return best_ever, hist, snaps
