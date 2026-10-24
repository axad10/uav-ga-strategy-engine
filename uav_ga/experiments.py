"""Hyper-parameter sensitivity + GA-vs-random-search comparison."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import mannwhitneyu
from .environment import Environment
from .fitness import FitnessEvaluator
from .ga import GeneticAlgorithm
from .baseline import random_search


def _run_ga(args):
    cfg, seed = args
    env = Environment(cfg); ev = FitnessEvaluator(env, cfg)
    best, hist, _ = GeneticAlgorithm(env, cfg, ev).run(seed=seed, verbose=False)
    return best.fitness, best.metrics["feasible"], best.genes


def _run_rs(args):
    cfg, seed, n = args
    env = Environment(cfg); ev = FitnessEvaluator(env, cfg)
    best, curve = random_search(env, cfg, ev, n, seed)
    return best.fitness, best.metrics["feasible"], best.genes


def sensitivity(cfg, out_dir, budget=2400, n_seeds=3):
    """Vary one parameter at a time. Evaluation budget is kept equal by scaling generations."""
    grid = {"pop_size": [20, 50, 100], "mutation_rate": [0.02, 0.10, 0.30], "tournament_k": [2, 4, 8]}
    rows = []
    with ProcessPoolExecutor() as ex:
        for param, values in grid.items():
            for v in values:
                c = replace(cfg, **{param: v}, patience=10 ** 6)
                c = replace(c, generations=max(5, budget // c.pop_size))
                res = list(ex.map(_run_ga, [(c, s) for s in range(n_seeds)]))
                f = [r[0] for r in res]
                rows.append(dict(param=param, value=v, generations=c.generations,
                                 mean_fitness=np.mean(f), std_fitness=np.std(f),
                                 feasible=np.mean([r[1] for r in res])))
                print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    labels = {"pop_size": "Population size", "mutation_rate": "Mutation rate",
              "tournament_k": "Tournament size (selection pressure)"}
    for ax, p in zip(axes, grid):
        d = df[df.param == p]
        ax.errorbar(d.value.astype(str), d.mean_fitness, yerr=d.std_fitness, fmt="-o", capsize=4)
        ax.set_xlabel(labels[p]); ax.grid(alpha=.3)
    axes[0].set_ylabel("Final best fitness (mean ± std)")
    fig.suptitle(f"Hyper-parameter sensitivity (equal budget = {budget} evaluations, {n_seeds} seeds)")
    fig.tight_layout(); fig.savefig(f"{out_dir}/sensitivity.png", dpi=150); plt.close(fig)
    return df


def baseline_comparison(cfg, out_dir, n_seeds=5, budget=4000):
    c = replace(cfg, generations=budget // cfg.pop_size, patience=10 ** 6)
    with ProcessPoolExecutor() as ex:
        ga = list(ex.map(_run_ga, [(c, s) for s in range(n_seeds)]))
        rs = list(ex.map(_run_rs, [(cfg, s, budget) for s in range(n_seeds)]))
    fg, fr = np.array([r[0] for r in ga]), np.array([r[0] for r in rs])
    p = mannwhitneyu(fg, fr, alternative="greater").pvalue

    def mean_pairwise_dist(sols):          # order-independent: mean distance between waypoint sets
        from scipy.spatial.distance import cdist
        ds = []
        for i in range(len(sols)):
            for j in range(i + 1, len(sols)):
                a, b = sols[i][..., :2].reshape(-1, 2), sols[j][..., :2].reshape(-1, 2)
                dm = cdist(a, b)
                ds.append(0.5 * (dm.min(1).mean() + dm.min(0).mean()))
        return float(np.mean(ds))

    summary = dict(ga_mean=fg.mean(), ga_std=fg.std(), rs_mean=fr.mean(), rs_std=fr.std(),
                   ga_feasible=np.mean([r[1] for r in ga]), rs_feasible=np.mean([r[1] for r in rs]),
                   mannwhitney_p=p, ga_solution_spread=mean_pairwise_dist([r[2] for r in ga]),
                   rs_solution_spread=mean_pairwise_dist([r[2] for r in rs]))
    pd.DataFrame([summary]).to_csv(f"{out_dir}/baseline_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([fg, fr]); ax.set_xticklabels(["Genetic Algorithm", "Random search"])
    ax.set_ylabel("Best fitness"); ax.grid(alpha=.3)
    ax.set_title(f"Same budget ({budget} evals), {n_seeds} seeds\nMann-Whitney p={p:.3f}")
    fig.tight_layout(); fig.savefig(f"{out_dir}/ga_vs_random.png", dpi=150); plt.close(fig)
    print(summary)
    return summary
