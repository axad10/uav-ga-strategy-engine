"""Run the UAV GA Strategy Engine.
   python main.py                 -> single GA run + all plots (about 1-2 minutes)
   python main.py --experiments   -> also sensitivity study + GA vs random search
   python main.py --quick         -> tiny run for testing"""
import argparse, os, json
from dataclasses import replace
import numpy as np
from uav_ga import Config, Environment, FitnessEvaluator, GeneticAlgorithm, random_search
from uav_ga import visualize as viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--experiments", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    cfg = Config()
    if a.quick: cfg = replace(cfg, generations=20, pop_size=30)
    if a.generations: cfg = replace(cfg, generations=a.generations)
    if a.pop: cfg = replace(cfg, pop_size=a.pop)
    os.makedirs(a.out, exist_ok=True)

    env = Environment(cfg); ev = FitnessEvaluator(env, cfg)
    ga = GeneticAlgorithm(env, cfg, ev)
    best, hist, snaps = ga.run(seed=a.seed)

    # random-search baseline with identical budget
    rb, rhist = random_search(env, cfg, FitnessEvaluator(env, cfg), ev.n_evals, seed=a.seed)

    viz.plot_convergence(hist, f"{a.out}/convergence.png")
    viz.plot_paths(env, ev, best, f"{a.out}/best_paths.png")
    viz.plot_coverage_heatmap(env, ev, best, f"{a.out}/coverage_heatmap.png")
    viz.plot_snapshots(env, ev, snaps, f"{a.out}/path_evolution.png")
    viz.plot_coverage_evolution(env, ev, snaps, f"{a.out}/coverage_evolution.png")
    viz.plot_paths(env, ev, rb, f"{a.out}/random_search_paths.png", title="Best random-search solution")

    m = ev.analyze(best)
    summary = {k: (float(v) if not isinstance(v, (list, dict, np.ndarray)) else None)
               for k, v in m.items() if k not in ("traces", "cover_count", "detected", "lengths")}
    summary.update(random_search_fitness=float(rb.fitness), evaluations=ev.n_evals,
                   generations_run=len(hist["best"]) - 1, path_lengths=[float(x) for x in m["lengths"]])
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

    if a.experiments:
        from uav_ga import experiments as ex
        ex.sensitivity(cfg, a.out)
        ex.baseline_comparison(cfg, a.out)


if __name__ == "__main__":
    main()
