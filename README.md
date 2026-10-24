# UAV Fleet Strategy Engine — Genetic Algorithm

Bahria University Lahore · Artificial Intelligence (BSCS 5) · Assignment 2 (CLO4)

A multi-objective **Genetic Algorithm** that plans waypoints, speeds and sensor power for a
fleet of 3 UAVs surveying a 50 × 50 disaster grid containing high-priority zones, hazard
pockets (wind/radiation) and no-fly zones.

## Project layout
```
uav-ga-strategy-engine/
├── main.py                 # entry point (run this)
├── requirements.txt
├── README.md
└── uav_ga/
    ├── config.py           # every parameter (weights, GA settings, drone physics)
    ├── environment.py      # grid: priority map, hazard map, no-fly zones, targets
    ├── chromosome.py       # encoding + repair mechanism
    ├── fitness.py          # multi-objective fitness + penalties
    ├── operators.py        # tournament selection, crossover, mutation
    ├── ga.py               # GA engine (init, elitism, adaptive mutation, termination)
    ├── baseline.py         # random-search baseline
    ├── experiments.py      # hyper-parameter study + GA-vs-random comparison
    └── visualize.py        # all plots
```

## Install & run
```bash
git clone <your-repo-url>
cd uav-ga-strategy-engine
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main.py                  # one GA run + plots (~2 min)  -> outputs/
python main.py --experiments    # + sensitivity study + GA vs random search (~10-15 min)
python main.py --quick          # 20-generation smoke test
python main.py --generations 200 --pop 100 --seed 3
```
Outputs (PNG + JSON/CSV) are written to `outputs/`.

## Design summary
| Component | Choice |
|---|---|
| Chromosome | real-valued tensor `[drone, waypoint, (x, y, speed, sensor_power)]` — 3 × 8 × 4 genes |
| Fitness | `0.35·priority coverage + 0.25·detection − 0.15·energy − 0.10·time − 0.15·hazard − penalties` |
| Selection | tournament (size k), elitism = 2 |
| Crossover | 50 % segment-swap (route one-point), 50 % geometric/blend |
| Mutation | Gaussian jitter with decaying σ, waypoint teleport, swap, sub-route reversal; auto-boost when diversity drops |
| Constraints | repair (snap waypoints out of no-fly zones) + penalties (no-fly crossings, range, battery) |
| Termination | max generations or `patience` generations without improvement |
| Baseline | random search with identical evaluation budget |

Change any weight or GA setting in `uav_ga/config.py`.
