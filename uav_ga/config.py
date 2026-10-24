"""All tunable parameters live here (single source of truth)."""
from dataclasses import dataclass


@dataclass
class Config:
    # ---------------- environment ----------------
    grid_size: int = 50          # 50 x 50 grid of cells
    n_uav: int = 3               # fleet size (one gene block per drone)
    n_waypoints: int = 8         # waypoints per drone (excluding base)
    n_targets: int = 40          # hidden targets (victims / hotspots)
    env_seed: int = 7            # fixes the map so all runs are comparable

    # ---------------- drone physics ----------------
    v_min: float = 0.5           # cells / time-unit
    v_max: float = 2.0
    p_min: float = 0.2           # sensor power (0..1)
    p_max: float = 1.0
    sense_r_min: float = 1.5     # sensing radius at power 0 (cells)
    sense_r_max: float = 5.0     # sensing radius at power 1 (cells)
    e_move: float = 1.0          # energy per cell (baseline drag)
    e_speed: float = 0.5         # extra energy per cell ~ v^2
    e_sensor: float = 2.0        # sensor energy per time-unit ~ p^2
    max_range: float = 130.0     # hard limit: path length per drone (cells)
    battery: float = 260.0       # hard limit: energy per drone
    sample_step: float = 1.5     # path sampling resolution (cells)

    # ---------------- fitness weights (sum of rewards = 0.6, penalties = 0.4)
    w_cov: float = 0.35          # priority-weighted coverage      (maximise)
    w_det: float = 0.25          # target detection rate           (maximise)
    w_energy: float = 0.15       # normalised energy               (minimise)
    w_time: float = 0.10         # normalised mission time         (minimise)
    w_hazard: float = 0.15       # mean hazard exposure            (minimise)

    # ---------------- constraint penalties ----------------
    pen_nofly_base: float = 0.5
    pen_nofly_per_sample: float = 0.02
    pen_limit_base: float = 0.3
    pen_limit_scale: float = 2.0

    # ---------------- genetic algorithm ----------------
    pop_size: int = 80
    generations: int = 150
    tournament_k: int = 3        # selection pressure
    crossover_rate: float = 0.9
    mutation_rate: float = 0.15  # per-waypoint probability
    elite: int = 2
    sigma_start: float = 8.0     # std-dev of spatial mutation (cells), decays
    sigma_end: float = 1.5
    reset_prob: float = 0.08     # chance a mutated waypoint is re-randomised
    informed_fraction: float = 0.5   # share of priority-biased individuals at gen 0
    min_diversity: float = 3.0   # below this, mutation is boosted x2
    patience: int = 60           # stop if no improvement for this many gens
