"""Plots: convergence, evolved paths, coverage heat-maps, evolution snapshots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]


def _base_map(ax, env, title):
    ax.imshow(env.priority, origin="lower", cmap="YlOrRd", alpha=0.75, vmin=0, vmax=1)
    ax.contour(env.hazard, levels=[0.35, 0.6, 0.85], colors="purple", linewidths=0.8, alpha=0.8)
    ys, xs = np.nonzero(env.nofly)
    ax.add_patch(Rectangle((xs.min() - .5, ys.min() - .5), 0, 0))  # keeps legend order stable
    for (x0, y0, x1, y1) in _rects(env):
        ax.add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fill=True, fc="black",
                               alpha=0.55, ec="black", hatch="//"))
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-.5, env.n - .5); ax.set_ylim(-.5, env.n - .5)


def _rects(env):
    """Recover rectangles from the nofly mask (connected components, simple)."""
    from scipy.ndimage import label, find_objects
    lab, k = label(env.nofly)
    out = []
    for s in find_objects(lab):
        out.append((s[1].start, s[0].start, s[1].stop, s[0].stop))
    return out


def plot_paths(env, ev, chrom, path=None, ax=None, title="Evolved UAV flight paths"):
    m = ev.analyze(chrom)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 6.5))
    _base_map(ax, env, f"{title}\nfit={m['fitness']:.3f} cov={m['coverage']:.2f} "
                       f"det={m['detection']:.2f} haz={m['hazard']:.2f}")
    for d, t in enumerate(m["traces"]):
        w = t["way"]
        ax.plot(w[:, 0], w[:, 1], "-o", color=COLORS[d % 5], ms=3.5, lw=1.8,
                label=f"UAV {d+1}: {t['length']:.0f} cells, E={t['energy']:.0f}")
    det = m["detected"]
    ax.scatter(env.targets[det, 0], env.targets[det, 1], c="green", s=14, marker="x", label="detected target")
    ax.scatter(env.targets[~det, 0], env.targets[~det, 1], c="black", s=18, marker="x", label="missed target")
    ax.scatter(*env.base, marker="*", s=220, c="gold", ec="k", zorder=5, label="base")
    if own:
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        if path: fig.savefig(path, dpi=150); plt.close(fig)
    return m


def plot_coverage_heatmap(env, ev, chrom, path):
    m = ev.analyze(chrom)
    fig, ax = plt.subplots(figsize=(7, 6.2))
    im = ax.imshow(m["cover_count"], origin="lower", cmap="viridis")
    for (x0, y0, x1, y1) in _rects(env):
        ax.add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fc="none", ec="red", lw=1.5, hatch="//"))
    ax.contour(env.priority, levels=[0.5], colors="white", linestyles="--", linewidths=1)
    ax.set_title("Coverage heat-map (sensor passes per cell)\nwhite dashed = high-priority zones, red = no-fly")
    plt.colorbar(im, ax=ax, label="# sensor samples covering cell")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_convergence(hist, path, title="Fitness convergence"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    g = np.arange(len(hist["best"]))
    ax[0].plot(g, hist["best"], label="Best fitness", lw=2)
    ax[0].plot(g, hist["avg"], label="Average fitness", lw=1.5)
    ax[0].set_xlabel("Generation"); ax[0].set_ylabel("Fitness"); ax[0].set_title(title)
    ax[0].set_ylim(min(-1.5, min(hist["avg"])) if min(hist["avg"]) > -3 else -3, max(hist["best"]) + 0.1)
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(g, hist["diversity"], color="tab:orange", label="Population diversity (std of x,y)")
    ax2 = ax[1].twinx(); ax2.plot(g, hist["feasible"], color="tab:green", ls="--", label="Feasible fraction")
    ax[1].set_xlabel("Generation"); ax[1].set_ylabel("diversity"); ax2.set_ylabel("feasible fraction")
    ax[1].set_title("Diversity & constraint satisfaction")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="center right"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_snapshots(env, ev, snaps, path):
    keys = sorted(k for k in snaps if k != "final") + ["final"]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5))
    for ax, k in zip(np.atleast_1d(axes), keys):
        lab = "Final (best ever)" if k == "final" else f"Generation {k}"
        plot_paths(env, ev, snaps[k], ax=ax, title=lab)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_coverage_evolution(env, ev, snaps, path):
    keys = sorted(k for k in snaps if k != "final") + ["final"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.4))
    for ax, k in zip(np.atleast_1d(axes), keys):
        m = ev.analyze(snaps[k])
        ax.imshow(m["cover_count"], origin="lower", cmap="viridis")
        ax.set_title(("Final" if k == "final" else f"Gen {k}") + f"  cov={m['coverage']:.2f}", fontsize=10)
    fig.suptitle("Coverage heat-maps across generations"); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
