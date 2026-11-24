"""Run PSO on 5 benchmark functions, report statistics, and save plots + an animated GIF.

Usage:  python examples/benchmark_demo.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pso import PSO
from pso.benchmarks import BENCHMARKS

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)

DIM, ITERS, RUNS = 10, 300, 20


def statistics_and_convergence():
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(18, 3.6))
    rows = []
    for ax, (name, (func, bounds)) in zip(axes, BENCHMARKS.items()):
        histories, finals = [], []
        for seed in range(RUNS):
            res = PSO(func, bounds, dim=DIM, n_particles=40, seed=seed).run(ITERS)
            histories.append(res.history)
            finals.append(res.best_value)
        finals = np.array(finals)
        rows.append((name, finals.mean(), finals.std(), finals.min()))
        ax.semilogy(np.mean(histories, axis=0) + 1e-16)
        ax.set_title(name)
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean best fitness (log)")
    fig.suptitle(f"PSO convergence ({DIM}-D, mean of {RUNS} runs)")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "convergence.png"), dpi=130)
    plt.close(fig)

    print(f"\nPSO on {DIM}-D benchmarks ({RUNS} runs, {ITERS} iterations, 40 particles)")
    print(f"{'Function':<12}{'Mean':>14}{'Std':>14}{'Best':>14}")
    for name, m, s, b in rows:
        print(f"{name:<12}{m:>14.3e}{s:>14.3e}{b:>14.3e}")
    return rows


def swarm_animation():
    func, bounds = BENCHMARKS["Rastrigin"]
    res = PSO(func, bounds, dim=2, n_particles=30, seed=3).run(60, record_positions=True)

    g = np.linspace(bounds[0], bounds[1], 250)
    X, Y = np.meshgrid(g, g)
    Z = func(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(X, Y, Z, 30, cmap="viridis")
    sc = ax.scatter([], [], c="red", s=25, edgecolor="white")
    title = ax.set_title("")
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        sc.set_offsets(res.snapshots[i])
        title.set_text(f"PSO on Rastrigin - iteration {i}")
        return sc, title

    anim = FuncAnimation(fig, update, frames=len(res.snapshots), interval=120)
    anim.save(os.path.join(ASSETS, "swarm.gif"), writer=PillowWriter(fps=8))
    plt.close(fig)


def inertia_study():
    func, bounds = BENCHMARKS["Rastrigin"]
    plt.figure(figsize=(6, 4))
    settings = {"w = 0.4 (fixed)": (0.4, None), "w = 0.9 (fixed)": (0.9, None), "w 0.9 -> 0.4 (linear decay)": (0.9, 0.4)}
    for label, (w, w_end) in settings.items():
        runs = [PSO(func, bounds, dim=DIM, n_particles=40, w=w, w_end=w_end, seed=s).run(ITERS).history
                for s in range(RUNS)]
        plt.plot(np.mean(runs, axis=0), label=label)
    plt.yscale("log")
    plt.xlabel("Iteration"); plt.ylabel("Mean best fitness (log)")
    plt.title("Effect of inertia weight on Rastrigin")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, "inertia_study.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    statistics_and_convergence()
    swarm_animation()
    inertia_study()
    print(f"\nSaved plots to {os.path.abspath(ASSETS)}")
