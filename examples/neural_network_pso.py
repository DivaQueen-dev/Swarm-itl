"""Train a small neural network with PSO instead of backpropagation.

Each particle is a full set of network weights (flattened). The fitness is the
binary cross-entropy loss on the "two moons" dataset. No gradients are used.

Usage:  python examples/neural_network_pso.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pso import PSO

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)

HIDDEN = 8
N_WEIGHTS = 2 * HIDDEN + HIDDEN + HIDDEN * 1 + 1  # W1, b1, W2, b2


def make_moons(n, noise, rng):
    t = rng.uniform(0, np.pi, n // 2)
    upper = np.c_[np.cos(t), np.sin(t)]
    lower = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    X = np.vstack([upper, lower]) + rng.normal(0, noise, (n // 2 * 2, 2))
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    return X, y


def unpack(w):
    i = 0
    W1 = w[i:i + 2 * HIDDEN].reshape(2, HIDDEN); i += 2 * HIDDEN
    b1 = w[i:i + HIDDEN]; i += HIDDEN
    W2 = w[i:i + HIDDEN].reshape(HIDDEN, 1); i += HIDDEN
    b2 = w[i]
    return W1, b1, W2, b2


def predict_proba(w, X):
    W1, b1, W2, b2 = unpack(w)
    h = np.tanh(X @ W1 + b1)
    z = (h @ W2).ravel() + b2
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def main():
    rng = np.random.default_rng(42)
    X, y = make_moons(400, 0.15, rng)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    Xtr, ytr, Xte, yte = X[:280], y[:280], X[280:], y[280:]

    def fitness(population):
        losses = np.empty(len(population))
        for k, w in enumerate(population):
            p = np.clip(predict_proba(w, Xtr), 1e-9, 1 - 1e-9)
            losses[k] = -np.mean(ytr * np.log(p) + (1 - ytr) * np.log(1 - p))
        return losses

    pso = PSO(fitness, bounds=(-5, 5), dim=N_WEIGHTS, n_particles=50, w=0.9, w_end=0.4, seed=1)
    result = pso.run(iters=300)

    acc = lambda Xs, ys: np.mean((predict_proba(result.best_position, Xs) > 0.5) == ys)
    print(f"Network: 2-{HIDDEN}-1 ({N_WEIGHTS} weights) trained with PSO")
    print(f"Final training loss : {result.best_value:.4f}")
    print(f"Train accuracy      : {acc(Xtr, ytr):.1%}")
    print(f"Test accuracy       : {acc(Xte, yte):.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(result.history)
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Training loss (BCE)")
    axes[0].set_title("PSO training curve"); axes[0].grid(alpha=0.3)

    gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 250), np.linspace(-1.2, 1.7, 250))
    Z = predict_proba(result.best_position, np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    axes[1].contourf(gx, gy, Z, levels=20, cmap="RdBu_r", alpha=0.7)
    axes[1].scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap="RdBu_r", edgecolor="k", s=25)
    axes[1].set_title(f"Decision boundary (test acc {acc(Xte, yte):.1%})")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "nn_pso.png"), dpi=130)
    print(f"Saved {os.path.abspath(os.path.join(ASSETS, 'nn_pso.png'))}")


if __name__ == "__main__":
    main()
