"""Particle Swarm Optimization (PSO) - a small, vectorised NumPy implementation."""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class PSOResult:
    best_position: np.ndarray
    best_value: float
    history: List[float] = field(default_factory=list)       # best fitness per iteration
    snapshots: List[np.ndarray] = field(default_factory=list)  # particle positions per iteration


class PSO:
    """Minimise `func` using Particle Swarm Optimization.

    Parameters
    ----------
    func : callable
        Vectorised objective. Takes an array of shape (n_particles, dim)
        and returns an array of shape (n_particles,).
    bounds : (low, high)
        Search-space limits, applied to every dimension.
    dim : int
        Number of dimensions of the problem.
    n_particles : int
        Swarm size.
    w, w_end : float
        Inertia weight. If `w_end` is given, it decreases linearly from `w`
        to `w_end` over the run (explore first, exploit later).
    c1, c2 : float
        Cognitive (own best) and social (swarm best) coefficients.
    v_max_frac : float
        Velocity clamp as a fraction of the search range.
    seed : int, optional
        Random seed for reproducible runs.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        bounds: Tuple[float, float],
        dim: int,
        n_particles: int = 30,
        w: float = 0.9,
        w_end: Optional[float] = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max_frac: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.func = func
        self.low, self.high = bounds
        self.dim = dim
        self.n = n_particles
        self.w, self.w_end = w, w_end
        self.c1, self.c2 = c1, c2
        self.v_max = v_max_frac * (self.high - self.low)
        self.rng = np.random.default_rng(seed)

    def run(self, iters: int = 100, record_positions: bool = False) -> PSOResult:
        rng, n, d = self.rng, self.n, self.dim

        x = rng.uniform(self.low, self.high, (n, d))
        v = rng.uniform(-self.v_max, self.v_max, (n, d))

        pbest = x.copy()
        pbest_val = self.func(x)
        g = int(np.argmin(pbest_val))
        gbest, gbest_val = pbest[g].copy(), float(pbest_val[g])

        history = [gbest_val]
        snapshots = [x.copy()] if record_positions else []

        for t in range(iters):
            w = self.w if self.w_end is None else self.w + (self.w_end - self.w) * t / max(iters - 1, 1)
            r1, r2 = rng.random((2, n, d))

            v = w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            v = np.clip(v, -self.v_max, self.v_max)
            x = np.clip(x + v, self.low, self.high)

            val = self.func(x)
            better = val < pbest_val
            pbest[better], pbest_val[better] = x[better], val[better]

            g = int(np.argmin(pbest_val))
            if pbest_val[g] < gbest_val:
                gbest, gbest_val = pbest[g].copy(), float(pbest_val[g])

            history.append(gbest_val)
            if record_positions:
                snapshots.append(x.copy())

        return PSOResult(gbest, gbest_val, history, snapshots)
