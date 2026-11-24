"""Classic benchmark functions. All take (n_particles, dim) and return (n_particles,)."""
import numpy as np


def sphere(x):
    return np.sum(x ** 2, axis=1)


def rastrigin(x):
    return 10 * x.shape[1] + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


def ackley(x):
    d = x.shape[1]
    a = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2, axis=1) / d))
    b = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / d)
    return a + b + 20 + np.e


def rosenbrock(x):
    return np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)


def griewank(x):
    i = np.arange(1, x.shape[1] + 1)
    return 1 + np.sum(x ** 2, axis=1) / 4000 - np.prod(np.cos(x / np.sqrt(i)), axis=1)


# name -> (function, bounds, known global minimum location description)
BENCHMARKS = {
    "Sphere": (sphere, (-5.12, 5.12)),
    "Rastrigin": (rastrigin, (-5.12, 5.12)),
    "Ackley": (ackley, (-32.0, 32.0)),
    "Rosenbrock": (rosenbrock, (-5.0, 10.0)),
    "Griewank": (griewank, (-600.0, 600.0)),
}
