import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pso import PSO
from pso.benchmarks import sphere, rastrigin, BENCHMARKS


def test_sphere_converges():
    res = PSO(sphere, (-5.12, 5.12), dim=5, n_particles=30, seed=0).run(200)
    assert res.best_value < 1e-6


def test_history_is_non_increasing():
    res = PSO(rastrigin, (-5.12, 5.12), dim=5, seed=1).run(100)
    assert all(b <= a + 1e-12 for a, b in zip(res.history, res.history[1:]))


def test_particles_stay_in_bounds():
    res = PSO(sphere, (-1, 1), dim=3, seed=2).run(50, record_positions=True)
    for snap in res.snapshots:
        assert snap.min() >= -1 and snap.max() <= 1


def test_reproducible_with_seed():
    a = PSO(rastrigin, (-5.12, 5.12), dim=4, seed=7).run(50)
    b = PSO(rastrigin, (-5.12, 5.12), dim=4, seed=7).run(50)
    assert np.allclose(a.best_position, b.best_position)


def test_benchmarks_zero_at_known_minimum():
    for name, (f, _) in BENCHMARKS.items():
        point = np.ones((1, 4)) if name == "Rosenbrock" else np.zeros((1, 4))
        assert abs(f(point)[0]) < 1e-9, name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
