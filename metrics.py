import numpy as np


def sliced_wasserstein_1(x, y, projections=128, seed=0):
    """Simple sample-based sliced-W1 diagnostic for multivariate distributions."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same dimension")

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    errors = []
    n = min(len(x), len(y))

    for direction in directions:
        xp = np.sort(x @ direction)[:n]
        yp = np.sort(y @ direction)[:n]
        errors.append(np.mean(np.abs(xp - yp)))

    return float(np.mean(errors))
