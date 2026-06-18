"""DCPE / Scale-and-Perturb: an approximate distance-preserving encryption.

Unlike ASPE (which preserves inner products *exactly* and is therefore broken
by a linear known-plaintext attack), DCPE deliberately injects a bounded random
perturbation so that exact distances cannot be recovered, at the cost of a small
ranking error. Vectors are stored as rows.

For a secret scalar lambda and per-row bounded Gaussian noise delta, the
document transform is

    c = lambda * p + delta        (delta ~ N(0, sigma^2), norm-clipped)

and the query transform is q' = lambda * q (queries are scaled, not perturbed).
The encrypted score is then

    <c, q'> = lambda^2 <p, q> + lambda <delta, q>,

i.e. the true score scaled by lambda^2 (monotone, rank-preserving) plus a noise
term lambda <delta, q> that obscures the exact value. Because delta is fresh per
vector and never revealed, a linear KPA cannot undo it -- the irreducible noise
shows up as a non-zero reconstruction MSE (see src/security.py).
"""

import numpy as np


class DCPE:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def generate_key(self, lambda_range: tuple) -> float:
        """Secret positive scale lambda drawn uniformly from lambda_range."""
        lo, hi = lambda_range
        return float(self.rng.uniform(lo, hi))

    def encrypt_document(self, P: np.ndarray, lam: float, sigma: float) -> np.ndarray:
        """c = lambda * P + delta, with delta a bounded Gaussian perturbation.

        delta is drawn ~N(0, sigma^2) per coordinate and clipped so each row's
        perturbation norm is bounded by 3*sigma*sqrt(d) (a 3-sigma ball), keeping
        the distortion bounded as required for approximate distance preservation.
        """
        d = P.shape[1]
        delta = self.rng.normal(0.0, sigma, size=P.shape)
        bound = 3.0 * sigma * np.sqrt(d)
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, bound / np.maximum(norms, 1e-12))
        delta = delta * scale
        return (lam * P + delta).astype(np.float32)

    def encrypt_query(self, Q: np.ndarray, lam: float) -> np.ndarray:
        """q' = lambda * Q (queries are scaled but not perturbed)."""
        return (lam * Q).astype(np.float32)
