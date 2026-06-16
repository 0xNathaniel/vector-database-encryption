"""Asymmetric Scalar-Product-Preserving Encryption (ASPE), numpy only.

Vectors are stored as rows. For a secret invertible M, the document-side
transform is M^T p (rows: P @ M) and the query-side transform is M^{-1} q
(rows: Q @ inv(M)^T). Their inner product is then

    <M^T p, M^{-1} q> = p^T M M^{-1} q = p^T q

i.e. ASPE preserves inner products exactly, so Vanilla ASPE retrieval ranking
is identical to plaintext -- its cost is security (known-plaintext attacks),
not utility. Dummy dimensions add a noise term sum_i e_i*d_i to each score,
which is what trades utility for obfuscation.
"""

import numpy as np


class ASPE:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def generate_key(self, dim: int) -> np.ndarray:
        """Random invertible key M (dim x dim).

        Built as an orthogonal basis with random column scaling rather than a
        plain Gaussian matrix, so it stays well-conditioned and float32
        encryption is numerically exact (no near-singular keys, no rounding
        ties that would break the inner-product-preservation property).
        """
        z = self.rng.standard_normal((dim, dim))
        q, _ = np.linalg.qr(z)
        s = self.rng.uniform(0.5, 2.0, size=dim)
        return (q * s).astype(np.float32)

    @staticmethod
    def encrypt_document(P: np.ndarray, M: np.ndarray) -> np.ndarray:
        return (P @ M).astype(np.float32)

    @staticmethod
    def encrypt_query(Q: np.ndarray, M: np.ndarray) -> np.ndarray:
        Minv = np.linalg.inv(M.astype(np.float64))
        return (Q @ Minv.T).astype(np.float32)

    def add_dummy_dims(self, P: np.ndarray, Q: np.ndarray, k: int, sigma: float):
        """Append k dummy dimensions ~N(0, sigma^2) to docs/queries (d -> d+k).
        Doc dummies e and query dummies d add a sum_i e_i*d_i noise term to
        each score (mean 0, variance k*sigma^4), perturbing ranking."""
        n, m = P.shape[0], Q.shape[0]
        E = self.rng.normal(0.0, sigma, size=(n, k)).astype(np.float32)
        D = self.rng.normal(0.0, sigma, size=(m, k)).astype(np.float32)
        return np.hstack([P, E]), np.hstack([Q, D])
