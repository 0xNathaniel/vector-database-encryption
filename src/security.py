"""Empirical security under a linear Known-Plaintext Attack (KPA).

Threat model: an attacker has observed `kpa_pairs` (plaintext, ciphertext) pairs
-- e.g. by knowing which public documents map to which stored vectors -- and
tries to learn the secret encryption transform from them. Every matrix/scale
based DPE scheme here (Raw, ASPE, ASPE+Dummy, DCPE) is an (approximately) linear
map c = f(p), so the best linear attack is least squares: fit the ciphertext ->
plaintext map W on the known pairs and apply it to the remaining ciphertexts.

We report the Mean Squared Error between the true held-out plaintexts and the
attacker's reconstruction. Low MSE = weak security (the scheme is essentially
recovered); high MSE = the perturbation resists the linear attack.

  * Raw         : ciphertext == plaintext, W ~ identity -> MSE ~ 0 (no security).
  * Vanilla ASPE: exact linear bijection -> MSE ~ 0 (fully broken by KPA).
  * ASPE+Dummy  : dummy-dim noise is unknown per vector -> MSE > 0.
  * DCPE        : bounded perturbation delta is unknown per vector -> MSE > 0.
  * FHE (CKKS)  : no linear plaintext->ciphertext relation without the secret
                  key, so a linear KPA is mathematically impossible -> MSE = inf.
"""

import numpy as np

from .config import Config, log


def mse_kpa(plain_corpus: np.ndarray, cipher_corpus: np.ndarray,
            cfg: Config) -> float:
    """MSE of a least-squares KPA reconstruction of held-out plaintexts.

    plain_corpus  : (n, d)     original (pre-encryption) plaintext rows.
    cipher_corpus : (n, d_c)   stored ciphertext rows (d_c may differ from d,
                               e.g. d+k for ASPE+Dummy). lstsq handles this.
    """
    n = plain_corpus.shape[0]
    n_known = min(cfg.kpa_pairs, n - 1)
    if n_known < 1:
        return float("nan")

    P_known = plain_corpus[:n_known].astype(np.float64)
    C_known = cipher_corpus[:n_known].astype(np.float64)
    P_hold = plain_corpus[n_known:].astype(np.float64)
    C_hold = cipher_corpus[n_known:].astype(np.float64)

    # Solve C_known @ W ~= P_known  ->  recover the ciphertext->plaintext map.
    W, _, _, _ = np.linalg.lstsq(C_known, P_known, rcond=None)
    P_hat = C_hold @ W
    mse = float(np.mean((P_hold - P_hat) ** 2))
    log(f"  KPA: {n_known} known pairs, reconstruction MSE = {mse:.3e}")
    return mse
