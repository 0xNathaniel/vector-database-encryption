"""Fully Homomorphic Encryption baseline (CKKS scheme, via tenseal).

FHE encrypts each vector under a public key so that inner products can be
computed on ciphertexts without ever decrypting -- the gold standard for
security, but orders of magnitude slower and larger than plaintext or DPE.

Running the full 500 x 5000 encrypted MIPS is prohibitively slow, so we follow
the standard micro-benchmark recipe: encrypt a tiny subset, measure the true
per-query latency and per-vector ciphertext footprint, and extrapolate to the
full corpus. Retrieval quality is taken to equal Raw (CKKS keeps ~plaintext
precision), and the empirical KPA security is infinite (a linear known-plaintext
attack cannot recover the secret key).

If tenseal is not installed, `TENSEAL_AVAILABLE` is False and `benchmark_fhe`
returns a NaN-filled row so the rest of the benchmark still completes.
"""

import time

import numpy as np
import psutil

from .config import Config, log

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    ts = None
    TENSEAL_AVAILABLE = False


def make_context(cfg: Config):
    """CKKS context with the paper's parameters."""
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=cfg.fhe_poly_modulus_degree,
        coeff_mod_bit_sizes=list(cfg.fhe_coeff_mod_bit_sizes),
    )
    ctx.global_scale = 2 ** cfg.fhe_global_scale_bits
    ctx.generate_galois_keys()
    return ctx


def _empty_row(cfg: Config, raw_recalls: dict) -> dict:
    out = {f"recall@{k}": float("nan") for k in cfg.k_list}
    out.update(latency_ms=float("nan"), dram_mb=float("nan"),
               dim=float("nan"), security_mse=cfg.fhe_max_mse)
    return out


def benchmark_fhe(corpus_emb: np.ndarray, query_emb: np.ndarray,
                  raw_recalls: dict, cfg: Config) -> dict:
    """Measure FHE latency / DRAM on a subset and extrapolate to full corpus.

    raw_recalls: {k: recall@k} from the Raw baseline (copied for FHE, since
    CKKS retains precision). Returns a metrics dict shaped like evaluate().
    """
    if not TENSEAL_AVAILABLE:
        log("  tenseal not available -- skipping FHE (row recorded as NaN).")
        return _empty_row(cfg, raw_recalls)

    try:
        n_docs = min(cfg.fhe_sub_docs, corpus_emb.shape[0])
        n_q = min(cfg.fhe_sub_queries, query_emb.shape[0])
        d = corpus_emb.shape[1]
        log(f"  FHE: setting up CKKS context "
            f"(poly_modulus_degree={cfg.fhe_poly_modulus_degree}) ...")
        ctx = make_context(cfg)

        proc = psutil.Process()
        rss_before = proc.memory_info().rss

        # Encrypt the document subset; time it and capture ciphertext size.
        log(f"  FHE: encrypting subset ({n_docs} docs, {n_q} queries) ...")
        enc_docs = [ts.ckks_vector(ctx, corpus_emb[i].tolist())
                    for i in range(n_docs)]
        enc_queries = [ts.ckks_vector(ctx, query_emb[i].tolist())
                       for i in range(n_q)]

        rss_after = proc.memory_info().rss
        bytes_per_vec = len(enc_docs[0].serialize())
        dram_mb = bytes_per_vec * cfg.n_corpus / (1024.0 ** 2)
        log(f"  FHE: {bytes_per_vec/1024:.1f} KB/ciphertext "
            f"(d={d}); RSS delta for subset = "
            f"{(rss_after - rss_before)/1024**2:.1f} MB")

        # Time one encrypted query against the full document subset, then
        # extrapolate per-query latency to the full corpus size.
        t0 = time.perf_counter()
        for eq in enc_queries:
            for ed in enc_docs:
                _ = eq.dot(ed).decrypt()
        elapsed = time.perf_counter() - t0
        per_query_sub_ms = elapsed / n_q * 1000.0
        latency_ms = per_query_sub_ms * (cfg.n_corpus / n_docs)
        log(f"  FHE: {per_query_sub_ms:.1f} ms/query over {n_docs} docs "
            f"-> extrapolated {latency_ms:.1f} ms/query over {cfg.n_corpus}")

        out = {f"recall@{k}": float(raw_recalls.get(f"recall@{k}", float("nan")))
               for k in cfg.k_list}
        out.update(latency_ms=latency_ms, dram_mb=dram_mb, dim=d,
                   security_mse=cfg.fhe_max_mse)
        return out
    except Exception as exc:  # pragma: no cover - runtime safety
        log(f"  FHE benchmark failed ({type(exc).__name__}: {exc}); "
            "recording NaN row.")
        return _empty_row(cfg, raw_recalls)
