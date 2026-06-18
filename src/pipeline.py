"""Experiment driver: builds per-method matrices, runs the headline
comparison and the (k, sigma) utility-security sweep."""

import numpy as np
import pandas as pd

from .aspe import ASPE
from .config import Config, log
from .dcpe import DCPE
from . import fhe
from .retrieval import evaluate
from .security import mse_kpa


def build_method_matrices(method, corpus_emb, query_emb, aspe: ASPE, cfg: Config,
                          dcpe: "DCPE | None" = None,
                          k: int = 0, sigma: float = 0.0):
    if method == "Raw":
        return corpus_emb, query_emb
    if method == "Vanilla ASPE":
        M = aspe.generate_key(corpus_emb.shape[1])
        return (aspe.encrypt_document(corpus_emb, M),
                aspe.encrypt_query(query_emb, M))
    if method == "ASPE + Dummy":
        P_aug, Q_aug = aspe.add_dummy_dims(corpus_emb, query_emb, k, sigma)
        M = aspe.generate_key(P_aug.shape[1])
        return (aspe.encrypt_document(P_aug, M),
                aspe.encrypt_query(Q_aug, M))
    if method == "DCPE":
        lam = dcpe.generate_key(cfg.dcpe_lambda_range)
        return (dcpe.encrypt_document(corpus_emb, lam, cfg.dcpe_delta_sigma),
                dcpe.encrypt_query(query_emb, lam))
    raise ValueError(method)


def sanity_check(corpus_emb, query_emb, aspe: ASPE):
    """Assert ASPE preserves inner products (Vanilla ASPE == Raw scores)."""
    log("Sanity check: verifying ASPE inner-product preservation ...")
    M = aspe.generate_key(corpus_emb.shape[1])
    enc_c = aspe.encrypt_document(corpus_emb[:50], M)
    enc_q = aspe.encrypt_query(query_emb[:50], M)
    raw_scores = query_emb[:50] @ corpus_emb[:50].T
    enc_scores = enc_q @ enc_c.T
    max_err = np.max(np.abs(raw_scores - enc_scores))
    log(f"  max |raw - encrypted| score error = {max_err:.2e}")
    assert max_err < 1e-3, "ASPE failed to preserve inner products!"


def run_headline(corpus_emb, query_emb, gold, aspe: ASPE, dcpe: DCPE,
                 cfg: Config) -> pd.DataFrame:
    rows = []
    for method in ["Raw", "Vanilla ASPE", "ASPE + Dummy", "DCPE"]:
        log(f"Evaluating method: {method} ...")
        cm, qm = build_method_matrices(
            method, corpus_emb, query_emb, aspe, cfg, dcpe=dcpe,
            k=cfg.dummy_k, sigma=cfg.dummy_sigma)
        res = evaluate(qm, cm, gold, cfg)
        res["method"] = method
        # Empirical security: KPA reconstruction MSE of the original embeddings
        # from the stored ciphertext (Raw maps to itself -> ~0, no protection).
        res["security_mse"] = mse_kpa(corpus_emb, cm, cfg)
        if method == "ASPE + Dummy":
            res["dummy_k"], res["dummy_sigma"] = cfg.dummy_k, cfg.dummy_sigma
        else:
            res["dummy_k"], res["dummy_sigma"] = 0, 0.0
        rows.append(res)
        log(f"    {res}")

    # FHE row: measured on a subset and extrapolated; recall copied from Raw.
    if cfg.run_fhe:
        log("Evaluating method: FHE (CKKS) ...")
        raw_recalls = rows[0]
        res = fhe.benchmark_fhe(corpus_emb, query_emb, raw_recalls, cfg)
        res["method"] = "FHE (CKKS)"
        res["dummy_k"], res["dummy_sigma"] = 0, 0.0
        rows.append(res)
        log(f"    {res}")

    df = pd.DataFrame(rows)[
        ["method", "dim", "dummy_k", "dummy_sigma",
         *[f"recall@{k}" for k in cfg.k_list],
         "latency_ms", "dram_mb", "security_mse"]
    ]
    df.to_csv(cfg.out_csv, index=False)
    log(f"Saved headline metrics -> {cfg.out_csv}")
    return df


def run_sweep(corpus_emb, query_emb, gold, aspe: ASPE, cfg: Config) -> pd.DataFrame:
    log("Running (k, sigma) sweep ...")
    raw_scores = query_emb @ corpus_emb.T
    raw_std = float(raw_scores.std())

    rows = []
    for k in cfg.sweep_k:
        for sigma in cfg.sweep_sigma:
            P_aug, Q_aug = aspe.add_dummy_dims(corpus_emb, query_emb, k, sigma)
            M = aspe.generate_key(P_aug.shape[1])
            cm = aspe.encrypt_document(P_aug, M)
            qm = aspe.encrypt_query(Q_aug, M)
            res = evaluate(qm, cm, gold, cfg)
            # Security/obfuscation proxy: RMS score distortion vs plaintext,
            # normalised by the plaintext score spread.
            enc_scores = qm @ cm.T
            distortion = float(np.sqrt(np.mean((enc_scores - raw_scores) ** 2)))
            res.update(dummy_k=k, dummy_sigma=sigma,
                       distortion=distortion,
                       distortion_norm=distortion / raw_std)
            rows.append(res)
            log(f"  k={k:>3} sigma={sigma:<5} "
                f"R@10={res['recall@10']:.3f} distortion={distortion:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(cfg.sweep_csv, index=False)
    log(f"Saved sweep metrics -> {cfg.sweep_csv}")
    return df
