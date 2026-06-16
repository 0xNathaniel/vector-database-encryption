"""Exact MIPS retrieval and evaluation metrics."""

import time

import numpy as np

from .config import Config


def mips(query_mat: np.ndarray, corpus_mat: np.ndarray, k: int) -> np.ndarray:
    """Exact Maximum-Inner-Product-Search via dense matmul + top-k selection.
    Returns an (n_queries, k) array of corpus row indices, best first."""
    scores = query_mat @ corpus_mat.T
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1)
    return part[rows, order]


def recall_at_k(retrieved: np.ndarray, gold, k: int) -> float:
    vals = []
    for i, g in enumerate(gold):
        if g.size == 0:
            continue
        hits = np.isin(retrieved[i, :k], g).sum()
        vals.append(hits / g.size)
    return float(np.mean(vals))


def measure_latency(query_mat, corpus_mat, k: int, repeats: int) -> float:
    """Median MIPS latency in ms per query (score + top-k, all queries)."""
    m = query_mat.shape[0]
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        scores = query_mat @ corpus_mat.T
        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(scores.shape[0])[:, None]
        _ = part[rows, np.argsort(-scores[rows, part], axis=1)]
        times.append(time.perf_counter() - t0)
    return float(np.median(times) / m * 1000.0)


def corpus_dram_mb(corpus_mat: np.ndarray) -> float:
    return corpus_mat.nbytes / (1024.0 ** 2)


def evaluate(query_mat, corpus_mat, gold, cfg: Config) -> dict:
    k_max = max(cfg.k_list)
    retrieved = mips(query_mat, corpus_mat, k_max)
    out = {f"recall@{k}": recall_at_k(retrieved, gold, k) for k in cfg.k_list}
    out["latency_ms"] = measure_latency(query_mat, corpus_mat, k_max,
                                        cfg.latency_repeats)
    out["dram_mb"] = corpus_dram_mb(corpus_mat)
    out["dim"] = corpus_mat.shape[1]
    return out
