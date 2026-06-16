#!/usr/bin/env python
"""ASPE distance-preserving encryption micro-benchmark for RAG vector
databases. Compares Raw, Vanilla ASPE, and ASPE+Dummy-Dimensions retrieval
on FiQA2018, and sweeps the dummy (k, sigma) utility-security trade-off.

Usage: python experiment.py [--n-corpus N] [--n-queries N] [--dummy-k K]
                            [--dummy-sigma S] [--no-sweep]
Outputs land in results/: experiment_results.{csv,png}, sweep_results.{csv,png}.
"""

import argparse
import os
import time

import numpy as np

from src.config import Config, log
from src.data import embed, load_fiqa, sample_subset
from src.aspe import ASPE
from src.pipeline import run_headline, run_sweep, sanity_check
from src.plotting import plot_headline, plot_sweep


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="ASPE DPE micro-benchmark for RAG.")
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--n-corpus", type=int, default=cfg.n_corpus)
    p.add_argument("--n-queries", type=int, default=cfg.n_queries)
    p.add_argument("--dummy-k", type=int, default=cfg.dummy_k)
    p.add_argument("--dummy-sigma", type=float, default=cfg.dummy_sigma)
    p.add_argument("--no-sweep", action="store_true", help="Skip the (k, sigma) sweep.")
    a = p.parse_args()
    cfg.seed = a.seed
    cfg.n_corpus = a.n_corpus
    cfg.n_queries = a.n_queries
    cfg.dummy_k = a.dummy_k
    cfg.dummy_sigma = a.dummy_sigma
    cfg.run_sweep = not a.no_sweep
    return cfg


def main():
    cfg = parse_args()
    os.makedirs(cfg.results_dir, exist_ok=True)
    t_start = time.perf_counter()
    log("=" * 70)
    log("ASPE Distance-Preserving Encryption micro-benchmark")
    log("=" * 70)

    rng = np.random.default_rng(cfg.seed)

    corpus, queries, qrels = load_fiqa(cfg)
    doc_ids, query_ids, gold = sample_subset(corpus, queries, qrels, cfg, rng)
    corpus_emb, query_emb, gold = embed(corpus, queries, doc_ids, query_ids,
                                        gold, cfg)
    log(f"Embeddings: corpus {corpus_emb.shape}, queries {query_emb.shape}.")

    aspe = ASPE(rng)
    sanity_check(corpus_emb, query_emb, aspe)

    headline = run_headline(corpus_emb, query_emb, gold, aspe, cfg)
    print("\nHeadline results:\n", headline.to_string(index=False), "\n")
    plot_headline(headline, cfg)

    if cfg.run_sweep:
        sweep = run_sweep(corpus_emb, query_emb, gold, aspe, cfg)
        plot_sweep(sweep, headline, cfg)

    log(f"Done in {time.perf_counter() - t_start:.1f}s. Outputs in {cfg.results_dir}/")


if __name__ == "__main__":
    main()
