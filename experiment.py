#!/usr/bin/env python
"""Privacy-preserving RAG micro-benchmark for encrypted vector databases.

Compares five retrieval methods on FiQA2018 -- Raw (plaintext), Vanilla ASPE,
ASPE+Dummy-Dimensions, DCPE (Scale-and-Perturb), and FHE (CKKS) -- across four
metrics: Recall@K, query latency, DRAM overhead, and empirical security (MSE
under a known-plaintext attack). Also sweeps the dummy (k, sigma) trade-off.

Usage: python experiment.py [--n-corpus N] [--n-queries N] [--dummy-k K]
                            [--dummy-sigma S] [--dcpe-sigma S]
                            [--no-fhe] [--no-sweep]
Outputs land in results/: experiment_results.csv, plot_{recall,latency,dram,
security}.png, and sweep_results.{csv,png}.
"""

import argparse
import os
import time

import numpy as np

from src.config import Config, load_dotenv, log
from src.data import embed, load_fiqa, sample_subset
from src.aspe import ASPE
from src.dcpe import DCPE
from src.pipeline import run_headline, run_sweep, sanity_check
from src.plotting import plot_all, plot_sweep


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="ppRAG DPE/FHE micro-benchmark.")
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--n-corpus", type=int, default=cfg.n_corpus)
    p.add_argument("--n-queries", type=int, default=cfg.n_queries)
    p.add_argument("--dummy-k", type=int, default=cfg.dummy_k)
    p.add_argument("--dummy-sigma", type=float, default=cfg.dummy_sigma)
    p.add_argument("--dcpe-sigma", type=float, default=cfg.dcpe_delta_sigma,
                   help="DCPE bounded-perturbation std.")
    p.add_argument("--kpa-pairs", type=int, default=cfg.kpa_pairs,
                   help="Known (plaintext, ciphertext) pairs for the KPA.")
    p.add_argument("--no-fhe", action="store_true", help="Skip the FHE (CKKS) baseline.")
    p.add_argument("--no-sweep", action="store_true", help="Skip the (k, sigma) sweep.")
    a = p.parse_args()
    cfg.seed = a.seed
    cfg.n_corpus = a.n_corpus
    cfg.n_queries = a.n_queries
    cfg.dummy_k = a.dummy_k
    cfg.dummy_sigma = a.dummy_sigma
    cfg.dcpe_delta_sigma = a.dcpe_sigma
    cfg.kpa_pairs = a.kpa_pairs
    cfg.run_fhe = not a.no_fhe
    cfg.run_sweep = not a.no_sweep
    return cfg


def main():
    cfg = parse_args()
    load_dotenv()  # pick up HF_TOKEN (.env) before any HuggingFace download
    os.makedirs(cfg.results_dir, exist_ok=True)
    t_start = time.perf_counter()
    log("=" * 70)
    log("Privacy-Preserving RAG (DPE vs FHE) micro-benchmark")
    log("=" * 70)
    log("HF auth: " + ("HF_TOKEN found" if os.environ.get("HF_TOKEN")
                       else "no HF_TOKEN (anonymous downloads)"))

    rng = np.random.default_rng(cfg.seed)

    corpus, queries, qrels = load_fiqa(cfg)
    doc_ids, query_ids, gold = sample_subset(corpus, queries, qrels, cfg, rng)
    corpus_emb, query_emb, gold = embed(corpus, queries, doc_ids, query_ids,
                                        gold, cfg)
    log(f"Embeddings: corpus {corpus_emb.shape}, queries {query_emb.shape}.")

    aspe = ASPE(rng)
    dcpe = DCPE(rng)
    sanity_check(corpus_emb, query_emb, aspe)

    headline = run_headline(corpus_emb, query_emb, gold, aspe, dcpe, cfg)
    print("\nHeadline results:\n", headline.to_string(index=False), "\n")
    plot_all(headline, cfg)

    if cfg.run_sweep:
        sweep = run_sweep(corpus_emb, query_emb, gold, aspe, cfg)
        plot_sweep(sweep, headline, cfg)

    log(f"Done in {time.perf_counter() - t_start:.1f}s. Outputs in {cfg.results_dir}/")


if __name__ == "__main__":
    main()
