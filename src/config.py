"""Experiment configuration and shared logging."""

import time
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    dataset: str = "FiQA2018"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    n_corpus: int = 5000
    n_queries: int = 500
    k_list: tuple = (5, 10)               # Recall@K cut-offs.
    dummy_k: int = 50                     # Headline ASPE+Dummy config.
    dummy_sigma: float = 0.05
    sweep_k: tuple = (10, 50, 100, 200)    # Utility-security sweep grid.
    sweep_sigma: tuple = (0.01, 0.02, 0.05, 0.1)
    latency_repeats: int = 5              # MIPS timing repeats (median reported).
    run_sweep: bool = True
    cache_dir: str = ".cache"
    results_dir: str = "results"

    @property
    def out_csv(self) -> str:
        return f"{self.results_dir}/experiment_results.csv"

    @property
    def out_png(self) -> str:
        return f"{self.results_dir}/experiment_results.png"

    @property
    def sweep_csv(self) -> str:
        return f"{self.results_dir}/sweep_results.csv"

    @property
    def sweep_png(self) -> str:
        return f"{self.results_dir}/sweep_results.png"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
