"""Experiment configuration and shared logging."""

import os
import time
from dataclasses import dataclass


def load_dotenv(path: str = ".env") -> None:
    """Minimal, dependency-free .env loader.

    Reads KEY=VALUE lines from `path` into os.environ without overriding
    variables already set in the real environment. Used so HuggingFace picks up
    HF_TOKEN for authenticated (faster, rate-limit-free) dataset and model
    downloads. Also defaults HF_HUB_DISABLE_XET=1 to avoid the Xet transfer
    backend stalling at 0 bytes (observed on this setup); export it yourself to
    override.
    """
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(),
                                      val.strip().strip('"').strip("'"))
    # Belt-and-braces robustness for the FiQA download.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        # huggingface_hub honours HF_TOKEN; also expose the alias it checks.
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    return None


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

    # DCPE (Scale-and-Perturb, approximate DPE): secret scalar lambda drawn from
    # this range, plus a bounded Gaussian perturbation (std dcpe_delta_sigma)
    # added to the corpus to obscure exact distances.
    dcpe_lambda_range: tuple = (0.5, 2.0)
    dcpe_delta_sigma: float = 0.02

    # FHE (CKKS via tenseal). FHE retrieval is too slow to run in full, so we
    # measure latency / DRAM on a tiny subset and extrapolate to the full corpus.
    run_fhe: bool = True
    fhe_poly_modulus_degree: int = 8192
    fhe_coeff_mod_bit_sizes: tuple = (60, 40, 40, 60)
    fhe_global_scale_bits: int = 40
    fhe_sub_queries: int = 5
    fhe_sub_docs: int = 50
    fhe_max_mse: float = float("inf")     # KPA is infeasible without the key.

    # Empirical security: known-plaintext-attack sample size. Must exceed the
    # ciphertext dimension (d + dummy_k = 434 here) for the least-squares attack
    # to be well-posed; with fewer pairs the linear system is underdetermined
    # and even Raw (no encryption) shows a spurious non-zero reconstruction
    # error, making the metric non-discriminative. 1000 over-determines every
    # method so exact-linear schemes resolve to ~0 (genuinely broken) while
    # DCPE's per-vector perturbation leaves an irreducible residual.
    kpa_pairs: int = 1000

    cache_dir: str = ".cache"
    results_dir: str = "results"

    @property
    def out_csv(self) -> str:
        return f"{self.results_dir}/experiment_results.csv"

    @property
    def out_png(self) -> str:
        return f"{self.results_dir}/experiment_results.png"

    @property
    def plot_recall_png(self) -> str:
        return f"{self.results_dir}/plot_recall.png"

    @property
    def plot_latency_png(self) -> str:
        return f"{self.results_dir}/plot_latency.png"

    @property
    def plot_dram_png(self) -> str:
        return f"{self.results_dir}/plot_dram.png"

    @property
    def plot_security_png(self) -> str:
        return f"{self.results_dir}/plot_security.png"

    @property
    def sweep_csv(self) -> str:
        return f"{self.results_dir}/sweep_results.csv"

    @property
    def sweep_png(self) -> str:
        return f"{self.results_dir}/sweep_results.png"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
