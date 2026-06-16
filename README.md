<div align="center">
  <h1>Encrypted Vector Database Benchmark</h1>
  <h3>To Encrypt or Not to Encrypt? Evaluating the Trade-offs of Distance-Preserving Encryption in RAG Vector Databases</h3>

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?logo=numpy&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-MiniLM--L6--v2-orange?logo=huggingface&logoColor=white)
![MTEB](https://img.shields.io/badge/MTEB-FiQA2018-yellow)
![Pandas](https://img.shields.io/badge/Pandas-Results-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-11557C?logo=plotly&logoColor=white)

</div>

## Overview

Vector databases used in Retrieval Augmented Generation systems typically store raw embeddings in plaintext, which leaks semantic information about the underlying documents if the database is compromised. Distance Preserving Encryption schemes such as Asymmetric Scalar Product Preserving Encryption, or ASPE, attempt to protect those embeddings while still allowing similarity search to run directly on the encrypted vectors.

This repository contains a micro benchmark that measures what ASPE actually costs and protects, by comparing three retrieval setups on a real financial question answering dataset.

1. Raw, plaintext embeddings, used as the utility baseline.
2. Vanilla ASPE, embeddings encrypted with a secret invertible matrix and no added noise.
3. ASPE with dummy dimensions, where random noise dimensions are injected into the vectors before encryption to obscure the true inner products.

## Why this matters

ASPE encrypts a document vector p as M transpose times p, and encrypts a query vector q as M inverse times q, for a secret invertible matrix M. The inner product of the encrypted vectors works out to p transpose times q, which is exactly the plaintext inner product. This means Vanilla ASPE preserves ranking perfectly and should retrieve identically to the plaintext baseline. Its weakness is purely cryptographic, since a known plaintext attack can recover M.

Dummy dimensions break that exact preservation on purpose. Random noise values are appended to both document and query vectors before encryption, adding an extra term to every inner product score that depends only on the noise. This perturbs the ranking and therefore costs retrieval utility, in exchange for making the encrypted scores harder to interpret. The size and number of these dummy dimensions control how much utility is traded for how much obfuscation, which is the central trade off this benchmark quantifies.

## Experiment design

Dataset: FiQA2018, a financial domain question answering retrieval task, loaded through the MTEB library. A subset of 5000 corpus documents and 500 queries is sampled, with every gold document for a sampled query guaranteed to remain in the sampled corpus so the relevance judgments stay valid.

Embedding model: Sentence Transformers all MiniLM L6 v2, with embeddings L2 normalized so that inner product equals cosine similarity.

Retrieval: exact Maximum Inner Product Search implemented directly with NumPy matrix multiplication and top k selection, with no external vector database or approximate index involved.

Metrics collected for each method:

Recall at 5 and Recall at 10, comparing retrieved document ids against ground truth relevance judgments.

Average query latency in milliseconds, timed with high resolution perf counters over the full score and top k computation for all queries.

DRAM overhead, measured as the exact in memory size of the stored corpus matrix, which captures the cost of the extra dummy dimensions.

In addition to the three method headline comparison, the script sweeps the dummy dimension count and noise scale over a grid of values and records how recall and score distortion change together, producing a direct picture of the utility versus security trade off.

## Repository structure

experiment.py is the entry point that wires the pipeline together and exposes command line options.

src/config.py holds the experiment configuration and a small logging helper.

src/data.py loads FiQA2018 through MTEB, samples a valid subset, and embeds it with Sentence Transformers, caching embeddings to disk.

src/aspe.py implements the ASPE encryption scheme, including key generation, document and query encryption, and dummy dimension injection.

src/retrieval.py implements exact MIPS retrieval and the recall, latency, and DRAM metrics.

src/pipeline.py runs the headline three method comparison and the dummy dimension sweep, and writes the result tables.

src/plotting.py renders the result figures.

results/ holds the generated CSV tables and PNG figures from the most recent run.

## Running the experiment

Install the dependencies listed in requirements.txt, then run experiment.py from the repository root. The default run samples 5000 documents and 500 queries, evaluates the three methods, runs the dummy dimension sweep, and writes its outputs into the results folder. Command line flags allow overriding the corpus size, query count, the dummy dimension count and noise scale used in the headline comparison, and skipping the sweep entirely.

Embeddings are cached locally after the first run, so subsequent runs with the same sample and model skip re embedding and finish much faster.

## Outputs

experiment_results.csv and experiment_results.png contain the headline comparison of Raw, Vanilla ASPE, and ASPE with dummy dimensions across recall, latency, and DRAM overhead.

sweep_results.csv and sweep_results.png contain the dummy dimension and noise scale sweep, showing how recall degrades and score distortion grows as more or stronger noise is injected.
