<div align="center">
  <h1>Encrypted Vector Database Benchmark</h1>
  <h3>To Encrypt or Not to Encrypt? Evaluating the Utility-Security Trade-offs of Distance-Preserving Encryption and Fully Homomorphic Encryption in RAG Vector Databases</h3>

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?logo=numpy&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-MiniLM--L6--v2-orange?logo=huggingface&logoColor=white)
![MTEB](https://img.shields.io/badge/MTEB-FiQA2018-yellow)
![TenSEAL](https://img.shields.io/badge/TenSEAL-CKKS-red)
![Pandas](https://img.shields.io/badge/Pandas-Results-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-11557C?logo=plotly&logoColor=white)

</div>

## Overview

Vector databases used in Retrieval Augmented Generation systems typically store raw embeddings in plaintext, which leaks semantic information about the underlying documents if the database is compromised. Distance Preserving Encryption schemes such as Asymmetric Scalar Product Preserving Encryption, or ASPE, attempt to protect those embeddings while still allowing similarity search to run directly on the encrypted vectors.

This repository contains a micro benchmark that measures what encryption actually costs and protects, by comparing five retrieval setups on a real financial question answering dataset, spanning plaintext, three Distance Preserving Encryption schemes, and Fully Homomorphic Encryption.

1. Raw, plaintext embeddings, used as the utility baseline.
2. Vanilla ASPE, embeddings encrypted with a secret invertible matrix and no added noise.
3. ASPE with dummy dimensions, where random noise dimensions are injected into the vectors before encryption to obscure the true inner products.
4. DCPE, a Scale and Perturb approximate DPE that multiplies vectors by a secret scalar and adds a bounded random perturbation to the corpus so exact distances cannot be recovered.
5. FHE using the CKKS scheme through TenSEAL, the cryptographic gold standard, benchmarked on a tiny subset and extrapolated because full encrypted retrieval is prohibitively slow.

## Why this matters

ASPE encrypts a document vector p as M transpose times p, and encrypts a query vector q as M inverse times q, for a secret invertible matrix M. The inner product of the encrypted vectors works out to p transpose times q, which is exactly the plaintext inner product. This means Vanilla ASPE preserves ranking perfectly and should retrieve identically to the plaintext baseline. Its weakness is purely cryptographic, since a known plaintext attack can recover M.

Dummy dimensions break that exact preservation on purpose. Random noise values are appended to both document and query vectors before encryption, adding an extra term to every inner product score that depends only on the noise. This perturbs the ranking and therefore costs retrieval utility, in exchange for making the encrypted scores harder to interpret. The size and number of these dummy dimensions control how much utility is traded for how much obfuscation, which is the central trade off this benchmark quantifies.

DCPE takes a different route. Instead of a matrix, it scales each vector by a secret scalar lambda and adds a fresh bounded Gaussian perturbation to every stored document vector. Because the perturbation is random per vector and never revealed, the mapping from plaintext to ciphertext is no longer an exact deterministic linear function, which is exactly what gives DCPE resistance to the linear known plaintext attack below, at the cost of a small ranking error.

FHE with CKKS encrypts each vector under a public key so that inner products can be computed homomorphically on the ciphertexts without ever decrypting. It leaks nothing about the plaintext and no linear known plaintext attack can recover the secret key, but it is orders of magnitude slower and larger than plaintext or DPE.

A note on the security result. The empirical security metric is the Mean Squared Error of a least squares known plaintext attack that, given a set of plaintext and ciphertext pairs, fits the best linear ciphertext to plaintext map and applies it to reconstruct held out plaintexts. Because Vanilla ASPE and even ASPE with dummy dimensions remain exact linear bijections, this attack recovers the original embeddings essentially perfectly, so their reconstruction error collapses to zero once the attacker has at least as many pairs as the ciphertext dimension. Dummy dimensions obscure the scores, not the plaintext. DCPE is the only DPE scheme here whose additive per vector perturbation leaves an irreducible reconstruction error, and FHE is unbreakable by this attack by construction, reported as infinity.

## Experiment design

Dataset: FiQA2018, a financial domain question answering retrieval task, loaded through the MTEB library. A subset of 5000 corpus documents and 500 queries is sampled, with every gold document for a sampled query guaranteed to remain in the sampled corpus so the relevance judgments stay valid.

Embedding model: Sentence Transformers all MiniLM L6 v2, with embeddings L2 normalized so that inner product equals cosine similarity.

Retrieval: exact Maximum Inner Product Search implemented directly with NumPy matrix multiplication and top k selection, with no external vector database or approximate index involved. FHE is the exception: because homomorphic retrieval is prohibitively slow, its latency and memory footprint are measured on a tiny subset of queries and documents and extrapolated to the full corpus, with recall taken equal to Raw since CKKS retains plaintext precision.

Metrics collected for each method:

Recall at 5 and Recall at 10, comparing retrieved document ids against ground truth relevance judgments.

Average query latency in milliseconds, timed with high resolution perf counters over the full score and top k computation for all queries.

DRAM overhead, the in memory size of the stored encrypted corpus, which captures the cost of the extra dummy dimensions for DPE and the large ciphertext expansion for FHE.

Empirical security, the Mean Squared Error of a linear known plaintext attack that fits a ciphertext to plaintext map from a sample of known pairs and reconstructs held out plaintexts. The number of known pairs must exceed the ciphertext dimension for the attack to be well posed; with fewer pairs the linear system is underdetermined and even Raw shows a spurious non zero error, so the default uses 1000 pairs.

In addition to the five method headline comparison, the script sweeps the dummy dimension count and noise scale over a grid of values and records how recall and score distortion change together, producing a direct picture of the utility versus security trade off.

## Repository structure

`experiment.py` is the entry point that wires the pipeline together and exposes command line options.

`src/config.py` holds the experiment configuration and a small logging helper.

`src/data.py` loads FiQA2018 through MTEB, samples a valid subset, and embeds it with Sentence Transformers, caching embeddings to disk.

`src/aspe.py` implements the ASPE encryption scheme, including key generation, document and query encryption, and dummy dimension injection.

`src/dcpe.py` implements the DCPE Scale and Perturb scheme, with secret scalar key generation, bounded perturbation of document vectors, and query scaling.

`src/fhe.py` implements the CKKS FHE baseline through TenSEAL, measuring latency and ciphertext footprint on a subset and extrapolating, and degrading gracefully to a skipped row if TenSEAL is not installed.

`src/security.py` implements the linear known plaintext attack and its reconstruction Mean Squared Error.

`src/retrieval.py` implements exact MIPS retrieval and the recall, latency, and DRAM metrics.

`src/pipeline.py` runs the headline five method comparison and the dummy dimension sweep, and writes the result tables.

`src/plotting.py` renders the four standalone metric figures and the sweep figure.

`results/` holds the generated CSV tables and PNG figures from the most recent run.

## Running the experiment

Install the dependencies listed in requirements.txt, then run experiment.py from the repository root. The default run samples 5000 documents and 500 queries, evaluates the five methods, runs the dummy dimension sweep, and writes its outputs into the results folder. Command line flags allow overriding the corpus size, query count, the dummy dimension count and noise scale, the DCPE perturbation scale, the known plaintext attack sample size, and skipping the FHE baseline or the sweep entirely.

TenSEAL is the one heavyweight dependency and can be awkward to install on some platforms. If it is unavailable the FHE method is skipped automatically, its row is recorded as not available, and the rest of the benchmark still completes. Use the `--no-fhe` flag to skip it explicitly.

Embeddings are cached locally after the first run, so subsequent runs with the same sample and model skip re embedding and finish much faster.

## Outputs

experiment_results.csv contains the headline comparison of Raw, Vanilla ASPE, ASPE with dummy dimensions, DCPE, and FHE across recall, latency, DRAM overhead, and empirical security.

plot_recall.png, plot_latency.png, plot_dram.png, and plot_security.png are four standalone, paper ready figures, one per metric, with the latency, DRAM, and security plots on a log scale so the FHE outliers and the near zero security values remain legible.

sweep_results.csv and sweep_results.png contain the dummy dimension and noise scale sweep, showing how recall degrades and score distortion grows as more or stronger noise is injected.
