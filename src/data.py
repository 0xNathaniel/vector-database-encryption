"""FiQA2018 loading, subsampling, and embedding."""

import os
from hashlib import sha1

import numpy as np

from .config import Config, log


def load_fiqa(cfg: Config):
    """Load FiQA2018 via mteb and return (corpus, queries, qrels) as plain dicts.

    mteb 2.x layout: task.dataset[subset][split] -> {
        'corpus': HF Dataset {id, title, text},
        'queries': HF Dataset {id, text},
        'relevant_docs': {qid: {doc_id: relevance}},
    }
    """
    import mteb

    log(f"Loading dataset '{cfg.dataset}' via mteb ...")
    task = mteb.get_tasks(tasks=[cfg.dataset])[0]
    task.load_data()

    try:
        split = task.metadata.eval_splits[0]
    except Exception:
        split = "test"
    subset = "default" if "default" in task.dataset else next(iter(task.dataset))
    log(f"Using subset '{subset}', split '{split}'.")

    bundle = task.dataset[subset][split]
    corpus_ds, queries_ds, qrels_raw = (
        bundle["corpus"], bundle["queries"], bundle["relevant_docs"])

    corpus = {}
    for row in corpus_ds:
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        corpus[str(row["id"])] = (title + " " + text).strip()

    queries = {str(row["id"]): str(row["text"]).strip() for row in queries_ds}

    # Keep only positive-relevance judgments.
    qrels = {
        str(qid): {str(did): r for did, r in rels.items() if r > 0}
        for qid, rels in qrels_raw.items()
    }
    log(f"Full dataset: {len(corpus)} docs, {len(queries)} queries, "
        f"{len(qrels)} qrels.")
    return corpus, queries, qrels


def sample_subset(corpus, queries, qrels, cfg: Config, rng: np.random.Generator):
    """Sample a manageable subset while guaranteeing every gold doc of a
    sampled query stays in the sampled corpus (so qrels remain valid).

    Returns:
        doc_ids   : list[str]           corpus row order
        query_ids : list[str]           query row order
        gold      : list[np.ndarray]    gold corpus row indices per query
    """
    log("Sampling subset (preserving qrels validity) ...")

    valid_qids = [
        qid for qid in queries
        if qid in qrels and any(did in corpus for did in qrels[qid])
    ]
    rng.shuffle(valid_qids)
    sel_qids = valid_qids[: cfg.n_queries]

    gold_doc_ids = set()
    for qid in sel_qids:
        for did in qrels[qid]:
            if did in corpus:
                gold_doc_ids.add(did)

    if len(gold_doc_ids) > cfg.n_corpus:
        # Gold set alone exceeds the corpus budget: trim queries to fit.
        log(f"  {len(gold_doc_ids)} gold docs exceed corpus budget "
            f"{cfg.n_corpus}; trimming queries.")
        gold_doc_ids = set()
        kept = []
        for qid in sel_qids:
            extra = {did for did in qrels[qid] if did in corpus}
            if len(gold_doc_ids | extra) > cfg.n_corpus:
                break
            gold_doc_ids |= extra
            kept.append(qid)
        sel_qids = kept

    distractor_pool = [d for d in corpus if d not in gold_doc_ids]
    rng.shuffle(distractor_pool)
    n_fill = max(0, cfg.n_corpus - len(gold_doc_ids))
    doc_ids = list(gold_doc_ids) + distractor_pool[:n_fill]
    rng.shuffle(doc_ids)

    doc_pos = {did: i for i, did in enumerate(doc_ids)}
    gold = [
        np.array([doc_pos[did] for did in qrels[qid] if did in doc_pos], dtype=np.int64)
        for qid in sel_qids
    ]
    keep = [i for i, g in enumerate(gold) if g.size > 0]
    sel_qids = [sel_qids[i] for i in keep]
    gold = [gold[i] for i in keep]

    log(f"  Sampled {len(doc_ids)} docs, {len(sel_qids)} queries "
        f"({len(gold_doc_ids)} of the docs are gold).")
    return doc_ids, sel_qids, gold


def _sig(cfg: Config, doc_ids, query_ids) -> str:
    """Stable signature so cached embeddings match the exact sample."""
    h = sha1()
    h.update(cfg.model_name.encode())
    h.update("|".join(doc_ids).encode())
    h.update("|".join(query_ids).encode())
    return h.hexdigest()[:16]


def embed(corpus, queries, doc_ids, query_ids, gold, cfg: Config):
    """Embed sampled corpus + queries, L2-normalised so inner product == cosine.
    Cached to .npz keyed on the exact sample to skip re-embedding on reruns."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    sig = _sig(cfg, doc_ids, query_ids)
    cache = os.path.join(cfg.cache_dir, f"emb_{sig}.npz")

    if os.path.exists(cache):
        log(f"Loading cached embeddings from {cache} ...")
        data = np.load(cache, allow_pickle=True)
        return (data["corpus_emb"].astype(np.float32),
                data["query_emb"].astype(np.float32),
                list(data["gold"]))

    from sentence_transformers import SentenceTransformer
    log(f"Loading embedding model '{cfg.model_name}' ...")
    model = SentenceTransformer(cfg.model_name)

    log(f"Embedding corpus ({len(doc_ids)} docs) ...")
    corpus_emb = model.encode(
        [corpus[d] for d in doc_ids],
        batch_size=128, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)

    log(f"Embedding queries ({len(query_ids)} queries) ...")
    query_emb = model.encode(
        [queries[q] for q in query_ids],
        batch_size=128, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)

    log(f"Caching embeddings to {cache} ...")
    np.savez(cache, corpus_emb=corpus_emb, query_emb=query_emb,
             gold=np.array(gold, dtype=object))
    return corpus_emb, query_emb, gold
