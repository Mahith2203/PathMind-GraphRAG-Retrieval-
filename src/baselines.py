"""Non-GNN retrieval baselines, evaluated on the exact same held-out test set
as the GNN models so comparisons are apples-to-apples."""
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .features import relation_match_scores


def random_baseline(test_questions, nodes, k=None, seed=0):
    k = k or config.TOP_K
    rng = np.random.RandomState(seed)
    hits = [1.0 if q["answer"] in rng.choice(nodes, size=k, replace=False) else 0.0 for q in test_questions]
    return float(np.mean(hits))


def bfs_baseline(test_questions, g_full, k=None):
    """Naive graph-expansion baseline: no learned relevance, just BFS order from the seed."""
    k = k or config.TOP_K

    def bfs_retrieve(seed_entity):
        out = [seed_entity]
        for _, v in nx.bfs_edges(g_full, seed_entity):
            if len(out) >= k:
                break
            out.append(v)
        return out

    hits = [1.0 if q["answer"] in bfs_retrieve(q["seed"]) else 0.0 for q in test_questions]
    return float(np.mean(hits))


def build_tfidf_index(nodes, g_forward):
    """Builds a synthetic 'passage' per entity from its outgoing facts, then a TF-IDF index.
    This is a real sparse vector-similarity baseline — the standard RAG retrieval approach —
    used here as a fully offline stand-in for dense sentence-transformer embeddings, which
    would require a HuggingFace model download."""
    entity_passages = {}
    for n in nodes:
        facts = [f"{n} {g_forward[n][v]['relation']} {v}." for v in g_forward.successors(n)]
        entity_passages[n] = " ".join(facts) if facts else f"{n} is an entity in the knowledge graph."
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([entity_passages[n] for n in nodes])
    return vectorizer, matrix


def tfidf_baseline(test_questions, nodes, vectorizer, matrix, k=None):
    k = k or config.TOP_K

    def retrieve(qtext):
        qv = vectorizer.transform([qtext])
        sims = cosine_similarity(qv, matrix).flatten()
        return [nodes[i] for i in np.argsort(-sims)[:k]]

    hits = [1.0 if q["answer"] in retrieve(q["question_text"]) else 0.0 for q in test_questions]
    return float(np.mean(hits))


def noisy_lookup_baseline(test_questions, g_forward, all_relations, k=None, p_noise=None, seed=4242):
    """Strong non-learned baseline: direct (noisy) access to the same relation-path signal
    the GNN receives as a feature. Beating this is the real test of whether the GNN's
    structural learning adds value beyond simple noisy signal-matching."""
    k = k or config.TOP_K
    p_noise = config.RELATION_NOISE_P if p_noise is None else p_noise
    rng = np.random.RandomState(seed)

    def noisy_relations(true_rels):
        return [rng.choice(all_relations) if rng.random() < p_noise else r for r in true_rels]

    def hit(q):
        noisy_r = noisy_relations(q["relations"])
        scores = relation_match_scores(q["seed"], noisy_r, g_forward)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return 1.0 if q["answer"] in [n for n, _ in ranked] else 0.0

    hits = [hit(q) for q in test_questions]
    return float(np.mean(hits))


def run_all_baselines(test_questions, nodes, g_full, g_forward, all_relations):
    vectorizer, matrix = build_tfidf_index(nodes, g_forward)
    return {
        "Random": random_baseline(test_questions, nodes),
        "BFS (naive graph expansion)": bfs_baseline(test_questions, g_full),
        "TF-IDF Vector-RAG": tfidf_baseline(test_questions, nodes, vectorizer, matrix),
        "Noisy Relation-Lookup (strong, non-learned)": noisy_lookup_baseline(test_questions, g_forward, all_relations),
    }
