"""Feature engineering for node relevance scoring.

Each node's feature vector is: [base identity embedding, is_seed flag,
forward distance from seed, backward distance from seed, relation-path
match score]. The relation-path match score is computed against a NOISE-
INJECTED version of the question's true relation path, simulating the
error rate of a real entity/relation-extraction pipeline (default 30%
per-hop substitution with a random wrong relation). This is what forces
the GNN to learn robustness rather than perform an exact lookup.
"""
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from . import config


def relation_match_scores(seed_entity, relation_path, g_forward):
    """BFS-with-relation-constraint: score(n) = fraction of relation_path steps
    satisfiable by SOME path from seed_entity ending at n."""
    scores = {n: 0.0 for n in g_forward.nodes()}
    frontier = {seed_entity}
    for i, rel in enumerate(relation_path, start=1):
        nxt = set()
        for u in frontier:
            for v in g_forward.successors(u):
                if g_forward[u][v]["relation"] == rel:
                    nxt.add(v)
        for v in nxt:
            scores[v] = max(scores[v], i / len(relation_path))
        if not nxt:
            break
        frontier = nxt
    return scores


class FeatureBuilder:
    """Stateful builder: precomputes per-node base embeddings once, then builds
    query-conditioned episode features on demand."""

    def __init__(self, g_full, g_forward, all_relations, feat_dim=None, seed=None):
        self.g_full = g_full
        self.g_forward = g_forward
        self.g_forward_rev = g_forward.reverse()
        self.all_relations = all_relations
        self.nodes = list(g_full.nodes())
        self.node2idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx2node = {i: n for n, i in self.node2idx.items()}
        self.edge_index = torch.tensor(
            [[self.node2idx[u], self.node2idx[v]] for u, v in g_full.edges()],
            dtype=torch.long
        ).t().contiguous()

        feat_dim = feat_dim or config.RANDOM_FEAT_DIM
        seed = seed if seed is not None else config.SEED
        self.node_base_feat = {}
        for n in self.nodes:
            rr = np.random.RandomState(abs(hash(n)) % (2 ** 32))
            self.node_base_feat[n] = rr.normal(0, 1, feat_dim).astype(np.float32)

    def _inject_noise(self, true_rels, rng, p_noise):
        return [rng.choice(self.all_relations) if rng.random() < p_noise else r for r in true_rels]

    def build_episode(self, question, noise_rng, p_noise=None):
        p_noise = config.RELATION_NOISE_P if p_noise is None else p_noise
        seed_entity = question["seed"]
        gold_set = set(question["gold_path"])
        noisy_r = self._inject_noise(question["relations"], noise_rng, p_noise)
        rel_scores = relation_match_scores(seed_entity, noisy_r, self.g_forward)

        feats, labels = [], []
        for n in self.nodes:
            base = self.node_base_feat[n]
            is_seed = 1.0 if n == seed_entity else 0.0
            try:
                dist = nx.shortest_path_length(self.g_forward, seed_entity, n)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                dist = 10
            try:
                dist_rev = nx.shortest_path_length(self.g_forward_rev, seed_entity, n)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                dist_rev = 10
            feats.append(np.concatenate([
                base, [is_seed, min(dist, 5) / 5.0, min(dist_rev, 5) / 5.0, rel_scores[n]]
            ]))
            labels.append(1.0 if n in gold_set else 0.0)

        x = torch.tensor(np.stack(feats), dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.float)
        return x, y

    def make_data(self, question, noise_rng, p_noise=None):
        x, y = self.build_episode(question, noise_rng, p_noise)
        return Data(
            x=x, edge_index=self.edge_index, y=y,
            seed=question["seed"], answer=question["answer"],
            question_text=question.get("question_text", ""),
            gold=set(question["gold_path"]),
        )

    @property
    def feature_dim(self):
        return self.node_base_feat[self.nodes[0]].shape[0] + 4


if __name__ == "__main__":
    from .graph_builder import build_graph, all_relations
    from .question_generator import generate_multihop_questions

    g_full, g_forward, triples = build_graph()
    fb = FeatureBuilder(g_full, g_forward, all_relations(triples))
    questions = generate_multihop_questions(g_forward)
    rng = np.random.RandomState(7777)
    data = fb.make_data(questions[0], rng)
    print(f"Feature dim: {fb.feature_dim}")
    print(f"Episode x shape: {data.x.shape}, y sum (gold nodes): {data.y.sum().item()}")
