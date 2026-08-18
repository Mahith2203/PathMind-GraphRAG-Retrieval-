"""Builds the multi-domain synthetic knowledge graph.

Node domains and relation templates are intentionally structured (not fully
random) so the graph has realistic branching factor and multi-hop paths,
similar in spirit to a small slice of a real knowledge graph.
"""
import random
import networkx as nx

from . import config

DOMAINS = {
    "AI_Pioneers": ["Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Andrew Ng", "Fei-Fei Li",
                    "Demis Hassabis", "Ilya Sutskever", "Andrej Karpathy", "John Hopfield", "Alan Turing"],
    "AI_Concepts": ["Backpropagation", "Deep Learning", "Convolutional Neural Network", "Transformer",
                     "Attention Mechanism", "Reinforcement Learning", "Large Language Model",
                     "Graph Neural Network", "Diffusion Model", "Generative Adversarial Network"],
    "Orgs": ["DeepMind", "OpenAI", "Google Brain", "Meta AI", "Stanford AI Lab", "Coursera",
             "Bletchley Park", "MILA", "Anthropic", "NVIDIA"],
    "Artifacts": ["AlphaFold", "AlexNet", "GPT", "ImageNet", "Turing Machine", "Analytical Engine",
                  "AlphaGo", "BERT", "ResNet", "Stable Diffusion"],
    "Outcomes": ["Protein Folding", "Modern Computer", "Nobel Prize in Physics", "Turing Award",
                 "Image Classification Breakthrough", "Game of Go Mastery", "Natural Language Understanding"],
}

REL_TEMPLATES = {
    ("AI_Pioneers", "AI_Concepts"): ["pioneered", "researched", "advanced"],
    ("AI_Pioneers", "AI_Pioneers"): ["advised", "collaborated_with", "mentored"],
    ("AI_Pioneers", "Orgs"): ["founded", "worked_at", "led"],
    ("Orgs", "Artifacts"): ["built", "released", "developed"],
    ("Artifacts", "AI_Concepts"): ["uses", "based_on", "demonstrates"],
    ("Artifacts", "Outcomes"): ["achieved", "solved", "enabled"],
    ("AI_Concepts", "AI_Concepts"): ["enabled", "influenced", "extends"],
    ("AI_Concepts", "Outcomes"): ["led_to", "influenced"],
    ("AI_Pioneers", "Outcomes"): ["won"],
}


def build_triples(n_triples=None, seed=None):
    """Generates (head, relation, tail) triples across the domain templates."""
    n_triples = n_triples or config.N_TRIPLES
    seed = seed if seed is not None else config.SEED
    rng = random.Random(seed)
    triples, seen = [], set()
    attempts = 0
    while len(triples) < n_triples and attempts < n_triples * 25:
        attempts += 1
        d1, d2 = rng.choice(list(REL_TEMPLATES.keys()))
        h = rng.choice(DOMAINS[d1])
        t = rng.choice(DOMAINS[d2])
        if h == t:
            continue
        r = rng.choice(REL_TEMPLATES[(d1, d2)])
        key = (h, r, t)
        if key in seen:
            continue
        seen.add(key)
        triples.append(key)
    return triples


def build_graph(triples=None):
    """Builds the bidirectional NetworkX graph (forward + inverse edges) used for
    GNN message passing, plus the forward-only graph used for distance features
    and relation-path matching."""
    triples = triples or build_triples()

    g_full = nx.DiGraph()      # forward + inverse edges, for GNN message passing
    g_forward = nx.DiGraph()   # forward-only, for distance/relation-path features
    for h, r, t in triples:
        g_full.add_edge(h, t, relation=r)
        g_full.add_edge(t, h, relation=f"inverse_{r}")
        g_forward.add_edge(h, t, relation=r)

    return g_full, g_forward, triples


def all_relations(triples):
    return sorted(set(r for _, r, _ in triples))


if __name__ == "__main__":
    triples = build_triples()
    g_full, g_forward, _ = build_graph(triples)
    print(f"Triples: {len(triples)}")
    print(f"Full graph (bidirectional): {g_full.number_of_nodes()} nodes, {g_full.number_of_edges()} edges")
    print(f"Forward-only graph: {g_forward.number_of_nodes()} nodes, {g_forward.number_of_edges()} edges")
    print(f"Weakly connected components: {nx.number_weakly_connected_components(g_full)}")
