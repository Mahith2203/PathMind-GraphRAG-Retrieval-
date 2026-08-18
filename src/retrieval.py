"""GraphRAG retrieval: converts GNN relevance scores into a minimal connected
subgraph via a Prize-Collecting Steiner Tree approximation, instead of handing
the LLM a fixed-radius neighborhood or a flat top-k node list."""
import networkx as nx
from networkx.algorithms.approximation import steinertree

from . import config
from .train import get_scores


def gnn_retrieve(model, data, g_full, idx2node, top_k=None):
    top_k = top_k or config.TOP_K
    scores = get_scores(model, data)
    ranked = scores.argsort()[::-1][:top_k]
    top_nodes = [idx2node[i] for i in ranked]

    undirected = g_full.to_undirected()
    terminals = [n for n in top_nodes if n in undirected]
    try:
        tree = steinertree.steiner_tree(undirected, terminals)
    except Exception:
        tree = undirected.subgraph(terminals)

    return scores, top_nodes, tree


def tree_to_context_edges(tree, g_forward):
    """Linearizes a retrieved subgraph into (subject relation object) strings for LLM context."""
    edges = []
    for u, v in tree.edges():
        if g_forward.has_edge(u, v):
            rel = g_forward[u][v]["relation"]
        elif g_forward.has_edge(v, u):
            rel = g_forward[v][u]["relation"]
        else:
            rel = "related_to"
        edges.append(f"{u} --{rel}--> {v}")
    return edges
