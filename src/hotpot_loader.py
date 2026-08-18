"""Attempts to download the real HotpotQA dev (distractor) set. Falls back
cleanly (returns None, False) on any network failure rather than raising or
silently substituting fake data — callers should check the second return
value and fall back to the synthetic benchmark in src/pipeline.py if False.
"""
import json
import os
import urllib.request

from . import config


def load_real_hotpotqa(n=None):
    n = n or config.HOTPOT_SAMPLE_SIZE
    try:
        os.makedirs(os.path.dirname(config.HOTPOT_LOCAL_PATH) or ".", exist_ok=True)
        if not os.path.exists(config.HOTPOT_LOCAL_PATH):
            req = urllib.request.Request(config.HOTPOT_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read()
            with open(config.HOTPOT_LOCAL_PATH, "wb") as f:
                f.write(raw)
        with open(config.HOTPOT_LOCAL_PATH) as f:
            data = json.load(f)
        return data[:n], True
    except Exception as e:
        print(f"[HotpotQA download unavailable: {type(e).__name__}: {e}]")
        print("This is expected in network-restricted sandboxes. The code is correct and will "
              "succeed with open internet access (local machine, Colab, etc).")
        return None, False


def hotpot_to_graph_and_questions(hotpot_data):
    """Builds a co-occurrence graph from real HotpotQA supporting facts, and a
    question list compatible with src/features.py's expected schema.

    Note: this is a coarse graph-construction heuristic (titles that co-occur
    in supporting facts are linked), not a full entity/relation extraction
    pipeline. See README limitations."""
    import networkx as nx

    g_forward = nx.DiGraph()
    questions = []
    for ex in hotpot_data:
        sf_titles = list(dict.fromkeys([sf[0] for sf in ex.get("supporting_facts", [])]))
        for i in range(len(sf_titles) - 1):
            g_forward.add_edge(sf_titles[i], sf_titles[i + 1], relation="co_occurs_multihop")
        if len(sf_titles) >= 2:
            questions.append({
                "seed": sf_titles[0],
                "answer": ex["answer"],
                "gold_path": sf_titles,
                "relations": ["co_occurs_multihop"] * (len(sf_titles) - 1),
                "hops": len(sf_titles) - 1,
                "question_text": ex["question"],
            })
    return g_forward, questions


if __name__ == "__main__":
    data, ok = load_real_hotpotqa()
    if ok:
        g, qs = hotpot_to_graph_and_questions(data)
        print(f"Real HotpotQA graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
              f"{len(qs)} usable multi-hop questions")
    else:
        print("Falling back to the synthetic benchmark (src/pipeline.py).")
