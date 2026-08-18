"""Generates multi-hop questions via random walks over the forward-relation graph,
recording the ground-truth relation path each question implicitly asks about.
"""
import numpy as np

from . import config

HOP_TEMPLATES = {
    2: "What is ultimately connected to {seed} through a two-step chain of relationships in AI history?",
    3: "Following a three-step chain of relationships starting from {seed}, what entity does it eventually lead to?",
}


def generate_multihop_questions(g_forward, n=None, seed=None, min_hops=None, max_hops=None):
    n = n or config.N_EVAL_QUESTIONS
    seed = seed if seed is not None else config.SEED
    min_hops = min_hops or config.MIN_HOPS
    max_hops = max_hops or config.MAX_HOPS

    rng = np.random.RandomState(seed)
    questions, seen_paths = [], set()
    nodelist = list(g_forward.nodes())
    attempts = 0
    while len(questions) < n and attempts < n * 60:
        attempts += 1
        seed_e = nodelist[rng.randint(len(nodelist))]
        hops = rng.randint(min_hops, max_hops + 1)
        path, rels, cur, ok = [seed_e], [], seed_e, True
        for _ in range(hops):
            succs = list(g_forward.successors(cur))
            if not succs:
                ok = False
                break
            v = succs[rng.randint(len(succs))]
            rel = g_forward[cur][v]["relation"]
            path.append(v)
            rels.append(rel)
            cur = v
        if not ok or len(set(path)) < len(path):
            continue
        path_key = tuple(path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        questions.append({
            "seed": seed_e,
            "answer": path[-1],
            "gold_path": path,
            "relations": rels,
            "hops": hops,
        })

    for q in questions:
        q["question_text"] = HOP_TEMPLATES[q["hops"]].format(seed=q["seed"])
    return questions


def train_val_test_split(questions, n_test=None, n_val=None, split_seed=None):
    n_test = n_test or config.N_TEST
    n_val = n_val or config.N_VAL
    split_seed = split_seed if split_seed is not None else config.SPLIT_SEED

    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(len(questions))
    shuffled = [questions[i] for i in perm]

    test_q = shuffled[:n_test]
    val_q = shuffled[n_test:n_test + n_val]
    train_q = shuffled[n_test + n_val:]
    return train_q, val_q, test_q


def sample_synthetic_walk(g_forward, nodes, rng):
    """Samples a single random-walk episode for pretraining (no held-out concerns —
    these never overlap with the eval question set by construction)."""
    seed = nodes[rng.randint(len(nodes))]
    hops = rng.randint(1, 4)
    path, rels, cur = [seed], [], seed
    for _ in range(hops):
        succs = list(g_forward.successors(cur))
        if not succs:
            break
        v = succs[rng.randint(len(succs))]
        rels.append(g_forward[cur][v]["relation"])
        path.append(v)
        cur = v
    return {"seed": seed, "answer": path[-1], "gold_path": path, "relations": rels}


if __name__ == "__main__":
    from .graph_builder import build_graph
    g_full, g_forward, triples = build_graph()
    questions = generate_multihop_questions(g_forward)
    train_q, val_q, test_q = train_val_test_split(questions)
    print(f"Generated {len(questions)} unique multi-hop questions")
    print(f"Split -> train={len(train_q)} val={len(val_q)} test={len(test_q)}")
