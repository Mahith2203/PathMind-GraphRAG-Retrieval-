"""End-to-end setup: builds the graph, questions, split, and episode tensors
that eval/run_ablation.py and src/query.py both depend on."""
import numpy as np

from . import config
from .graph_builder import build_graph, all_relations
from .question_generator import generate_multihop_questions, train_val_test_split, sample_synthetic_walk
from .features import FeatureBuilder


def build_everything(seed=None):
    seed = seed if seed is not None else config.SEED

    g_full, g_forward, triples = build_graph()
    relations = all_relations(triples)
    nodes = list(g_full.nodes())

    questions = generate_multihop_questions(g_forward, seed=999)
    train_q, val_q, test_q = train_val_test_split(questions)

    fb = FeatureBuilder(g_full, g_forward, relations, seed=seed)
    noise_rng = np.random.RandomState(7777)

    train_episodes = [fb.make_data(q, noise_rng) for q in train_q]
    val_episodes = [fb.make_data(q, noise_rng) for q in val_q]
    test_episodes = [fb.make_data(q, noise_rng) for q in test_q]

    pretrain_episodes = []
    for _ in range(config.N_PRETRAIN_EPISODES):
        synth_q = sample_synthetic_walk(g_forward, nodes, np.random.RandomState(np.random.randint(0, 10**8)))
        pretrain_episodes.append(fb.make_data(synth_q, noise_rng))

    return {
        "g_full": g_full, "g_forward": g_forward, "triples": triples, "relations": relations,
        "nodes": nodes, "node2idx": fb.node2idx, "idx2node": fb.idx2node,
        "questions": questions, "train_q": train_q, "val_q": val_q, "test_q": test_q,
        "feature_builder": fb, "in_dim": fb.feature_dim,
        "train_episodes": train_episodes, "val_episodes": val_episodes, "test_episodes": test_episodes,
        "pretrain_episodes": pretrain_episodes,
    }


if __name__ == "__main__":
    state = build_everything()
    print(f"Graph: {state['g_full'].number_of_nodes()} nodes")
    print(f"Questions: {len(state['questions'])} (train={len(state['train_q'])}, "
          f"val={len(state['val_q'])}, test={len(state['test_q'])})")
    print(f"Feature dim: {state['in_dim']}")
    print(f"Pretrain episodes: {len(state['pretrain_episodes'])}")
