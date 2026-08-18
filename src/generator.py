"""Ties retrieval + LLM generation together: given a trained model and a
question, retrieves the relevant subgraph and synthesizes an answer."""
from .retrieval import gnn_retrieve, tree_to_context_edges
from .llm_clients import generate_answer


def answer_question(model, data, question, g_full, g_forward, idx2node, top_k=None):
    scores, top_nodes, tree = gnn_retrieve(model, data, g_full, idx2node, top_k=top_k)
    context_edges = tree_to_context_edges(tree, g_forward)
    result = generate_answer(question["question_text"], context_edges, question["answer"])
    return {
        "question": question["question_text"],
        "retrieved_nodes": top_nodes,
        "context_edges": context_edges,
        "generation": result,
        "true_answer": question["answer"],
    }
