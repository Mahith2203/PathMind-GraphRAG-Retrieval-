"""Runs the full architecture ablation (GCN/GraphSAGE/GAT, multi-seed) plus all
baselines on the identical held-out test set, and writes results to
results/metrics.json in the schema defined by eval/metrics_schema.json.

Usage:
    python -m eval.run_ablation
"""
import json
import os
import time
from datetime import datetime, timezone

from src import config
from src.pipeline import build_everything
from src.train import run_ablation
from src.baselines import run_all_baselines
from src.models import GNNRelevanceScorer


def main():
    print("Building graph, questions, and features...")
    state = build_everything()

    print(f"Graph: {state['g_full'].number_of_nodes()} nodes, "
          f"{state['g_full'].number_of_edges()} edges, {len(state['triples'])} triples")
    print(f"Questions: {len(state['questions'])} total "
          f"(train={len(state['train_q'])}, val={len(state['val_q'])}, test={len(state['test_q'])})")

    print("\nRunning GNN architecture ablation (this takes a few minutes)...")
    t0 = time.time()
    gnn_results, best_models = run_ablation(
        config.ARCHITECTURES, state["in_dim"],
        state["pretrain_episodes"], state["train_episodes"],
        state["val_episodes"], state["test_episodes"], state["idx2node"],
    )
    print(f"Ablation completed in {time.time() - t0:.1f}s")
    for arch, r in gnn_results.items():
        print(f"  {arch:<10}: Hits@{config.TOP_K} = {r['hits_at_k_mean']*100:5.1f}% "
              f"(+/- {r['hits_at_k_std']*100:.1f})   MRR = {r['mrr_mean']:.3f}")

    print("\nRunning baselines on the identical held-out test set...")
    baseline_results = run_all_baselines(
        state["test_q"], state["nodes"], state["g_full"], state["g_forward"], state["relations"],
    )
    for name, score in baseline_results.items():
        print(f"  {name:<45}: {score*100:5.1f}%")

    best_arch = max(gnn_results, key=lambda a: gnn_results[a]["hits_at_k_mean"])
    tfidf_score = baseline_results["TF-IDF Vector-RAG"]
    strongest_baseline = max(baseline_results.values())

    metrics = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": "synthetic_multi_domain_kg",
            "n_nodes": state["g_full"].number_of_nodes(),
            "n_edges": state["g_full"].number_of_edges(),
            "n_triples": len(state["triples"]),
            "n_questions_total": len(state["questions"]),
            "n_train": len(state["train_q"]),
            "n_val": len(state["val_q"]),
            "n_test": len(state["test_q"]),
        },
        "config": {
            "seed": config.SEED,
            "ablation_seeds": config.ABLATION_SEEDS,
            "relation_noise_p": config.RELATION_NOISE_P,
            "top_k": config.TOP_K,
            "pretrain_epochs": config.PRETRAIN_EPOCHS,
            "finetune_epochs": config.FINETUNE_EPOCHS,
        },
        "baselines": {
            "random": {"hits_at_k": baseline_results["Random"]},
            "bfs": {"hits_at_k": baseline_results["BFS (naive graph expansion)"]},
            "tfidf_vector_rag": {"hits_at_k": baseline_results["TF-IDF Vector-RAG"]},
            "noisy_relation_lookup": {"hits_at_k": baseline_results["Noisy Relation-Lookup (strong, non-learned)"]},
        },
        "gnn_ablation": {
            arch: {
                "hits_at_k_mean": r["hits_at_k_mean"],
                "hits_at_k_std": r["hits_at_k_std"],
                "mrr_mean": r["mrr_mean"],
                "mrr_std": r["mrr_std"],
                "param_count": GNNRelevanceScorer(state["in_dim"], arch).param_count(),
            }
            for arch, r in gnn_results.items()
        },
        "best_architecture": best_arch,
        "improvement_over_tfidf_baseline_x": round(gnn_results[best_arch]["hits_at_k_mean"] / tfidf_score, 2) if tfidf_score > 0 else None,
        "improvement_over_strongest_baseline_pct": round(
            (gnn_results[best_arch]["hits_at_k_mean"] / strongest_baseline - 1) * 100, 1
        ) if strongest_baseline > 0 else None,
        "llm_generation": {
            "provider_attempted": config.LLM_PROVIDER,
            "mode_used": "not_tested_in_this_run",
            "n_queries_tested": 0,
        },
        "notes": "Synthetic benchmark graph (see README limitations section). "
                 "Re-run against real HotpotQA (src/hotpot_loader.py) for literature-comparable numbers.",
    }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {config.METRICS_OUTPUT_PATH}")

    return metrics, best_models, state


if __name__ == "__main__":
    main()
