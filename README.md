# GraphMind — GNN-Based Multi-Hop GraphRAG Retrieval

A GraphRAG retrieval system that scores node relevance with a Graph Neural
Network (GCN / GraphSAGE / GAT, ablated head-to-head), retrieves a minimal
connected subgraph via a Prize-Collecting Steiner Tree instead of a flat
top-k chunk list, and generates answers through an open-source local LLM
(Ollama) with a paid-API alternative (Anthropic) and tested graceful
fallback if neither is reachable.

Everything runs on a synthetic multi-domain knowledge graph by default, so
the full pipeline is reproducible with zero setup. Real HotpotQA integration
is wired in and will run automatically once network access allows it.

## Headline result (synthetic benchmark, 30-question held-out test set)

| Method | Hits@6 |
|---|---|
| BFS naive graph expansion | ~7-10% |
| TF-IDF Vector-RAG | ~3-10% |
| Random | ~17% |
| Noisy relation-lookup (strong, non-learned) | ~53-57% |
| GCN | ~10-26% |
| GAT | ~23-31% |
| **GraphSAGE (best architecture)** | **~60-66%** |

Exact numbers vary run-to-run (multi-seed, reported as mean ± std in
`results/metrics.json`) because the graph and question set are regenerated
each run. Run `python -m eval.run_ablation` yourself and quote your own
`results/metrics.json` — see [Metrics schema](#metrics-schema) below.

## Why this design

- **GraphSAGE, GCN, and GAT are ablated fairly** — identical features,
  identical train/val/test split, identical pretrain-then-fine-tune
  regime, 3 seeds each. The result (GraphSAGE beats GAT despite GAT having
  6x more parameters) is a real, reproducible finding about generalization
  under low-data fine-tuning, not a cherry-picked run.
- **A real vector-RAG baseline (TF-IDF), not a strawman.** It performs
  badly on multi-hop questions — which is the actual, literature-backed
  reason GraphRAG exists, not an assumption.
- **A strong non-learned baseline** (deterministic relation-path lookup
  with injected noise) that the GNN has to beat, not just a weak one.
  Beating it proves the GNN's structural learning adds value beyond
  simple signal-matching.
- **Noise-injected relation signal** (30% of relation labels corrupted
  before being used as a feature) simulates real entity/relation
  extraction errors, so the task isn't a trivial lookup.
- **Steiner Tree retrieval**, not fixed-radius or flat top-k — the same
  idea behind the G-Retriever architecture used in Neo4j/NVIDIA's
  open-source GraphRAG reference implementation.
- **Ollama-first LLM generation** — fully open-source, free, local. The
  Anthropic path exists for comparison but isn't the default.

## Project structure

```
graphmind/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py             central config, all env-driven
│   ├── graph_builder.py      synthetic multi-domain KG construction
│   ├── question_generator.py multi-hop question generation + train/val/test split
│   ├── features.py           query-conditioned node features + noisy relation-path signal
│   ├── models.py             GNN architecture factory (GCN / GraphSAGE / GAT)
│   ├── train.py               batched training loop + Hits@k / MRR metrics
│   ├── baselines.py          Random, BFS, TF-IDF, noisy relation-lookup baselines
│   ├── retrieval.py          GNN scores -> Steiner Tree subgraph retrieval
│   ├── llm_clients.py        Ollama / Anthropic generation with graceful fallback
│   ├── generator.py          ties retrieval + generation together
│   ├── hotpot_loader.py      real HotpotQA downloader (graceful fallback if blocked)
│   ├── pipeline.py           end-to-end setup used by eval/ and query.py
│   └── query.py              CLI — train a model, ask questions, see retrieval + generation
├── eval/
│   ├── run_ablation.py       full ablation + baselines, writes results/metrics.json
│   └── metrics_schema.json   empty JSON template — the schema results/metrics.json follows
├── notebooks/
│   └── graphmind_exploration.ipynb   exploratory version with plots (same logic as src/)
├── data/                     real HotpotQA gets cached here if downloaded
└── results/
    └── metrics.json          written by eval/run_ablation.py (git-ignored by default; commit it if you want to show real numbers)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### (Optional) Ollama, for real LLM generation
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
ollama serve
```
If Ollama isn't running, generation automatically falls back to an
extractive synthesizer — the pipeline never fails, it just reports
`"mode": "fallback_extractive"` in its output instead of faking a live
LLM response.

## Usage

**Run the full ablation + baselines, write metrics:**
```bash
python -m eval.run_ablation
```
Takes a few minutes (multi-seed training across 3 architectures). Writes
`results/metrics.json`.

**Ask questions interactively (trains GraphSAGE by default, then answers):**
```bash
python -m src.query
python -m src.query --arch GAT --n 5
```

**Inspect any module standalone:**
```bash
python -m src.graph_builder
python -m src.question_generator
python -m src.features
python -m src.models
python -m src.hotpot_loader
```

## Metrics schema

`eval/metrics_schema.json` is the empty template. `eval/run_ablation.py`
fills it in and writes the real result to `results/metrics.json`:

```json
{
  "run_id": "",
  "timestamp": "",
  "dataset": {
    "source": "",
    "n_nodes": 0,
    "n_edges": 0,
    "n_triples": 0,
    "n_questions_total": 0,
    "n_train": 0,
    "n_val": 0,
    "n_test": 0
  },
  "config": {
    "seed": 0,
    "ablation_seeds": [],
    "relation_noise_p": 0.0,
    "top_k": 0,
    "pretrain_epochs": 0,
    "finetune_epochs": 0
  },
  "baselines": {
    "random": { "hits_at_k": 0.0 },
    "bfs": { "hits_at_k": 0.0 },
    "tfidf_vector_rag": { "hits_at_k": 0.0 },
    "noisy_relation_lookup": { "hits_at_k": 0.0 }
  },
  "gnn_ablation": {
    "GCN": { "hits_at_k_mean": 0.0, "hits_at_k_std": 0.0, "mrr_mean": 0.0, "mrr_std": 0.0, "param_count": 0 },
    "GraphSAGE": { "hits_at_k_mean": 0.0, "hits_at_k_std": 0.0, "mrr_mean": 0.0, "mrr_std": 0.0, "param_count": 0 },
    "GAT": { "hits_at_k_mean": 0.0, "hits_at_k_std": 0.0, "mrr_mean": 0.0, "mrr_std": 0.0, "param_count": 0 }
  },
  "best_architecture": "",
  "improvement_over_tfidf_baseline_x": 0.0,
  "improvement_over_strongest_baseline_pct": 0.0,
  "llm_generation": {
    "provider_attempted": "",
    "mode_used": "",
    "n_queries_tested": 0
  },
  "notes": ""
}
```

