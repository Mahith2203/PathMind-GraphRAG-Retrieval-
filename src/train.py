"""Training loop (batched, two-stage: pretrain on synthetic walks, then fine-tune
on the real train split) and retrieval-quality metrics (Hits@k, MRR)."""
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from . import config
from .models import GNNRelevanceScorer


def run_epoch(model, data_list, optimizer=None, pos_weight=None, batch_size=32):
    pos_weight = config.POS_WEIGHT if pos_weight is None else pos_weight
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=is_train)
    pw = torch.tensor(pos_weight)
    total, n_batches = 0.0, 0
    for batch in loader:
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(batch.x, batch.edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, batch.y, pos_weight=pw)
        if is_train:
            loss.backward()
            optimizer.step()
        total += loss.item()
        n_batches += 1
    return total / n_batches


def get_scores(model, data):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(data.x, data.edge_index)).numpy()


def hits_at_k(scores, gold_answer, idx2node, k=None):
    k = k or config.TOP_K
    ranked = np.argsort(-scores)[:k]
    return 1.0 if gold_answer in [idx2node[i] for i in ranked] else 0.0


def mrr_score(scores, gold_answer, idx2node):
    ranked = np.argsort(-scores)
    for rank, i in enumerate(ranked, start=1):
        if idx2node[i] == gold_answer:
            return 1.0 / rank
    return 0.0


def train_and_evaluate(conv_type, in_dim, pretrain_episodes, train_episodes, val_episodes,
                        test_episodes, idx2node, seed, pretrain_epochs=None, ft_epochs=None,
                        track_history=False):
    pretrain_epochs = pretrain_epochs or config.PRETRAIN_EPOCHS
    ft_epochs = ft_epochs or config.FINETUNE_EPOCHS

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GNNRelevanceScorer(in_dim, conv_type)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    history = {"pretrain_loss": [], "ft_loss": [], "ft_val_hits_at_k": []}

    for _ in range(pretrain_epochs):
        loss = run_epoch(model, pretrain_episodes, opt, batch_size=config.BATCH_SIZE_PRETRAIN)
        if track_history:
            history["pretrain_loss"].append(loss)

    ft_opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(ft_epochs):
        loss = run_epoch(model, train_episodes, ft_opt, batch_size=config.BATCH_SIZE_FINETUNE)
        if track_history:
            history["ft_loss"].append(loss)
            val_h = np.mean([hits_at_k(get_scores(model, d), d.answer, idx2node) for d in val_episodes])
            history["ft_val_hits_at_k"].append(val_h)

    test_h = [hits_at_k(get_scores(model, d), d.answer, idx2node) for d in test_episodes]
    test_mrr = [mrr_score(get_scores(model, d), d.answer, idx2node) for d in test_episodes]

    return {
        "model": model,
        "hits_at_k_mean": float(np.mean(test_h)),
        "mrr_mean": float(np.mean(test_mrr)),
        "history": history,
    }


def run_ablation(conv_types, in_dim, pretrain_episodes, train_episodes, val_episodes,
                  test_episodes, idx2node, seeds=None):
    """Trains each architecture across multiple seeds; returns mean/std summary."""
    seeds = seeds or config.ABLATION_SEEDS
    results = {}
    best_models = {}
    for conv_type in conv_types:
        h_list, mrr_list = [], []
        for s in seeds:
            r = train_and_evaluate(conv_type, in_dim, pretrain_episodes, train_episodes,
                                    val_episodes, test_episodes, idx2node, seed=s)
            h_list.append(r["hits_at_k_mean"])
            mrr_list.append(r["mrr_mean"])
            if s == seeds[0]:
                best_models[conv_type] = r["model"]
        results[conv_type] = {
            "hits_at_k_mean": float(np.mean(h_list)),
            "hits_at_k_std": float(np.std(h_list)),
            "mrr_mean": float(np.mean(mrr_list)),
            "mrr_std": float(np.std(mrr_list)),
            "seeds": seeds,
        }
    return results, best_models
