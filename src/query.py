"""CLI entry point: trains the best-known architecture (GraphSAGE by default)
on the synthetic benchmark and answers a sample of held-out test questions,
printing retrieval + generation results.

Usage:
    python -m src.query
    python -m src.query --arch GAT --n 3
"""
import argparse

from . import config
from .pipeline import build_everything
from .train import train_and_evaluate
from .generator import answer_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="GraphSAGE", choices=config.ARCHITECTURES)
    parser.add_argument("--n", type=int, default=3, help="number of test questions to demo")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    print("Building graph and features...")
    state = build_everything()

    print(f"Training {args.arch} (pretrain + fine-tune)...")
    result = train_and_evaluate(
        args.arch, state["in_dim"], state["pretrain_episodes"], state["train_episodes"],
        state["val_episodes"], state["test_episodes"], state["idx2node"], seed=args.seed,
    )
    print(f"Test Hits@{config.TOP_K}: {result['hits_at_k_mean']*100:.1f}%   Test MRR: {result['mrr_mean']:.3f}\n")

    model = result["model"]
    for q, data in list(zip(state["test_q"], state["test_episodes"]))[:args.n]:
        out = answer_question(model, data, q, state["g_full"], state["g_forward"], state["idx2node"])
        print(f"Q: {out['question']}")
        print(f"  Retrieved: {out['retrieved_nodes']}")
        print(f"  Generation mode: {out['generation']['mode']}")
        print(f"  Answer: {out['generation']['text']}")
        print(f"  True answer: {out['true_answer']}")
        print("-" * 100)


if __name__ == "__main__":
    main()
