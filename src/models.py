"""GNN architecture factory: GCN, GraphSAGE, and GAT, sharing an identical
head/training interface so they can be swapped and ablated fairly."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from . import config


class GNNRelevanceScorer(nn.Module):
    """Scores every node's relevance to a query given query-conditioned node features."""

    def __init__(self, in_dim, conv_type, hidden_dim=None, heads=None, layers=None, dropout=None):
        super().__init__()
        hidden_dim = hidden_dim or config.HIDDEN_DIM
        heads = heads or config.GAT_HEADS
        layers = layers or config.N_LAYERS
        dropout = config.DROPOUT if dropout is None else dropout

        if conv_type not in ("GCN", "GraphSAGE", "GAT"):
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self.conv_type = conv_type
        self.dropout = dropout
        self.convs = nn.ModuleList()

        def make_conv(i, o, last=False):
            if conv_type == "GCN":
                return GCNConv(i, o)
            if conv_type == "GraphSAGE":
                return SAGEConv(i, o)
            return GATConv(i, o, heads=heads, dropout=dropout, concat=not last)

        if conv_type == "GAT":
            self.convs.append(make_conv(in_dim, hidden_dim))
            for _ in range(layers - 2):
                self.convs.append(make_conv(hidden_dim * heads, hidden_dim))
            self.convs.append(make_conv(hidden_dim * heads, hidden_dim, last=True))
        else:
            self.convs.append(make_conv(in_dim, hidden_dim))
            for _ in range(layers - 1):
                self.convs.append(make_conv(hidden_dim, hidden_dim))

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, return_attn=False):
        attn_last = None
        for i, conv in enumerate(self.convs):
            if return_attn and self.conv_type == "GAT" and i == len(self.convs) - 1:
                x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
                attn_last = (ei, alpha)
            else:
                x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.out(x).squeeze(-1)
        return (logits, attn_last) if return_attn else logits

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    for conv_type in config.ARCHITECTURES:
        m = GNNRelevanceScorer(in_dim=28, conv_type=conv_type)
        print(f"{conv_type}: {m.param_count():,} parameters")
