from dataclasses import dataclass
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, global_mean_pool


@dataclass
class GNNConfig:
    node_dim: int = 20
    edge_dim: int = 20
    hidden_dim: int = 192
    num_layers: int = 6
    heads: int = 4
    elo_dim: int = 64
    side_info_dim: int = 0
    dropout: float = 0.1


class Maia2GNNModel(nn.Module):
    def __init__(self, num_moves: int, elo_vocab_size: int, cfg: GNNConfig):
        super().__init__()
        self.cfg = cfg
        self.num_moves = num_moves

        self.node_proj = nn.Linear(cfg.node_dim, cfg.hidden_dim)
        self.edge_proj = nn.Linear(cfg.edge_dim, cfg.hidden_dim)
        self.convs = nn.ModuleList(
            [
                GATv2Conv(
                    cfg.hidden_dim,
                    cfg.hidden_dim // cfg.heads,
                    heads=cfg.heads,
                    concat=True,
                    edge_dim=cfg.hidden_dim,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.norms = nn.ModuleList([BatchNorm(cfg.hidden_dim) for _ in range(cfg.num_layers)])
        self.dropout = nn.Dropout(cfg.dropout)

        self.elo_embedding = nn.Embedding(elo_vocab_size, cfg.elo_dim)
        self.elo_proj = nn.Linear(cfg.elo_dim * 2, cfg.hidden_dim)

        self.edge_policy = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.side_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, num_moves + 6 + 6 + 1 + 64 + 64),
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, batch):
        x = self.node_proj(batch.x)
        edge_attr = self.edge_proj(batch.edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            res = x
            x = conv(x, batch.edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + res

        graph_embed = global_mean_pool(x, batch.batch)
        elo_emb = torch.cat(
            [self.elo_embedding(batch.elo_self), self.elo_embedding(batch.elo_oppo)], dim=-1
        )
        elo_ctx = self.elo_proj(elo_emb)
        context = torch.cat([graph_embed, elo_ctx], dim=-1)

        src, dst = batch.edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_logits = self.edge_policy(edge_input).squeeze(-1)
        policy_logits = edge_logits.view(-1, self.num_moves)

        side_logits = self.side_head(context)
        value = self.value_head(context).squeeze(-1)
        return policy_logits, side_logits, value


def from_pretrained(checkpoint_path: str, device: str = "cpu", cfg: Optional[GNNConfig] = None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    num_moves = payload["num_moves"]
    elo_vocab_size = payload["elo_vocab_size"]
    cfg_dict = payload.get("gnn_cfg", None)
    if cfg is None and cfg_dict is not None:
        cfg = GNNConfig(**cfg_dict)
    cfg = cfg or GNNConfig()
    model = Maia2GNNModel(num_moves=num_moves, elo_vocab_size=elo_vocab_size, cfg=cfg)
    model.load_state_dict(payload["model_state_dict"])
    if device == "gpu":
        model = model.cuda()
    return model
