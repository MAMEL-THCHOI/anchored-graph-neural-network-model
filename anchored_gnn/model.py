"""Anchored grain graph neural network."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def weighted_pool(
    values: torch.Tensor, volume: torch.Tensor, graph_id: torch.Tensor, graph_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted_sum = torch.zeros(graph_count, values.shape[1], dtype=values.dtype, device=values.device)
    weighted_sum.index_add_(0, graph_id, values * volume[:, None])
    weight_sum = torch.zeros(graph_count, 1, dtype=values.dtype, device=values.device)
    weight_sum.index_add_(0, graph_id, volume[:, None])
    mean = weighted_sum / weight_sum.clamp_min(1.0e-12)
    maximum = torch.full_like(weighted_sum, -1.0e30)
    maximum.scatter_reduce_(0, graph_id[:, None].expand_as(values), values, reduce="amax", include_self=True)
    return mean, maximum


class GINELayer(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.edge = nn.Linear(hidden, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)
        )
        self.norm = nn.LayerNorm(hidden)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(
        self, node_state: torch.Tensor, edge_index: torch.Tensor, edge_embedding: torch.Tensor
    ) -> torch.Tensor:
        source, target = edge_index
        message = torch.relu(node_state[source] + self.edge(edge_embedding))
        aggregate = torch.zeros_like(node_state).index_add_(0, target, message)
        count = torch.zeros(
            node_state.shape[0], 1, dtype=node_state.dtype, device=node_state.device
        ).index_add_(0, target, torch.ones(len(target), 1, dtype=node_state.dtype, device=node_state.device))
        update = self.mlp((1.0 + self.eps) * node_state + aggregate / count.clamp_min(1.0))
        return self.norm(node_state + update)


class AnchoredGrainGraph(nn.Module):
    """Joint hardening and Lankford-coefficient surrogate.

    The Lankford decoder starts from two externally supplied anchor estimates
    and learns both their graph-dependent mixing and a three-direction residual.
    """

    def __init__(
        self,
        coefficient_count: int = 36,
        lambda_init: float = 0.5,
        hidden: int = 48,
        dropout: float = 0.05,
        message_layers: int = 2,
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(220, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.edge_encoder = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.message_layers = nn.ModuleList(GINELayer(hidden, dropout) for _ in range(message_layers))
        self.layer_gates = nn.Parameter(torch.ones(message_layers))
        pooled = 2 * hidden
        self.node_coefficient_head = nn.Linear(hidden, coefficient_count)
        self.global_coefficient_head = nn.Sequential(
            nn.Linear(pooled, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, coefficient_count)
        )
        self.r_head = nn.Sequential(
            nn.Linear(pooled, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 3)
        )
        self.stress_aux_head = nn.Linear(pooled, 126)
        self.flow_aux_head = nn.Linear(pooled, 9)
        self.lambda_head = nn.Linear(pooled, 1)
        for layer in (self.node_coefficient_head, self.global_coefficient_head[-1], self.r_head[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.lambda_head.weight)
        clipped = min(max(float(lambda_init), 0.05), 0.95)
        nn.init.constant_(self.lambda_head.bias, math.log(clipped / (1.0 - clipped)))

    def forward(
        self, batch: dict[str, torch.Tensor | int], log_anchor_lower: torch.Tensor, log_anchor_upper: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        node_state = self.node_encoder(batch["x"])
        edge_embedding = self.edge_encoder(batch["edge_attr"])
        for gate, layer in zip(self.layer_gates, self.message_layers):
            candidate = layer(node_state, batch["edge_index"], edge_embedding)
            node_state = node_state + gate * (candidate - node_state)
        mean, maximum = weighted_pool(
            node_state, batch["volume"], batch["graph_id"], int(batch["graph_count"])
        )
        embedding = torch.cat([mean, maximum], dim=1)
        node_coefficients, _ = weighted_pool(
            self.node_coefficient_head(node_state),
            batch["volume"],
            batch["graph_id"],
            int(batch["graph_count"]),
        )
        coefficients = self.global_coefficient_head(embedding) + node_coefficients
        mixing = torch.sigmoid(self.lambda_head(embedding)).expand(-1, 3)
        correction = self.r_head(embedding)
        log_r = mixing * log_anchor_upper + (1.0 - mixing) * log_anchor_lower + correction
        return {
            "coefficients": coefficients,
            "log_r": log_r,
            "mixing": mixing,
            "correction": correction,
            "stress_aux": self.stress_aux_head(embedding),
            "flow_aux": self.flow_aux_head(embedding),
        }
