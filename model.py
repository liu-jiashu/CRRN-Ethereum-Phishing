"""Counterparty Role-Relation Network (CRRN).

This module follows Section IV of the paper: account-role inference, two
independently parameterized role-aware transaction encoders, the
role-conditioned Inter-Neighbor Relation Encoder (INRE), branch-specific
fusion projections, and a shared binary classifier.
"""

import torch
import torch.nn as nn


def _init_linear_layers(module):
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight, gain=0.5)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class RoleConditionedMessageLayer(nn.Module):
    """One direction- and layer-specific gated message-passing layer."""

    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + edge_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        _init_linear_layers(self)

    def forward(self, h, role_repr, edge_index, edge_attr):
        num_nodes = h.shape[0]
        if edge_index.shape[1] == 0:
            aggregated = h.new_zeros((num_nodes, self.hidden_dim))
            updated = self.update_mlp(torch.cat([h, aggregated], dim=-1))
            return updated, h.new_zeros(0)

        source, target = edge_index
        gate_input = torch.cat(
            [role_repr[source], edge_attr, h[source], h[target]], dim=-1
        )
        gates = torch.sigmoid(self.score_mlp(gate_input).squeeze(-1))
        messages = self.message_mlp(
            torch.cat([h[source], edge_attr], dim=-1)
        ) * gates.unsqueeze(-1)
        aggregated = h.new_zeros((num_nodes, self.hidden_dim))
        aggregated.scatter_add_(
            0, target.unsqueeze(-1).expand(-1, self.hidden_dim), messages
        )
        updated = self.update_mlp(torch.cat([h, aggregated], dim=-1))
        return updated, gates


class RoleAwareTransactionEncoder(nn.Module):
    """Incoming or outgoing CRRN transaction encoder."""

    def __init__(self, role_dim, edge_dim, hidden_dim, num_layers):
        super().__init__()
        self.role_projection = nn.Linear(role_dim, hidden_dim)
        self.layers = nn.ModuleList(
            RoleConditionedMessageLayer(hidden_dim, edge_dim)
            for _ in range(num_layers)
        )
        _init_linear_layers(self.role_projection)

    def forward(self, h0, role_probs, edge_index, edge_attr, target_idx):
        role_repr = self.role_projection(role_probs)
        h = h0
        gates_by_layer = []
        for layer in self.layers:
            h, gates = layer(h, role_repr, edge_index, edge_attr)
            gates_by_layer.append(gates)
        return h[target_idx], gates_by_layer[-1], gates_by_layer, h


class InterNeighborRelationEncoder(nn.Module):
    """Role-conditioned symmetric relation encoder from Eqs. (8)-(10)."""

    def __init__(self, role_dim=3, hidden_dim=128, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(role_dim * 2 + 1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.relation_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.aggregate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        _init_linear_layers(self)

    def forward(self, role_probs, edge_index, target_idx):
        device = role_probs.device
        if edge_index.shape[1] == 0:
            return self.aggregate_mlp(role_probs.new_zeros(self.hidden_dim))

        source, target = edge_index
        outgoing = target[source == target_idx]
        incoming = source[target == target_idx]
        neighbors = torch.unique(torch.cat([outgoing, incoming]), sorted=True)
        neighbors = neighbors[neighbors != target_idx]
        if neighbors.numel() < 2:
            return self.aggregate_mlp(role_probs.new_zeros(self.hidden_dim))

        edge_set = {
            (int(source[index].item()), int(target[index].item()))
            for index in range(edge_index.shape[1])
        }
        pair_features = []
        for left_index in range(neighbors.numel()):
            for right_index in range(left_index + 1, neighbors.numel()):
                left = int(neighbors[left_index].item())
                right = int(neighbors[right_index].item())
                connected = float(
                    (left, right) in edge_set or (right, left) in edge_set
                )
                left_role = role_probs[left]
                right_role = role_probs[right]
                pair_features.append(
                    torch.cat(
                        [
                            left_role + right_role,
                            torch.abs(left_role - right_role),
                            torch.tensor(
                                [connected], device=device, dtype=role_probs.dtype
                            ),
                        ]
                    )
                )

        pair_tensor = torch.stack(pair_features)
        pair_embeddings = self.pair_mlp(pair_tensor).unsqueeze(0)
        attended, _ = self.relation_attention(
            pair_embeddings, pair_embeddings, pair_embeddings, need_weights=False
        )
        pooled = attended.mean(dim=1)
        return self.aggregate_mlp(pooled).squeeze(0)


class CRRN(nn.Module):
    """Counterparty Role-Relation Network described in Section IV."""

    def __init__(
        self,
        node_dim,
        edge_dim=4,
        hidden_dim=128,
        num_layers=3,
        role_dim=3,
        dropout=0.2,
        target_mean_gate=0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_mean_gate = target_mean_gate

        self.role_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, role_dim),
        )
        self.initial_projection = nn.Linear(node_dim, hidden_dim)
        self.incoming_encoder = RoleAwareTransactionEncoder(
            role_dim, edge_dim, hidden_dim, num_layers
        )
        self.outgoing_encoder = RoleAwareTransactionEncoder(
            role_dim, edge_dim, hidden_dim, num_layers
        )
        self.inre = InterNeighborRelationEncoder(role_dim, hidden_dim)

        self.incoming_fusion = nn.Linear(hidden_dim, hidden_dim)
        self.outgoing_fusion = nn.Linear(hidden_dim, hidden_dim)
        self.relation_fusion = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        _init_linear_layers(self)

    def forward(self, x, edge_index, edge_attr, target_idx):
        role_probs = torch.softmax(self.role_mlp(x), dim=-1)
        initial_states = self.initial_projection(x)

        incoming, gates_in, _, _ = self.incoming_encoder(
            initial_states, role_probs, edge_index, edge_attr, target_idx
        )
        reversed_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
        outgoing, gates_out, _, _ = self.outgoing_encoder(
            initial_states, role_probs, reversed_edges, edge_attr, target_idx
        )
        relation = self.inre(role_probs, edge_index, target_idx)

        fused = torch.cat(
            [
                self.incoming_fusion(incoming),
                self.outgoing_fusion(outgoing),
                self.relation_fusion(relation),
            ],
            dim=-1,
        )
        logit = self.classifier(fused).reshape(1)
        gates = {"incoming": gates_in, "outgoing": gates_out}
        return logit, gates

    def compute_gate_loss(self, gates_by_direction):
        """Eq. (15): squared deviation of final-layer mean gates."""
        penalties = []
        for direction in ("incoming", "outgoing"):
            gates = gates_by_direction.get(direction)
            if gates is not None and gates.numel() > 0:
                penalties.append((gates.mean() - self.target_mean_gate).pow(2))
        if not penalties:
            return next(self.parameters()).new_zeros(())
        return torch.stack(penalties).mean()
