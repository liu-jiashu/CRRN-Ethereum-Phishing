"""
BiSRN: Bidirectional Selective Representation Network

Three-channel directional encoding for Ethereum phishing detection:
  - Forward encoder: outgoing fund flow patterns
  - Backward encoder: incoming fund flow patterns
  - Neighbor Relation Network (NRN): inter-neighbor topology
"""

import torch
import torch.nn as nn
import numpy as np


class EdgeAttentionLayer(nn.Module):
    """Edge-attention message passing layer with learnable edge importance scores."""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_transform = nn.Linear(node_dim, hidden_dim)
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.LeakyReLU(0.2), nn.Linear(hidden_dim, 1))
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim), nn.LeakyReLU(0.2))
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2))
        self.hidden_dim = hidden_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr, edge_mask=None):
        num_nodes, num_edges = x.shape[0], edge_index.shape[1]
        if num_edges == 0:
            h = self.node_transform(x) if x.shape[-1] != self.hidden_dim else x
            return h, torch.zeros(0, device=x.device), torch.zeros(0, device=x.device)
        h = self.node_transform(x)
        src, tgt = edge_index[0], edge_index[1]
        edge_input = torch.cat([h[src], h[tgt], edge_attr], dim=-1)
        raw_scores = self.edge_scorer(edge_input).squeeze(-1)
        edge_scores = torch.sigmoid(raw_scores).clamp(0.01, 0.99)
        new_edge_mask = edge_scores if edge_mask is None else 0.8 * edge_mask + 0.2 * edge_scores
        msg_input = torch.cat([h[src], edge_attr], dim=-1)
        messages = self.message_net(msg_input) * new_edge_mask.unsqueeze(-1)
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=x.device, dtype=h.dtype)
        aggregated.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, self.hidden_dim), messages)
        new_h = self.update_net(torch.cat([h, aggregated], dim=-1))
        return new_h, new_edge_mask, edge_scores


class SRNEncoder(nn.Module):
    """Stacked edge-attention layers for one direction (forward or backward)."""

    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_dim if i == 0 else hidden_dim
            self.layers.append(EdgeAttentionLayer(in_dim, edge_dim, hidden_dim))

    def forward(self, x, edge_index, edge_attr, target_idx):
        h, edge_mask = x, None
        all_masks = []
        for layer in self.layers:
            h, edge_mask, scores = layer(h, edge_index, edge_attr, edge_mask)
            all_masks.append(edge_mask)
        return h[target_idx], edge_mask, all_masks, h


class NeighborRelationNetwork(nn.Module):
    """Models pairwise relationships among a target node's neighbors."""

    def __init__(self, hidden_dim=128, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pair_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim))
        self.relation_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True)
        self.relation_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim))
        self.stats_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_features, edge_index, target_idx, _):
        device = node_features.device
        src, tgt = edge_index[0], edge_index[1]
        out_neighbors = tgt[src == target_idx].unique()
        in_neighbors = src[tgt == target_idx].unique()
        neighbors = torch.cat([out_neighbors, in_neighbors]).unique()
        if len(neighbors) < 2:
            return self.stats_encoder(torch.zeros(4, device=device)).unsqueeze(0)
        neighbors = neighbors[:min(len(neighbors), 20)]
        n = len(neighbors)
        edge_set = set()
        for i in range(edge_index.shape[1]):
            edge_set.add((edge_index[0, i].item(), edge_index[1, i].item()))
        pair_feats, pairs = [], []
        for i in range(n):
            for j in range(i + 1, n):
                ni, nj = neighbors[i].item(), neighbors[j].item()
                has_edge = 1.0 if ((ni, nj) in edge_set or (nj, ni) in edge_set) else 0.0
                feat = torch.cat([node_features[neighbors[i]], node_features[neighbors[j]],
                                  torch.tensor([has_edge], device=device)])
                pair_feats.append(feat)
                pairs.append(has_edge)
        if not pair_feats:
            return self.stats_encoder(torch.zeros(4, device=device)).unsqueeze(0)
        pair_feats = torch.stack(pair_feats)
        pair_encoded = self.pair_encoder(pair_feats).unsqueeze(0)
        attended, _ = self.relation_attention(pair_encoded, pair_encoded, pair_encoded)
        relation_repr = self.relation_aggregator(attended.mean(dim=1))
        pairs_arr = np.array(pairs)
        stats = torch.tensor([
            pairs_arr.mean(), pairs_arr.std() if len(pairs_arr) > 1 else 0,
            float(n) / 20.0, float(len(pairs_arr)) / max(n * (n - 1) / 2, 1)
        ], device=device, dtype=torch.float32)
        return (relation_repr.squeeze(0) + self.stats_encoder(stats)).unsqueeze(0)


class BiSRN(nn.Module):
    """
    Bidirectional Selective Representation Network.

    Args:
        node_dim: Input node feature dimension (default: 20)
        edge_dim: Edge feature dimension (default: 4)
        hidden_dim: Hidden representation dimension (default: 128)
        num_layers: Number of edge-attention layers per encoder (default: 3)
        dropout: Dropout rate for classifier (default: 0.2)
        target_keep_ratio: Target mean attention score for sparse regularization (default: 0.5)
        use_nrn: Whether to use the Neighbor Relation Network (default: True)
    """

    def __init__(self, node_dim, edge_dim=4, hidden_dim=128,
                 num_layers=3, dropout=0.2, target_keep_ratio=0.5, use_nrn=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_keep_ratio = target_keep_ratio
        self.use_nrn = use_nrn
        self.forward_encoder = SRNEncoder(node_dim, edge_dim, hidden_dim, num_layers)
        self.backward_encoder = SRNEncoder(node_dim, edge_dim, hidden_dim, num_layers)
        if use_nrn:
            self.nrn = NeighborRelationNetwork(hidden_dim)
            cls_dim = hidden_dim * 3
        else:
            cls_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(cls_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        self.forward_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 2))
        self.backward_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 2))

    def forward(self, x, edge_index, edge_attr, target_idx, mode='fusion'):
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        h_forward, mask_fwd, _, all_h = self.forward_encoder(x, edge_index, edge_attr, target_idx)
        h_backward, mask_bwd, _, _ = self.backward_encoder(x, edge_index_rev, edge_attr, target_idx)
        masks = {'forward': mask_fwd, 'backward': mask_bwd}
        if mode == 'forward_only':
            return self.forward_classifier(h_forward), masks
        if mode == 'backward_only':
            return self.backward_classifier(h_backward), masks
        if h_forward.dim() == 1:
            h_forward = h_forward.unsqueeze(0)
        if h_backward.dim() == 1:
            h_backward = h_backward.unsqueeze(0)
        if self.use_nrn:
            h_nrn = self.nrn(all_h, edge_index, target_idx, None)
            if h_nrn.dim() == 1:
                h_nrn = h_nrn.unsqueeze(0)
            h_fused = torch.cat([h_forward, h_backward, h_nrn], dim=-1)
        else:
            h_fused = torch.cat([h_forward, h_backward], dim=-1)
        return self.classifier(h_fused), masks

    def compute_sparse_loss(self, masks_dict):
        """Sparse attention regularization loss."""
        loss, count = 0.0, 0
        for key in ['forward', 'backward']:
            mask = masks_dict.get(key)
            if mask is not None and mask.numel() > 0:
                kr = mask.mean()
                if not torch.isnan(kr):
                    loss += (kr - self.target_keep_ratio).pow(2)
                    count += 1
        return loss / max(count, 1)
