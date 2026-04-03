"""Shared data structures for subgraph-based training."""

import torch
from torch.utils.data import Dataset


class SubgraphData:
    """Container for a single extracted subgraph around a target node."""

    def __init__(self, x, edge_index, edge_attr, y, target_idx,
                 in_degree, out_degree, in_ratio):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.target_idx = target_idx
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.in_ratio = in_ratio


class SubgraphDataset(Dataset):
    def __init__(self, subgraphs):
        self.subgraphs = subgraphs

    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, idx):
        return self.subgraphs[idx]


def collate_subgraphs(batch):
    return batch
