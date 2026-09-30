"""
CRRN training on MulDiGraph dataset.

Usage:
    python train_muldigraph.py --data_dir /path/to/data --epochs 40 --n_seeds 3
"""

import torch
import numpy as np
import pickle
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, argparse, time, warnings
warnings.filterwarnings('ignore')

from data_utils import SubgraphData, SubgraphDataset, collate_subgraphs
from metrics import compute_metrics, compute_full_metrics
from train_utils import evaluate, train_one_model


class MulDiGraphLoader:
    """从networkx MultiDiGraph加载数据, 计算20维特征, 提取子图"""
    
    NODE_DIM = 20

    def __init__(self, pkl_path, k_hop=2, max_nodes=50, device='cuda'):
        self.pkl_path = pkl_path
        self.k_hop = k_hop; self.max_nodes = max_nodes
        self.device = torch.device(device)
        self.adjacency = defaultdict(list)
        self.reverse_adj = defaultdict(list)
        self.edge_data = defaultdict(list)
        self.node_features = {}
        self.labels = {}
        self.phisher_nodes = []
        self.normal_nodes = []
        self.node_to_id = {}
        self.in_degree = {}
        self.out_degree = {}
        self.feature_mean = None
        self.feature_std = None
        self.features_normalized = False
        self.edge_feature_stats = {}

    def load(self):
        print(f"Loading MulDiGraph from {self.pkl_path}...")
        t0 = time.time()
        with open(self.pkl_path, 'rb') as f:
            G = pickle.load(f)
        print(f"   done Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges ({time.time()-t0:.1f}s)")

        # ============ 建立节点ID映射 ============
        print("   Building node ID mapping...", end=" ")
        all_nodes = sorted(G.nodes())
        self.node_to_id = {nd: i for i, nd in enumerate(all_nodes)}
        print(f"done ({len(all_nodes):,} nodes)")

        # ============ 标签 ============
        for nd in all_nodes:
            nid = self.node_to_id[nd]
            isp = G.nodes[nd].get('isp', 0)
            if isp == 1:
                self.labels[nid] = 1
                self.phisher_nodes.append(nid)
            else:
                self.labels[nid] = 0
                self.normal_nodes.append(nid)

        # ============ 构建邻接表 + 边特征 ============
        print("   Building adjacency + edge features...", end=" ")
        t0 = time.time()
        
        # 收集所有amount和timestamp用于归一化
        all_amounts = []
        all_timestamps = []
        all_gas = []
        all_gas_prices = []
        for u, v, data in G.edges(data=True):
            amt = data.get('amount', 0)
            ts = data.get('timestamp', 0)
            if amt is not None: all_amounts.append(float(amt))
            if ts is not None: all_timestamps.append(float(ts))
            all_gas.append(float(data.get('gas', 0) or 0))
            all_gas_prices.append(float(data.get('gas_price', 0) or 0))
        
        all_amounts = np.array(all_amounts, dtype=np.float64)
        all_timestamps = np.array(all_timestamps, dtype=np.float64)
        
        log_amounts = np.log1p(np.abs(all_amounts) + 1e-8)
        val_mean, val_std = log_amounts.mean(), log_amounts.std() + 1e-8
        ts_min, ts_max = all_timestamps.min(), all_timestamps.max() + 1e-8
        log_gas = np.log1p(np.abs(np.asarray(all_gas)) + 1e-8)
        gas_mean, gas_std = log_gas.mean(), log_gas.std() + 1e-8
        log_gas_prices = np.log1p(np.abs(np.asarray(all_gas_prices)) + 1e-8)
        gas_price_mean = log_gas_prices.mean()
        gas_price_std = log_gas_prices.std() + 1e-8
        self.edge_feature_stats = {
            'amount_log_mean': float(val_mean),
            'amount_log_std': float(val_std),
            'timestamp_min': float(ts_min),
            'timestamp_max': float(ts_max),
            'gas_log_mean': float(gas_mean),
            'gas_log_std': float(gas_std),
            'gas_price_log_mean': float(gas_price_mean),
            'gas_price_log_std': float(gas_price_std),
        }
        
        print(f"(stats done)...", end=" ")
        
        # 每笔交易保留为独立边实例。
        for u, v, data in G.edges(data=True):
            src_id = self.node_to_id[u]
            tgt_id = self.node_to_id[v]
            amt = float(data.get('amount', 0) or 0)
            ts = float(data.get('timestamp', 0) or 0)
            gas = float(data.get('gas', 0) or 0)
            gas_price = float(data.get('gas_price', 0) or 0)
            val_norm = np.clip((np.log1p(abs(amt) + 1e-8) - val_mean) / val_std, -5, 5)
            ts_norm = (ts - ts_min) / (ts_max - ts_min)
            gas_norm = np.clip(
                (np.log1p(abs(gas) + 1e-8) - gas_mean) / gas_std, -5, 5
            )
            gas_price_norm = np.clip(
                (np.log1p(abs(gas_price) + 1e-8) - gas_price_mean)
                / gas_price_std,
                -5,
                5,
            )
            self.adjacency[src_id].append(tgt_id)
            self.reverse_adj[tgt_id].append(src_id)
            self.edge_data[(src_id, tgt_id)].append(
                [val_norm, ts_norm, gas_norm, gas_price_norm]
            )
        
        print(f"done ({time.time()-t0:.1f}s)")

        # ============ 计算20维特征 ============
        print("   Computing 20-dim features...", end=" ")
        t0 = time.time()
        self._compute_features(G)
        print(f"done ({time.time()-t0:.1f}s)")

        print(f"\n   Phisher: {len(self.phisher_nodes):,}, Normal: {len(self.normal_nodes):,}")
        
        del G  # 释放大图内存
        return self

    def _compute_features(self, G):
        """计算20维节点特征 (和B4E版一致)"""
        
        # ---- 基础度和金额统计 (直接从networkx计算) ----
        node_in_degree = defaultdict(int)
        node_out_degree = defaultdict(int)
        node_in_amounts = defaultdict(list)
        node_out_amounts = defaultdict(list)
        node_all_timestamps = defaultdict(list)
        node_in_timestamps = defaultdict(list)
        node_out_timestamps = defaultdict(list)
        
        for u, v, data in G.edges(data=True):
            uid = self.node_to_id[u]
            vid = self.node_to_id[v]
            amt = float(data.get('amount', 0) or 0)
            ts = float(data.get('timestamp', 0) or 0)
            
            node_out_degree[uid] += 1
            node_in_degree[vid] += 1
            node_out_amounts[uid].append(amt)
            node_in_amounts[vid].append(amt)
            node_out_timestamps[uid].append(ts)
            node_in_timestamps[vid].append(ts)
            node_all_timestamps[uid].append(ts)
            node_all_timestamps[vid].append(ts)
        
        # ---- 计算每个节点的20维特征 ----
        for nd in sorted(G.nodes()):
            nid = self.node_to_id[nd]
            feat = np.zeros(self.NODE_DIM, dtype=np.float32)
            
            in_deg = node_in_degree.get(nid, 0)
            out_deg = node_out_degree.get(nid, 0)
            self.in_degree[nid] = in_deg
            self.out_degree[nid] = out_deg
            
            # [0-1] degree
            feat[0] = np.log1p(in_deg)
            feat[1] = np.log1p(out_deg)
            
            # [2-3] in amount sum/mean
            in_amts = node_in_amounts.get(nid, [])
            if in_amts:
                feat[2] = np.log1p(abs(sum(in_amts)) + 1e-8)
                feat[3] = np.log1p(abs(np.mean(in_amts)) + 1e-8)
            
            # [4-5] out amount sum/mean
            out_amts = node_out_amounts.get(nid, [])
            if out_amts:
                feat[4] = np.log1p(abs(sum(out_amts)) + 1e-8)
                feat[5] = np.log1p(abs(np.mean(out_amts)) + 1e-8)
            
            # [6] net_flow
            feat[6] = sum(in_amts) - sum(out_amts)
            
            # [7] in_ratio
            total_deg = in_deg + out_deg
            if total_deg > 0: feat[7] = in_deg / total_deg
            
            # [8-9] unique neighbors
            feat[8] = np.log1p(len(set(self.reverse_adj.get(nid, []))))
            feat[9] = np.log1p(len(set(self.adjacency.get(nid, []))))
            
            # [10] out_ratio
            out_in_ratio = out_deg / (in_deg + 1.0)
            feat[10] = out_in_ratio
            
            # [11] out_in_log
            feat[11] = np.log1p(out_in_ratio)
            
            # [12-19] 时序特征
            all_ts = sorted(node_all_timestamps.get(nid, []))
            out_ts = sorted(node_out_timestamps.get(nid, []))
            in_ts_list = sorted(node_in_timestamps.get(nid, []))
            
            if len(all_ts) >= 2:
                ts_arr = np.array(all_ts)
                duration = ts_arr[-1] - ts_arr[0]
                
                # [12] activity_duration
                feat[12] = np.log1p(max(duration, 0))
                
                # [13] tx_rate
                if duration > 0:
                    feat[13] = np.log1p(len(all_ts) / (duration / 3600 + 1e-8))
                else:
                    feat[13] = np.log1p(len(all_ts))
                
                # [14] burst_score
                if duration > 0 and len(ts_arr) >= 3:
                    max_burst = max(
                        np.searchsorted(ts_arr, ts_arr[i] + 3600, side='right') - i
                        for i in range(len(ts_arr))
                    )
                    feat[14] = max_burst / len(all_ts)
                else:
                    feat[14] = 1.0
                
                # [15] inter_tx_cv
                deltas = np.diff(ts_arr)
                if len(deltas) > 1 and deltas.mean() > 0:
                    feat[15] = min(deltas.std() / (deltas.mean() + 1e-8), 10.0)
                
                # [16] fast_forward_ratio
                if in_ts_list and out_ts:
                    out_arr = np.array(out_ts)
                    fast_count = 0
                    for its in in_ts_list:
                        idx = np.searchsorted(out_arr, its)
                        if idx < len(out_arr) and (out_arr[idx] - its) <= 3600:
                            fast_count += 1
                    feat[16] = fast_count / len(in_ts_list)
                
                # [17] amount_gini
                all_amts = np.abs(np.array(
                    node_in_amounts.get(nid, []) + node_out_amounts.get(nid, [])
                )) + 1e-8
                if len(all_amts) > 1 and all_amts.sum() > 0:
                    sorted_a = np.sort(all_amts)
                    n = len(sorted_a)
                    feat[17] = np.clip(
                        (2 * (np.arange(1, n+1) * sorted_a).sum() / (n * sorted_a.sum())) - (n+1)/n,
                        0, 1)
                
                # [18] address_reuse_ratio
                total_tx = len(in_amts) + len(out_amts)
                counterparties = set(self.adjacency.get(nid, [])) | set(
                    self.reverse_adj.get(nid, [])
                )
                counterparties.discard(nid)
                n_unique = len(counterparties)
                if total_tx > 0:
                    feat[18] = 1.0 - min(n_unique / total_tx, 1.0)
                
                # [19] dormancy_ratio
                if duration > 0 and len(deltas) > 0:
                    feat[19] = min(deltas.max() / (duration + 1e-8), 1.0)
            
            self.node_features[nid] = feat

    def normalize_node_features(self, train_indices):
        """Fit Z-score parameters on training targets and transform all graph nodes."""
        if self.features_normalized:
            return
        train_features = np.asarray(
            [self.node_features[node_id] for node_id in train_indices], dtype=np.float32
        )
        self.feature_mean = train_features.mean(axis=0)
        self.feature_std = train_features.std(axis=0) + 1e-8
        for node_id in self.node_features:
            self.node_features[node_id] = np.clip(
                (self.node_features[node_id] - self.feature_mean) / self.feature_std,
                -5,
                5,
            )
        self.features_normalized = True

    def extract_subgraph(self, node_id):
        visited = {node_id}; frontier = [node_id]
        for _ in range(self.k_hop):
            new_frontier = []
            for n in frontier:
                for nb in self.adjacency.get(n, []):
                    if nb not in visited and len(visited) < self.max_nodes:
                        visited.add(nb); new_frontier.append(nb)
                for nb in self.reverse_adj.get(n, []):
                    if nb not in visited and len(visited) < self.max_nodes:
                        visited.add(nb); new_frontier.append(nb)
            frontier = new_frontier
        subgraph_nodes = sorted(visited)
        local_map = {n: i for i, n in enumerate(subgraph_nodes)}
        if node_id not in local_map: return None
        target_idx = local_map[node_id]
        edges_src, edges_tgt, edge_feats = [], [], []
        for src in subgraph_nodes:
            pair_offsets = defaultdict(int)
            for tgt in self.adjacency.get(src, []):
                if tgt in local_map:
                    edges_src.append(local_map[src]); edges_tgt.append(local_map[tgt])
                    feats = self.edge_data.get((src, tgt), [[0, 0, 0, 0]])
                    offset = pair_offsets[tgt]
                    edge_feats.append(np.clip(feats[min(offset, len(feats) - 1)], -5, 5))
                    pair_offsets[tgt] += 1
        x = torch.tensor([self.node_features.get(n, np.zeros(self.NODE_DIM))
                          for n in subgraph_nodes], dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(
            [edges_src, edges_tgt], dtype=torch.long, device=self.device
        ).reshape(2, -1)
        edge_attr = torch.tensor(
            edge_feats, dtype=torch.float32, device=self.device
        ).reshape(-1, 4)
        y = torch.tensor(self.labels.get(node_id, 0), dtype=torch.long, device=self.device)
        return SubgraphData(x, edge_index, edge_attr, y, target_idx)

    def prepare_cached_dataset(self, node_indices, desc=""):
        subgraphs = []
        for nid in tqdm(node_indices, desc=desc, leave=False):
            sg = self.extract_subgraph(nid)
            if sg is not None: subgraphs.append(sg)
        return SubgraphDataset(subgraphs)

    def get_train_test_split(self, train_ratio=0.7):
        rng = np.random.default_rng(42)
        phisher = self.phisher_nodes.copy(); normal = self.normal_nodes.copy()
        rng.shuffle(phisher); rng.shuffle(normal)
        normal = normal[:min(len(normal), len(phisher) * 3)]
        n_train_p = int(len(phisher) * train_ratio)
        n_train_n = int(len(normal) * train_ratio)
        train = phisher[:n_train_p] + normal[:n_train_n]
        test = phisher[n_train_p:] + normal[n_train_n:]
        rng.shuffle(train); rng.shuffle(test)
        self.normalize_node_features(train)
        return train, test


def main():
    parser = argparse.ArgumentParser(description="CRRN on MulDiGraph")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pkl_name', type=str, default='MulDiGraph.pkl')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--k_hop', type=int, default=2)
    parser.add_argument('--max_nodes', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--seeds', type=str, default='42,123,456')
    parser.add_argument('--target_mean_gate', type=float, default=0.5)
    parser.add_argument('--gate_lambda', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seeds = [int(s) for s in args.seeds.split(',')][:args.n_seeds]
    pkl_path = os.path.join(args.data_dir, args.pkl_name)
    print(f"Device: {device}, Seeds: {seeds}")

    # Data loading
    t0 = time.time()
    loader = MulDiGraphLoader(
        pkl_path, k_hop=args.k_hop, max_nodes=args.max_nodes, device=str(device)
    )
    loader.load()
    train_indices, test_indices = loader.get_train_test_split()
    print(f"Train: {len(train_indices):,}, Test: {len(test_indices):,}")

    # Subgraph extraction
    train_dataset = loader.prepare_cached_dataset(train_indices, "Train")
    test_dataset = loader.prepare_cached_dataset(test_indices, "Test")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_subgraphs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2,
                             shuffle=False, collate_fn=collate_subgraphs)

    train_labels = [sg.y.item() for sg in train_dataset.subgraphs]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    weight_pos = min(n_neg / n_pos * 0.5, 5.0) if n_pos > 0 else 1.0
    class_weight = torch.tensor([1.0, weight_pos], device=device)
    print(f"Class weight: [1.0, {weight_pos:.2f}], Data time: {(time.time()-t0)/60:.1f} min")

    # Multi-seed training
    all_f1s, all_probs = [], []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Training model {i+1}/{len(seeds)} (seed={seed})")
        t1 = time.time()
        state, model, _ = train_one_model(
            seed, train_loader, class_weight,
            loader.NODE_DIM, args, device)
        checkpoint = {
            'model_state_dict': state,
            'seed': seed,
            'config': vars(args).copy(),
            'node_feature_mean': loader.feature_mean.copy(),
            'node_feature_std': loader.feature_std.copy(),
            'edge_feature_stats': loader.edge_feature_stats.copy(),
            'train_target_ids': list(train_indices),
            'test_target_ids': list(test_indices),
        }
        torch.save(checkpoint, f'crrn_muldigraph_seed{seed}.pt')
        probs, labels = evaluate(model, test_loader, device)
        all_probs.append(probs)
        final_metrics = compute_metrics(probs, labels)
        all_f1s.append(final_metrics['f1'])
        print(f"  seed={seed}: F1={final_metrics['f1']:.4f} ({(time.time()-t1)/60:.1f} min)")

    # Results
    print(f"\n{'='*60}")
    print(f"Mean F1: {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")

    compute_full_metrics(
        all_probs[0], labels, f"CRRN-single seed={seeds[0]} (MulDiGraph)"
    )
    if len(all_probs) > 1:
        ensemble_probs = np.mean(all_probs, axis=0)
        compute_full_metrics(ensemble_probs, labels, "CRRN-ensemble (MulDiGraph)")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
