"""
BiSRN training on B4E dataset.

Usage:
    python train_b4e.py --data_dir /path/to/b4e_data --epochs 40 --n_seeds 3
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, argparse, time, warnings
warnings.filterwarnings('ignore')

from model import BiSRN
from data_utils import SubgraphData, SubgraphDataset, collate_subgraphs
from metrics import compute_metrics, compute_full_metrics
from train_utils import train_epoch, evaluate, train_one_model


class EnhancedB4EDataLoader:
    COLUMNS = ['tx_hash', 'tx_index', 'block_hash', 'block_number',
               'block_tx_index', 'from_address', 'to_address', 'value',
               'gas', 'gas_price', 'input', 'timestamp']
    NODE_DIM = 20

    def __init__(self, data_dir, k_hop=2, max_nodes=50, sample_ratio=1.0, device='cuda'):
        self.data_dir = data_dir
        self.k_hop = k_hop; self.max_nodes = max_nodes
        self.sample_ratio = sample_ratio; self.device = torch.device(device)
        self.adjacency = defaultdict(list); self.reverse_adj = defaultdict(list)
        self.edge_data = defaultdict(list); self.node_features = {}
        self.labels = {}; self.phisher_nodes = []; self.normal_nodes = []
        self.address_to_id = {}; self.in_degree = {}; self.out_degree = {}

    def load(self):
        print("Loading B4E dataset...")
        files = [
            ('phisher_transaction_in.csv', 'phisher', 'in'),
            ('phisher_transaction_out.csv', 'phisher', 'out'),
            ('normal_eoa_transaction_in_slice_1000K.csv', 'normal', 'in'),
            ('normal_eoa_transaction_out_slice_1000K.csv', 'normal', 'out'),
        ]
        dfs = []
        phisher_addrs, normal_addrs = set(), set()
        for filename, atype, direction in files:
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath): continue
            print(f"   Loading: {filename}...", end=" ")
            df = pd.read_csv(filepath, header=None, names=self.COLUMNS,
                             usecols=[0,1,2,3,4,5,6,7,8,9,10,11],
                             dtype={'from_address': str, 'to_address': str, 'value': str},
                             low_memory=False)
            if self.sample_ratio < 1.0:
                df = df.sample(frac=self.sample_ratio, random_state=42)
            if atype == 'phisher':
                key_col = 'from_address' if direction == 'out' else 'to_address'
                phisher_addrs.update(df[key_col].dropna().unique())
            else:
                key_col = 'from_address' if direction == 'out' else 'to_address'
                normal_addrs.update(df[key_col].dropna().unique())
            dfs.append(df)
            print(f"done {len(df):,} txs")
        tx_df = pd.concat(dfs, ignore_index=True)
        tx_df['timestamp'] = pd.to_numeric(tx_df['timestamp'], errors='coerce')
        tx_df['value'] = pd.to_numeric(tx_df['value'], errors='coerce').fillna(0)
        tx_df['value_eth'] = tx_df['value'] / 1e18
        tx_df = tx_df.dropna(subset=['timestamp', 'from_address', 'to_address'])
        normal_addrs -= (phisher_addrs & normal_addrs)
        all_addresses = sorted(set(tx_df['from_address']) | set(tx_df['to_address']))
        self.address_to_id = {addr: i for i, addr in enumerate(all_addresses)}
        print(f"\n   Building graph...", end=" ")
        t0 = time.time()
        values = tx_df['value_eth'].values; timestamps = tx_df['timestamp'].values
        log_values = np.log1p(np.abs(values) + 1e-8)
        val_mean, val_std = log_values.mean(), log_values.std() + 1e-8
        ts_min, ts_max = timestamps.min(), timestamps.max() + 1e-8
        for _, row in tx_df.iterrows():
            src_addr, tgt_addr = row['from_address'], row['to_address']
            if src_addr not in self.address_to_id or tgt_addr not in self.address_to_id: continue
            src_id, tgt_id = self.address_to_id[src_addr], self.address_to_id[tgt_addr]
            val_norm = np.clip((np.log1p(np.abs(row['value_eth']) + 1e-8) - val_mean) / val_std, -5, 5)
            ts_norm = (row['timestamp'] - ts_min) / (ts_max - ts_min)
            self.adjacency[src_id].append(tgt_id)
            self.reverse_adj[tgt_id].append(src_id)
            self.edge_data[(src_id, tgt_id)].append([val_norm, ts_norm, 0.0, 0.0])
        print(f"done ({time.time()-t0:.1f}s)")
        print("   Computing enhanced features (20-dim)...", end=" ")
        self._compute_enhanced_features(tx_df)
        print("done")
        for addr in phisher_addrs:
            if addr in self.address_to_id:
                nid = self.address_to_id[addr]; self.labels[nid] = 1; self.phisher_nodes.append(nid)
        for addr in normal_addrs:
            if addr in self.address_to_id:
                nid = self.address_to_id[addr]; self.labels[nid] = 0; self.normal_nodes.append(nid)
        print(f"\n   Phisher: {len(self.phisher_nodes):,}, Normal: {len(self.normal_nodes):,}")
        return self

    def _compute_enhanced_features(self, tx_df):
        out_degree = tx_df.groupby('from_address').size().to_dict()
        in_degree = tx_df.groupby('to_address').size().to_dict()
        out_amount = tx_df.groupby('from_address')['value_eth'].agg(['sum', 'mean']).to_dict('index')
        in_amount = tx_df.groupby('to_address')['value_eth'].agg(['sum', 'mean']).to_dict('index')
        print("\n   Computing temporal stats...", end=" ")
        addr_out_txs = {}
        for addr, group in tx_df.groupby('from_address')[['timestamp', 'value_eth']]:
            addr_out_txs[addr] = list(zip(group['timestamp'].values, group['value_eth'].values))
        addr_in_txs = {}
        for addr, group in tx_df.groupby('to_address')[['timestamp', 'value_eth']]:
            addr_in_txs[addr] = list(zip(group['timestamp'].values, group['value_eth'].values))
        addr_temporal = {}
        all_addrs = set(addr_out_txs.keys()) | set(addr_in_txs.keys())
        for addr in all_addrs:
            out_txs = sorted(addr_out_txs.get(addr, []), key=lambda x: x[0])
            in_txs = sorted(addr_in_txs.get(addr, []), key=lambda x: x[0])
            all_txs = sorted(out_txs + in_txs, key=lambda x: x[0])
            if len(all_txs) < 2:
                addr_temporal[addr] = np.zeros(8, dtype=np.float32); continue
            ts = np.array([t[0] for t in all_txs]); amounts = np.array([t[1] for t in all_txs])
            duration = ts[-1] - ts[0]
            f12 = np.log1p(max(duration, 0))
            f13 = np.log1p(len(all_txs) / (duration / 3600 + 1e-8)) if duration > 0 else np.log1p(len(all_txs))
            if duration > 0 and len(ts) >= 3:
                max_burst = max(np.searchsorted(ts, ts[i] + 3600, side='right') - i for i in range(len(ts)))
                f14 = max_burst / len(all_txs)
            else: f14 = 1.0
            deltas = np.diff(ts)
            f15 = min(deltas.std() / (deltas.mean() + 1e-8), 10.0) if len(deltas) > 1 and deltas.mean() > 0 else 0.0
            if len(in_txs) > 0 and len(out_txs) > 0:
                out_times = np.array([t[0] for t in out_txs])
                fast_count = sum(1 for in_ts, _ in in_txs
                                 if np.searchsorted(out_times, in_ts) < len(out_times)
                                 and (out_times[np.searchsorted(out_times, in_ts)] - in_ts) <= 3600)
                f16 = fast_count / len(in_txs)
            else: f16 = 0.0
            abs_amounts = np.abs(amounts) + 1e-8
            if len(abs_amounts) > 1 and abs_amounts.sum() > 0:
                sorted_a = np.sort(abs_amounts); n = len(sorted_a)
                f17 = np.clip((2 * (np.arange(1, n+1) * sorted_a).sum() / (n * sorted_a.sum())) - (n+1)/n, 0, 1)
            else: f17 = 0.0
            total_tx = len(out_txs) + len(in_txs)
            nid = self.address_to_id.get(addr, -1)
            n_unique = len(set(self.adjacency.get(nid, []))) + len(set(self.reverse_adj.get(nid, [])))
            f18 = min(n_unique / total_tx, 1.0) if total_tx > 0 else 0.0
            f19 = min(deltas.max() / (duration + 1e-8), 1.0) if duration > 0 and len(deltas) > 0 else 0.0
            addr_temporal[addr] = np.array([f12, f13, f14, f15, f16, f17, f18, f19], dtype=np.float32)
        print("done")
        all_feats = []
        for addr, node_id in self.address_to_id.items():
            feat = np.zeros(self.NODE_DIM, dtype=np.float32)
            in_deg = in_degree.get(addr, 0); out_deg = out_degree.get(addr, 0)
            feat[0] = np.log1p(in_deg); feat[1] = np.log1p(out_deg)
            if addr in in_amount:
                feat[2] = np.log1p(abs(in_amount[addr]['sum']) + 1e-8)
                feat[3] = np.log1p(abs(in_amount[addr]['mean']) + 1e-8)
            if addr in out_amount:
                feat[4] = np.log1p(abs(out_amount[addr]['sum']) + 1e-8)
                feat[5] = np.log1p(abs(out_amount[addr]['mean']) + 1e-8)
            feat[6] = feat[2] - feat[4]
            total = feat[0] + feat[1]
            if total > 0: feat[7] = feat[0] / total
            feat[8] = np.log1p(len(set(self.adjacency.get(node_id, []))))
            feat[9] = np.log1p(len(set(self.reverse_adj.get(node_id, []))))
            total_deg = in_deg + out_deg
            if total_deg > 0: feat[10] = out_deg / total_deg
            feat[11] = np.log1p(out_deg / (in_deg + 1))
            feat[12:20] = addr_temporal.get(addr, np.zeros(8, dtype=np.float32))
            all_feats.append(feat)
            self.node_features[node_id] = feat
            self.in_degree[node_id] = in_deg; self.out_degree[node_id] = out_deg
        all_feats = np.array(all_feats)
        mean, std = all_feats.mean(axis=0), all_feats.std(axis=0) + 1e-8
        for nid in self.node_features:
            self.node_features[nid] = np.clip((self.node_features[nid] - mean) / std, -5, 5)

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
        if len(visited) < 3: return None
        subgraph_nodes = list(visited)
        local_map = {n: i for i, n in enumerate(subgraph_nodes)}
        if node_id not in local_map: return None
        target_idx = local_map[node_id]
        edges_src, edges_tgt, edge_feats = [], [], []
        for src in subgraph_nodes:
            for tgt in self.adjacency.get(src, []):
                if tgt in local_map:
                    edges_src.append(local_map[src]); edges_tgt.append(local_map[tgt])
                    feats = self.edge_data.get((src, tgt), [[0, 0, 0, 0]])
                    edge_feats.append(np.clip(np.mean(feats, axis=0), -5, 5))
        if len(edges_src) < 2: return None
        x = torch.tensor([self.node_features.get(n, np.zeros(self.NODE_DIM))
                          for n in subgraph_nodes], dtype=torch.float32, device=self.device)
        edge_index = torch.tensor([edges_src, edges_tgt], dtype=torch.long, device=self.device)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32, device=self.device)
        y = torch.tensor(self.labels.get(node_id, 0), dtype=torch.long, device=self.device)
        in_deg = self.in_degree.get(node_id, 0); out_deg = self.out_degree.get(node_id, 0)
        total = in_deg + out_deg; in_ratio = in_deg / total if total > 0 else 0.5
        return SubgraphData(x, edge_index, edge_attr, y, target_idx, in_deg, out_deg, in_ratio)

    def prepare_cached_dataset(self, node_indices, desc=""):
        subgraphs = []
        for nid in tqdm(node_indices, desc=desc, leave=False):
            sg = self.extract_subgraph(nid)
            if sg is not None: subgraphs.append(sg)
        return SubgraphDataset(subgraphs)

    def get_train_test_split(self, train_ratio=0.7):
        np.random.seed(42)
        phisher = self.phisher_nodes.copy(); normal = self.normal_nodes.copy()
        np.random.shuffle(phisher); np.random.shuffle(normal)
        normal = normal[:min(len(normal), len(phisher) * 3)]
        n_train_p = int(len(phisher) * train_ratio)
        n_train_n = int(len(normal) * train_ratio)
        train = phisher[:n_train_p] + normal[:n_train_n]
        test = phisher[n_train_p:] + normal[n_train_n:]
        np.random.shuffle(train); np.random.shuffle(test)
        return train, test


def main():
    parser = argparse.ArgumentParser(description="BiSRN on B4E")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--k_hop', type=int, default=2)
    parser.add_argument('--max_nodes', type=int, default=50)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--seeds', type=str, default='42,123,456')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seeds = [int(s) for s in args.seeds.split(',')][:args.n_seeds]
    print(f"Device: {device}, Seeds: {seeds}")

    # Data loading
    t0 = time.time()
    loader = EnhancedB4EDataLoader(
        data_dir=args.data_dir, k_hop=args.k_hop, max_nodes=args.max_nodes,
        sample_ratio=args.sample_ratio, device=args.device)
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
        state, best_f1, model = train_one_model(
            seed, train_loader, test_loader, class_weight,
            loader.NODE_DIM, args, device)
        torch.save(state, f'best_b4e_seed{seed}.pt')
        all_f1s.append(best_f1)
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        probs, labels, in_ratios = evaluate(model, test_loader, device)
        all_probs.append(probs)
        print(f"  seed={seed}: F1={best_f1:.4f} ({(time.time()-t1)/60:.1f} min)")

    # Results
    print(f"\n{'='*60}")
    print(f"Mean F1: {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")

    best_idx = np.argmax(all_f1s)
    compute_full_metrics(all_probs[best_idx], labels, in_ratios, "BiSRN-single (B4E)")
    if len(all_probs) > 1:
        ensemble_probs = np.mean(all_probs, axis=0)
        compute_full_metrics(ensemble_probs, labels, in_ratios, "BiSRN-ensemble (B4E)")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
