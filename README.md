# BiSRN: Bidirectional Selective Representation Network for Ethereum Phishing Detection

This repository contains the implementation of BiSRN, a three-channel directional encoding framework for detecting phishing accounts on the Ethereum blockchain.

## Architecture

BiSRN encodes the local transaction neighborhood of target accounts through three complementary channels:

- **Forward Encoder**: Captures outgoing fund flow patterns via edge-attention message passing along original edge directions.
- **Backward Encoder**: Captures incoming fund flow patterns via edge-attention on reversed edges.
- **Neighbor Relation Network (NRN)**: Models pairwise topological relationships among neighboring accounts.

Each encoder uses learned edge attention with sparse regularization to selectively focus on informative transaction edges.

## Repository Structure

```
BiSRN/
├── model.py              # BiSRN architecture (EdgeAttentionLayer, SRNEncoder, NRN, BiSRN)
├── data_utils.py         # Shared data structures (SubgraphData, SubgraphDataset)
├── metrics.py            # Evaluation metrics (F1, BAcc, AUC-ROC, AUC-PR, Recall@FPR)
├── train_utils.py        # Training loop, evaluation, multi-seed training
├── train_b4e.py          # B4E dataset loader + training script
├── train_muldigraph.py   # MulDiGraph dataset loader + training script
├── requirements.txt
└── README.md
```

## Datasets

### B4E
Transaction-level dataset with ~3,900 phishing accounts and ~590K normal accounts. Data files expected:
```
data_dir/
├── phisher_transaction_in.csv
├── phisher_transaction_out.csv
├── normal_eoa_transaction_in_slice_1000K.csv
└── normal_eoa_transaction_out_slice_1000K.csv
```

### MulDiGraph
Graph-structured dataset from XBlock platform with ~2.97M nodes and ~13.5M edges. Data file expected:
```
data_dir/
└── MulDiGraph.pkl
```

## Usage

### Training on B4E

```bash
python train_b4e.py --data_dir /path/to/b4e_data --epochs 40 --n_seeds 3 --seeds 42,123,456
```

### Training on MulDiGraph

```bash
python train_muldigraph.py --data_dir /path/to/data --epochs 40 --n_seeds 3 --seeds 42,123,456
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 40 | Maximum training epochs |
| `--lr` | 0.001 | Learning rate |
| `--batch_size` | 64 | Training batch size |
| `--hidden_dim` | 128 | Hidden representation dimension |
| `--num_layers` | 3 | Edge-attention layers per encoder |
| `--k_hop` | 2 | Neighborhood extraction depth |
| `--max_nodes` | 50 | Max nodes per subgraph |
| `--n_seeds` | 3 | Number of random seeds |

## Results

| Dataset | Mean F1 | Ensemble F1 | AUC-ROC | AUC-PR |
|---------|---------|-------------|---------|--------|
| B4E | 0.886 ± 0.001 | 0.893 | 0.981 | 0.957 |
| MulDiGraph | 0.963 ± 0.002 | 0.963 | 0.994 | 0.987 |

## Environment

Tested with Python 3.10, PyTorch 2.1.2 + CUDA 12.1, on NVIDIA GPUs.

```bash
pip install -r requirements.txt
```

## Citation

```
@article{bisrn2025,
  title={BiSRN: Directional Neighborhood Encoding with Selective Attention for Ethereum Phishing Detection},
  author={},
  year={2025}
}
```
