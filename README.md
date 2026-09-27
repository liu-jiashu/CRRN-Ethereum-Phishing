# CRRN: Counterparty Role-Relation Network for Ethereum Phishing Detection

This repository contains the implementation of **CRRN (Counterparty Role-Relation Network)** for Ethereum phishing-account detection.

CRRN models not only the target account's incoming and outgoing transaction behavior, but also the **fund-flow roles of its counterparties** and the **relations among those counterparties**. The method is described in the manuscript:

> **Modeling Transaction Counterparty Roles and Relations for Ethereum Phishing Detection**

## Architecture

CRRN represents a target account through three complementary branches:

- **Role-Aware Incoming Transaction Encoder**: Aggregates incoming transactions using a direction-specific encoder. Each transaction is weighted using the counterparty's inferred latent fund-flow role together with transaction and node features.
- **Role-Aware Outgoing Transaction Encoder**: Uses an independently parameterized encoder for outgoing transactions, allowing the same counterparty role to have different effects on incoming and outgoing fund flows.
- **Inter-Neighbor Relation Encoder (INRE)**: Models pairwise relations among the target account's counterparties using their latent role distributions and observed inter-counterparty connectivity.

A shared role-inference module maps account-level transaction statistics to a soft distribution over latent fund-flow roles. The incoming, outgoing, and relation representations are then projected independently, concatenated, and passed to a shared classifier.

## Repository Structure

```text
CRRN/
├── model.py              # Model architecture
├── data_utils.py         # Shared data structures and preprocessing utilities
├── metrics.py            # F1, BAcc, AUC-ROC, AUC-PR, Recall@FPR
├── train_utils.py        # Training, evaluation, and multi-seed utilities
├── train_b4e.py          # B4E dataset loader and training script
├── train_muldigraph.py   # MulDiGraph dataset loader and training script
├── requirements.txt
└── README.md
```

## Datasets

### B4E

B4E is a large-scale Ethereum phishing dataset containing approximately 3,900 confirmed phishing accounts and 590K normal accounts, with roughly 12 million transaction records.

Expected input files:

```text
data_dir/
├── phisher_transaction_in.csv
├── phisher_transaction_out.csv
├── normal_eoa_transaction_in_slice_1000K.csv
└── normal_eoa_transaction_out_slice_1000K.csv
```

### MulDiGraph

MulDiGraph is a graph-structured Ethereum dataset from the XBlock platform containing approximately 2.97M accounts and 13.55M transactions.

Expected input file:

```text
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
|---|---:|---|
| `--epochs` | 40 | Maximum training epochs |
| `--lr` | 0.001 | Learning rate |
| `--batch_size` | 64 | Training batch size |
| `--hidden_dim` | 128 | Hidden representation dimension |
| `--num_layers` | 3 | Number of encoding layers |
| `--k_hop` | 2 | Neighborhood extraction depth |
| `--max_nodes` | 50 | Maximum nodes retained per local subgraph |
| `--n_seeds` | 3 | Number of random seeds |

## Results

The main experiments use three random seeds (`42, 123, 456`) and probability averaging for the ensemble.

| Dataset | Mean F1 | Ensemble F1 | AUC-ROC | AUC-PR |
|---|---:|---:|---:|---:|
| B4E | 0.886 ± 0.001 | 0.893 | 0.981 | 0.957 |
| MulDiGraph | 0.963 ± 0.002 | 0.963 | 0.994 | 0.987 |

The ensemble F1 scores exceed the strongest reported baseline by **7.9 percentage points on B4E** and **6.7 percentage points on MulDiGraph** under the evaluation protocol used in the manuscript.

## Environment

Tested with Python 3.10 and PyTorch 2.1.2 + CUDA 12.1 on NVIDIA GPUs.

```bash
pip install -r requirements.txt
```

## Citation

If you use this repository, please cite the accompanying manuscript:

```bibtex
@misc{liu2026crrn,
  title  = {Modeling Transaction Counterparty Roles and Relations for Ethereum Phishing Detection},
  author = {Jiashu Liu and Zhihui Zhang and Zhiqiang Li and Zheng Xing and Yanbin Wang},
  year   = {2026}
}
```
