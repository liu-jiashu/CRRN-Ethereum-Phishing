# CRRN: Counterparty Role-Relation Network

This repository provides the implementation of **CRRN (Counterparty Role-Relation Network)** for Ethereum phishing-account detection, as described in:

> **Modeling Transaction Counterparty Roles and Relations for Ethereum Phishing Detection**

CRRN models not only a target account's incoming and outgoing transactions, but also the latent fund-flow roles of its counterparties and the relations among those counterparties.

## Method

CRRN contains three main branches:

- **Incoming encoder**: models transactions funding the target account.
- **Outgoing encoder**: independently models transactions sent by the target account.
- **INRE**: models pairwise relations among the target account's counterparties using their inferred roles and observed connectivity.

The three representations are projected independently, concatenated, and used for phishing-account prediction.

## Repository

```text
CRRN/
├── model.py
├── data_utils.py
├── metrics.py
├── train_utils.py
├── train_b4e.py
├── train_muldigraph.py
├── DATASETS.md
├── ENVIRONMENT.md
├── METHOD_ALIGNMENT.md
├── REPRODUCE.md
├── requirements.txt
└── README.md
```

Datasets and checkpoints are not included in the repository.

## Installation

The verified environment uses Python 3.10 and PyTorch 2.1.2 with CUDA 11.8.

```bash
pip install -r requirements.txt
```

See `ENVIRONMENT.md` for the full environment and `DATASETS.md` for dataset preparation.

## Training

B4E:

```bash
python train_b4e.py --data_dir ./data/B4E --epochs 40 \
  --n_seeds 3 --seeds 42,123,456 --device cuda
```

MulDiGraph:

```bash
python train_muldigraph.py --data_dir ./data/MulDiGraph --epochs 40 \
  --n_seeds 3 --seeds 42,123,456 --device cuda
```

For the complete experiment commands, see `REPRODUCE.md`.

## Results

| Dataset | Mean F1 | Ensemble F1 | AUC-ROC | AUC-PR |
|---|---:|---:|---:|---:|
| B4E | 0.886 ± 0.001 | 0.893 | 0.981 | 0.957 |
| MulDiGraph | 0.963 ± 0.002 | 0.963 | 0.994 | 0.987 |

## Citation

```bibtex
@misc{liu2026crrn,
  title  = {Modeling Transaction Counterparty Roles and Relations for Ethereum Phishing Detection},
  author = {Jiashu Liu and Zhihui Zhang and Zhiqiang Li and Zheng Xing and Yanbin Wang},
  year   = {2026}
}
```
