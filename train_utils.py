"""Shared training and evaluation loops."""

import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from model import CRRN


def train_epoch(model, dataloader, optimizer, device, class_weight, gate_lambda=0.1):
    """One training epoch with mini-batch gradient accumulation."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        bl, bc = 0, 0
        for sg in batch:
            logit, gates = model(sg.x, sg.edge_index, sg.edge_attr, sg.target_idx)
            target = sg.y.float().reshape(1)
            sample_weight = class_weight[sg.y].reshape(1)
            loss = F.binary_cross_entropy_with_logits(
                logit.reshape(1), target, weight=sample_weight
            )
            loss = loss + gate_lambda * model.compute_gate_loss(gates)
            (loss / len(batch)).backward()
            bl += loss.item()
            prediction = int(torch.sigmoid(logit).item() > 0.5)
            bc += prediction == sg.y.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += bl
        correct += bc
        total += len(batch)
    return {'loss': total_loss / total, 'accuracy': correct / total}


def evaluate(model, dataloader, device):
    """Collect prediction probabilities and labels for all test samples."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            for sg in batch:
                try:
                    logit, _ = model(sg.x, sg.edge_index, sg.edge_attr, sg.target_idx)
                    prob = torch.sigmoid(logit).item()
                    all_probs.append(prob)
                    all_labels.append(sg.y.item())
                except Exception as exc:
                    raise RuntimeError("CRRN evaluation failed for a retained subgraph") from exc
    return np.array(all_probs), np.array(all_labels)


def train_one_model(seed, train_loader, class_weight, node_dim, args, device):
    """Train one CRRN for the fixed epoch budget without consulting the test set."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = CRRN(
        node_dim=node_dim, edge_dim=4, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, target_mean_gate=args.target_mean_gate
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    history = []
    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weight,
            gate_lambda=args.gate_lambda,
        )
        history.append(train_metrics)
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(
                f"  [seed={seed}] Ep{epoch + 1}: "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f}"
            )
    final_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
    return final_state, model, history
