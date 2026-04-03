"""Shared training and evaluation loops."""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from model import BiSRN
from metrics import compute_metrics


def train_epoch(model, dataloader, optimizer, device, class_weight):
    """One training epoch with mini-batch gradient accumulation."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        bl, bc = 0, 0
        for sg in batch:
            logits, masks = model(sg.x, sg.edge_index, sg.edge_attr, sg.target_idx)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            loss = F.cross_entropy(logits, sg.y.unsqueeze(0), weight=class_weight)
            loss = loss + 0.1 * model.compute_sparse_loss(masks)
            (loss / len(batch)).backward()
            bl += loss.item()
            bc += (logits.argmax(-1).item() == sg.y.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += bl
        correct += bc
        total += len(batch)
    return {'loss': total_loss / total, 'accuracy': correct / total}


def evaluate(model, dataloader, device):
    """Collect prediction probabilities and labels for all test samples."""
    model.eval()
    all_probs, all_labels, all_in_ratios = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            for sg in batch:
                try:
                    logits, _ = model(sg.x, sg.edge_index, sg.edge_attr, sg.target_idx)
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    prob = F.softmax(logits, dim=-1)[0, 1].item()
                    all_probs.append(prob)
                    all_labels.append(sg.y.item())
                    all_in_ratios.append(sg.in_ratio)
                except Exception:
                    continue
    return np.array(all_probs), np.array(all_labels), np.array(all_in_ratios)


def train_one_model(seed, train_loader, test_loader, class_weight, node_dim,
                    args, device):
    """Train a single model with given seed. Returns best state dict and F1."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = BiSRN(
        node_dim=node_dim, edge_dim=4, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, target_keep_ratio=0.5, use_nrn=True
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_f1, best_state = 0, None
    patience, patience_count = 8, 0
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, device, class_weight)
        scheduler.step()
        if (epoch + 1) % 3 == 0 or epoch == 0:
            probs, labels, in_ratios = evaluate(model, test_loader, device)
            m = compute_metrics(probs, labels, in_ratios)
            f1 = m['f1']
            ratio_str = " ".join(f"[{k}]={v:.0%}" for k, v in sorted(m['by_ratio'].items()))
            print(f"  [seed={seed}] Ep{epoch+1}: F1={f1:.4f} P={m['p']:.4f} "
                  f"R={m['r']:.4f} BAcc={m['bacc']:.4f}  {ratio_str}")
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
                print(f"           * New best!")
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"           Early stop at epoch {epoch+1}")
                    break
    return best_state, best_f1, model
