"""Evaluation metrics for phishing detection."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_metrics(probs, labels, threshold=0.5):
    """Fast metrics for training loop monitoring."""
    preds = (probs > threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    bacc = (tpr + tnr) / 2
    return {'p': p, 'r': r, 'f1': f1, 'bacc': bacc,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}


def compute_full_metrics(probs, labels, model_name="CRRN"):
    """Comprehensive metrics report (run once after training)."""
    preds = (probs > 0.5).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    tpr_val = tp / (tp + fn + 1e-8)
    fpr_val = fp / (fp + tn + 1e-8)
    tnr_val = tn / (tn + fp + 1e-8)
    bacc = (tpr_val + tnr_val) / 2

    auc_roc = float(roc_auc_score(labels, probs))
    auc_pr = float(average_precision_score(labels, probs))

    fpr_arr, tpr_arr, _ = roc_curve(labels, probs)
    recall_at = {}
    for target_fpr in [0.01, 0.05, 0.10]:
        valid = fpr_arr <= target_fpr
        recall_at[target_fpr] = float(tpr_arr[valid].max()) if valid.any() else 0.0

    print(f"\n{'=' * 60}")
    print(f"  {model_name} - Full Metrics Report")
    print(f"{'=' * 60}")
    print(f"\n  Threshold-based (t=0.5):")
    print(f"    Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}  BAcc: {bacc:.4f}")
    print(f"    TPR: {tpr_val:.4f}  FPR: {fpr_val:.4f}  TNR: {tnr_val:.4f}")
    print(f"    TP={int(tp)} FP={int(fp)} FN={int(fn)} TN={int(tn)}")
    print(f"\n  Threshold-independent:")
    print(f"    AUC-ROC:  {auc_roc:.4f}")
    print(f"    AUC-PR:   {auc_pr:.4f}")
    print(f"\n  Operating points:")
    print(f"    Recall@1%FPR:   {recall_at[0.01]:.4f}")
    print(f"    Recall@5%FPR:   {recall_at[0.05]:.4f}")
    print(f"    Recall@10%FPR:  {recall_at[0.10]:.4f}")
    return {'precision': float(p), 'recall': float(r), 'f1': float(f1), 'bacc': float(bacc),
            'auc_roc': auc_roc, 'auc_pr': auc_pr,
            'recall@1%fpr': recall_at[0.01], 'recall@5%fpr': recall_at[0.05],
            'recall@10%fpr': recall_at[0.10]}
