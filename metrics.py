"""Evaluation metrics for phishing detection."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_metrics(probs, labels, in_ratios, threshold=0.5):
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
    by_ratio = {}
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        mask = (in_ratios >= lo) & (in_ratios < hi) & (labels == 1)
        if mask.sum() > 0:
            by_ratio[f'{lo:.1f}-{hi:.1f}'] = ((preds == 1) & mask).sum() / mask.sum()
    return {'p': p, 'r': r, 'f1': f1, 'bacc': bacc,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'by_ratio': by_ratio}


def compute_full_metrics(probs, labels, in_ratios, model_name="BiSRN"):
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

    best_f1_opt, best_t_opt = 0, 0.5
    for t in np.arange(0.20, 0.80, 0.01):
        pd_t = (probs > t).astype(int)
        tp_t = ((pd_t == 1) & (labels == 1)).sum()
        fp_t = ((pd_t == 1) & (labels == 0)).sum()
        fn_t = ((pd_t == 0) & (labels == 1)).sum()
        p_t = tp_t / (tp_t + fp_t + 1e-8)
        r_t = tp_t / (tp_t + fn_t + 1e-8)
        f1_t = 2 * p_t * r_t / (p_t + r_t + 1e-8)
        if f1_t > best_f1_opt:
            best_f1_opt = f1_t
            best_t_opt = t

    grp = {}
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        mask_f = (in_ratios >= lo) & (in_ratios < hi) & (labels == 1)
        n_f = mask_f.sum()
        if n_f > 0:
            recall_g = float(((preds == 1) & mask_f).sum() / n_f)
            auc_g = -1.0
            if n_f >= 2:
                grp_mask = mask_f | (labels == 0)
                try:
                    auc_g = float(roc_auc_score(labels[grp_mask], probs[grp_mask]))
                except Exception:
                    pass
            grp[f'{lo:.1f}-{hi:.1f}'] = {'n': int(n_f), 'recall': recall_g, 'auc': auc_g}

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
    print(f"    Best F1:        {best_f1_opt:.4f} (t={best_t_opt:.2f})")
    print(f"\n  Per in-ratio group:")
    print(f"    {'Group':<12} {'N':>6} {'Recall':>8} {'AUC':>8}")
    print(f"    {'-' * 36}")
    for k, v in sorted(grp.items()):
        a_str = f"{v['auc']:.4f}" if v['auc'] >= 0 else "N/A"
        print(f"    {k:<12} {v['n']:>6} {v['recall']:>8.4f} {a_str:>8}")

    return {'precision': float(p), 'recall': float(r), 'f1': float(f1), 'bacc': float(bacc),
            'auc_roc': auc_roc, 'auc_pr': auc_pr,
            'recall@1%fpr': recall_at[0.01], 'recall@5%fpr': recall_at[0.05],
            'recall@10%fpr': recall_at[0.10],
            'best_f1': float(best_f1_opt), 'best_threshold': float(best_t_opt),
            'by_ratio': grp}
