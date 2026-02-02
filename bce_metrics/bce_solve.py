import torch
import torch.nn.functional as F

def eval_bce_metrics(all_logits, all_x, threshold = 0.5):
    probs= torch.sigmoid(all_logits)
    preds = (probs >=0.5).float()
    targets = all_x.float()
    tp = ((preds ==1)& (targets == 1)).sum().item()
    fp = ((preds ==1)& (targets == 0)).sum().item()
    tn = ((preds ==0)& (targets == 0)).sum().item()
    fn = ((preds ==0)& (targets == 1)).sum().item()

    eps = 1e-8 # 일종의 지지장치
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    acc       = (tp + tn) / (tp + tn + fp + fn + eps)

    bce = F.binary_cross_entropy_with_logits(all_logits, targets, reduction='mean').item()

    return {
        "threshold": threshold,
        "bce": bce,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }
