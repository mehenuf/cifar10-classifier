import os
import json
import random
import time
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    accuracy_score,
    precision_recall_fscore_support,
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed=42):
    """Pin all RNGs for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    """Return best available device with GPU info."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"[device] Using: {dev}")
    if dev.type == "cuda":
        print(f"[device] GPU : {torch.cuda.get_device_name(0)}")
        print(f"[device] VRAM: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return dev


# ── Batch-level helpers ───────────────────────────────────────────────────────

def accuracy(logits, targets):
    """Top-1 accuracy for a single batch."""
    return (logits.argmax(1) == targets).float().mean().item()


def top5_accuracy(logits, targets):
    """
    Top-5 accuracy — correct if true label is in the 5 highest logits.
    Standard metric for ImageNet; useful context for CIFAR-10 (10 classes).
    """
    k = min(5, logits.size(1))
    top_k = logits.topk(k, dim=1).indices          # (B, k)
    correct = top_k.eq(targets.view(-1, 1).expand_as(top_k))
    return correct.any(dim=1).float().mean().item()


#  Full evaluation metrics
def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: tuple,
) -> dict:
    """
    Compute the full suite of evaluation metrics for CIFAR-10.

    Args:
        y_true:       Ground-truth class indices  (N,)
        y_pred:       Predicted class indices     (N,)
        y_prob:       Softmax probabilities       (N, C)
        class_names:  Tuple of class name strings

    Returns:
        Dict containing all metrics — ready to save as JSON.
    """
    results = {}

    # 1. Overall accuracy
    results["top1_accuracy"]     = float(accuracy_score(y_true, y_pred))
    results["top1_accuracy_pct"] = round(results["top1_accuracy"] * 100, 3)

    # Top-5 (using probabilities)
    top5_correct = 0
    for i, probs in enumerate(y_prob):
        top5_idx = np.argsort(probs)[::-1][:5]
        if y_true[i] in top5_idx:
            top5_correct += 1
    results["top5_accuracy"]     = float(top5_correct / len(y_true))
    results["top5_accuracy_pct"] = round(results["top5_accuracy"] * 100, 3)

    # 2. Per-class Precision / Recall / F1 
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    results["per_class"] = {
        cls: {
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]),  4),
            "f1_score":  round(float(f1[i]),   4),
            "support":   int(support[i]),
        }
        for i, cls in enumerate(class_names)
    }

    #  3. Macro & Weighted averages 
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        results[f"{avg}_precision"] = round(float(p), 4)
        results[f"{avg}_recall"]    = round(float(r), 4)
        results[f"{avg}_f1"]        = round(float(f), 4)

    # 4. Matthews Correlation Coefficient 
    # MCC is robust to class imbalance — ranges from -1 (worst) to +1 (best)
    results["matthews_corrcoef"] = round(
        float(matthews_corrcoef(y_true, y_pred)), 4
    )

    #  5. Cohen's Kappa 
    # Measures agreement beyond chance — good for balanced datasets like CIFAR-10
    results["cohen_kappa"] = round(
        float(cohen_kappa_score(y_true, y_pred)), 4
    )

    # 6. Per-class accuracy 
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    results["per_class_accuracy"] = {
        cls: round(float(per_class_acc[i]), 4)
        for i, cls in enumerate(class_names)
    }
    results["min_class_accuracy"] = round(float(per_class_acc.min()), 4)
    results["max_class_accuracy"] = round(float(per_class_acc.max()), 4)

    #  7. Mean confidence & mean correct confidence 
    # Tells us how confident the model is and whether confidence aligns with accuracy
    max_probs = y_prob.max(axis=1)
    results["mean_confidence"]         = round(float(max_probs.mean()), 4)
    correct_mask                       = y_true == y_pred
    results["mean_correct_confidence"] = round(
        float(max_probs[correct_mask].mean()), 4
    )
    results["mean_wrong_confidence"]   = round(
        float(max_probs[~correct_mask].mean()), 4
    ) if (~correct_mask).sum() > 0 else 0.0

    #  8. Expected Calibration Error (ECE) 
    # Measures how well confidence scores match actual accuracy
    # A well-calibrated model has ECE close to 0
    results["ece"] = round(
        float(_expected_calibration_error(y_true, y_pred, y_prob)), 4
    )

    #  9. Classification report (full) 
    results["classification_report"] = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return results


def _expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).
    Bins predictions by confidence and measures accuracy vs confidence gap.
    Lower is better; 0.0 = perfectly calibrated.
    """
    confidences = y_prob.max(axis=1)
    correct     = (y_true == y_pred).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece     += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)

    return ece


# Inference speed 

def measure_throughput(model, device, batch_size=256, n_batches=20) -> dict:
    """
    Measure inference speed on GPU.

    Returns:
        Dict with images_per_second and ms_per_image.
    """
    model.eval()
    dummy = torch.randn(batch_size, 3, 32, 32).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_batches):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed        = time.time() - t0
    total_images   = batch_size * n_batches
    imgs_per_sec   = total_images / elapsed
    ms_per_image   = (elapsed / total_images) * 1000

    print(f"[throughput] {imgs_per_sec:,.0f} images/sec "
          f"| {ms_per_image:.3f} ms/image")

    return {
        "images_per_second": round(imgs_per_sec, 1),
        "ms_per_image":      round(ms_per_image, 4),
        "batch_size":        batch_size,
    }


# Checkpoint management 

def save_checkpoint(state, filepath, is_best=False):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best:
        best = os.path.join(os.path.dirname(filepath), "best_model.pth")
        torch.save(state, best)
        print(f"[ckpt] New best saved → {best}")


def load_checkpoint(filepath, model, optimizer=None, device=None):
    device = device or get_device()
    state  = torch.load(filepath, map_location=device)
    model.load_state_dict(state["model_state"])
    if optimizer and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    print(f"[ckpt] Loaded epoch={state.get('epoch')} "
          f"val_acc={state.get('val_acc'):.4f}")
    return state


def save_results(results, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[results] Saved → {filepath}")


#  AverageMeter 

class AverageMeter:
    """Running average tracker for loss and accuracy per epoch."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count