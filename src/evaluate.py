
import os
import sys
import json
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import get_dataloaders, CLASSES, denormalise
from src.model      import build_model
from src.utilities      import (
    get_device, load_checkpoint, save_results,
    compute_all_metrics, measure_throughput,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CIFAR-10 classifier")
    p.add_argument("--checkpoint",  default="./checkpoints/best_model.pth")
    p.add_argument("--data-dir",    default="./data")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--batch-size",  type=int, default=512)
    p.add_argument("--history",     default="./results/training_history.json")
    return p.parse_args()


# ── GPU inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """
    Full inference pass on GPU.
    Returns predictions, labels, probabilities, and raw images.
    """
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []
    all_images = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)

        probs = torch.softmax(logits, dim=1).cpu()
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels)
        all_probs.append(probs)
        all_images.append(images.cpu())

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy(),
        torch.cat(all_images),
    )


# ── Plot: Confusion matrix ────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path):
    from sklearn.metrics import confusion_matrix
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, data, fmt, title in zip(
        axes,
        [cm,     norm],
        ["d",    ".2f"],
        ["Counts", "Row-Normalised"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.4, ax=ax, annot_kws={"size": 9},
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "Confusion Matrix — CIFAR-10 Test Set", fontsize=15, y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Confusion matrix     → {save_path}")


# ── Plot: Training curves ─────────────────────────────────────────────────────

def plot_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"],
                 label="Train", color="#2196F3", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"],
                 label="Val",   color="#FF5722", linewidth=1.5)
    axes[0].set_title("Loss",           fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Top-1 Accuracy
    axes[1].plot(epochs, [a * 100 for a in history["train_top1"]],
                 label="Train Top-1", color="#2196F3", linewidth=1.5)
    axes[1].plot(epochs, [a * 100 for a in history["val_top1"]],
                 label="Val Top-1",   color="#FF5722", linewidth=1.5)
    axes[1].plot(epochs, [a * 100 for a in history["train_top5"]],
                 label="Train Top-5", color="#2196F3",
                 linewidth=1.5, linestyle="--", alpha=0.6)
    axes[1].plot(epochs, [a * 100 for a in history["val_top5"]],
                 label="Val Top-5",   color="#FF5722",
                 linewidth=1.5, linestyle="--", alpha=0.6)
    axes[1].set_title("Top-1 & Top-5 Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    # Learning rate
    axes[2].plot(epochs, history["lr"], color="#4CAF50", linewidth=1.5)
    axes[2].set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_yscale("log"); axes[2].grid(alpha=0.3)

    best_epoch = int(np.argmax(history["val_top1"])) + 1
    best_acc   = max(history["val_top1"]) * 100
    fig.suptitle(
        f"Training History — Best val Top-1: {best_acc:.2f}% at epoch {best_epoch}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[eval] Training curves      → {save_path}")


# ── Plot: Per-class accuracy ──────────────────────────────────────────────────

def plot_per_class_accuracy(metrics, save_path):
    classes = list(metrics["per_class_accuracy"].keys())
    accs    = [v * 100 for v in metrics["per_class_accuracy"].values()]
    colors  = [
        "#4CAF50" if a >= 90 else
        "#FF9800" if a >= 80 else
        "#F44336"
        for a in accs
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(classes, accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=3)
    ax.axhline(
        np.mean(accs), color="navy", linestyle="--", linewidth=1.5,
        label=f"Mean: {np.mean(accs):.1f}%"
    )
    ax.set_ylim(0, 108)
    ax.set_title("Per-Class Test Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()

    # Colour legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="≥ 90%"),
        Patch(facecolor="#FF9800", label="80–90%"),
        Patch(facecolor="#F44336", label="< 80%"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[eval] Per-class accuracy   → {save_path}")


# ── Plot: Top confident mistakes ──────────────────────────────────────────────

def plot_top_mistakes(images, y_true, y_pred, y_prob, save_path, n=20):
    wrong_mask = y_true != y_pred
    wrong_idx  = np.where(wrong_mask)[0]

    if len(wrong_idx) == 0:
        print("[eval] No mistakes found!")
        return

    confidences = y_prob[wrong_idx, y_pred[wrong_idx]]
    order       = np.argsort(-confidences)[:n]
    wrong_idx   = wrong_idx[order]

    cols = 5
    rows = (len(wrong_idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.2))
    axes = axes.flatten() if rows > 1 else axes

    for k, idx in enumerate(wrong_idx):
        img = denormalise(images[idx]).permute(1, 2, 0).numpy()
        axes[k].imshow(img)
        axes[k].set_title(
            f"True : {CLASSES[y_true[idx]]}\n"
            f"Pred : {CLASSES[y_pred[idx]]}\n"
            f"Conf : {confidences[order[k]]:.1%}",
            fontsize=7.5,
        )
        axes[k].axis("off")

    for k in range(len(wrong_idx), len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        f"Top-{len(wrong_idx)} Most Confident Mistakes", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Top mistakes         → {save_path}")


# ── Plot: Calibration curve ───────────────────────────────────────────────────

def plot_calibration_curve(y_true, y_pred, y_prob, save_path, n_bins=15):
    """
    Reliability diagram — shows whether confidence scores match accuracy.
    A perfectly calibrated model follows the diagonal.
    """
    confidences = y_prob.max(axis=1)
    correct     = (y_true == y_pred).astype(float)
    bin_edges   = np.linspace(0, 1, n_bins + 1)

    bin_accs  = []
    bin_confs = []
    bin_sizes = []

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_accs.append(correct[mask].mean())
        bin_confs.append(confidences[mask].mean())
        bin_sizes.append(mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reliability diagram
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    axes[0].bar(
        bin_confs, bin_accs, width=0.05, alpha=0.6,
        color="#2196F3", edgecolor="white", label="Model"
    )
    axes[0].set_xlabel("Mean Confidence")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Reliability Diagram", fontsize=13, fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

    # Confidence histogram
    axes[1].hist(confidences, bins=50, color="#FF5722", alpha=0.7,
                 edgecolor="white")
    axes[1].axvline(
        confidences.mean(), color="navy", linestyle="--", linewidth=1.5,
        label=f"Mean: {confidences.mean():.3f}"
    )
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence Distribution", fontsize=13, fontweight="bold")
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[eval] Calibration curve    → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[eval] Loading test set …")
    _, _, test_loader = get_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = 4,
        augment     = False,
    )

    # ── Model → GPU ───────────────────────────────────────────────────────────
    model = build_model().to(device)
    state = load_checkpoint(args.checkpoint, model, device=device)

    # ── Inference ─────────────────────────────────────────────────────────────
    print("[eval] Running inference on test set …")
    y_pred, y_true, y_prob, images = run_inference(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("[eval] Computing metrics …")
    metrics = compute_all_metrics(y_true, y_pred, y_prob, CLASSES)

    # ── Throughput ────────────────────────────────────────────────────────────
    print("[eval] Measuring GPU throughput …")
    throughput = measure_throughput(model, device, batch_size=512)
    metrics["throughput"] = throughput

    # ── Console report ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  CIFAR-10 Test Set Evaluation")
    print(f"{'='*65}")
    print(f"  Top-1  Accuracy       : {metrics['top1_accuracy_pct']:.2f}%")
    print(f"  Top-5  Accuracy       : {metrics['top5_accuracy_pct']:.2f}%")
    print(f"  Macro  F1             : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1           : {metrics['weighted_f1']:.4f}")
    print(f"  Matthews Corr Coef    : {metrics['matthews_corrcoef']:.4f}")
    print(f"  Cohen's Kappa         : {metrics['cohen_kappa']:.4f}")
    print(f"  ECE (calibration)     : {metrics['ece']:.4f}")
    print(f"  Mean Confidence       : {metrics['mean_confidence']:.4f}")
    print(f"  GPU Throughput        : "
          f"{metrics['throughput']['images_per_second']:,.0f} imgs/sec")
    print(f"{'='*65}")
    print(f"\n  Per-class accuracy:")
    for cls, acc in metrics["per_class_accuracy"].items():
        bar    = "█" * int(acc * 20)
        status = "✓" if acc >= 0.90 else "~" if acc >= 0.80 else "✗"
        print(f"  {status} {cls:<12} {acc*100:5.1f}%  {bar}")
    print(f"{'='*65}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("[eval] Generating plots …")

    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(args.results_dir, "confusion_matrix.png"),
    )
    plot_per_class_accuracy(
        metrics,
        os.path.join(args.results_dir, "per_class_accuracy.png"),
    )
    plot_top_mistakes(
        images, y_true, y_pred, y_prob,
        os.path.join(args.results_dir, "top_mistakes.png"),
    )
    plot_calibration_curve(
        y_true, y_pred, y_prob,
        os.path.join(args.results_dir, "calibration_curve.png"),
    )

    if os.path.exists(args.history):
        with open(args.history) as f:
            history = json.load(f)
        plot_training_curves(
            history,
            os.path.join(args.results_dir, "training_curves.png"),
        )

    # ── Save full report ──────────────────────────────────────────────────────
    eval_report = {
        **metrics,
        "checkpoint":       args.checkpoint,
        "checkpoint_epoch": state.get("epoch"),
        "best_val_acc":     state.get("val_acc"),
    }
    save_results(
        eval_report,
        os.path.join(args.results_dir, "evaluation_report.json"),
    )

    print(f"\n[eval] All results saved to '{args.results_dir}/'")
    print("[eval] Files generated:")
    for f in sorted(os.listdir(args.results_dir)):
        size = os.path.getsize(os.path.join(args.results_dir, f))
        print(f"         {f:<35} {size/1024:>6.1f} KB")


if __name__ == "__main__":
    main()