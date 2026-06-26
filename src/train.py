"""
train.py  -  GPU-accelerated training for CIFAR-10 classifier.

Usage:
    python src/train.py                      # default 100 epochs
    python src/train.py --epochs 50          # custom epochs
    python src/train.py --batch-size 256     # larger batch
    python src/train.py --no-augment         # disable augmentation
"""
import os
import sys
import argparse
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import get_dataloaders
from src.model      import build_model
from src.utilities      import (
    set_seed, get_device,
    accuracy, top5_accuracy,
    save_checkpoint, save_results,
    AverageMeter,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CIFAR-10 classifier")
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch-size",      type=int,   default=128)
    p.add_argument("--lr",              type=float, default=0.1)
    p.add_argument("--weight-decay",    type=float, default=5e-4)
    p.add_argument("--momentum",        type=float, default=0.9)
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--val-split",       type=float, default=0.1)
    p.add_argument("--no-augment",      action="store_true")
    p.add_argument("--data-dir",        default="./data")
    p.add_argument("--ckpt-dir",        default="./checkpoints")
    p.add_argument("--results-dir",     default="./results")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--patience",        type=int,   default=15)
    return p.parse_args()


# ── GPU memory reporter ───────────────────────────────────────────────────────

def log_gpu_memory():
    """Print current GPU memory usage — helpful during training."""
    if torch.cuda.is_available():
        alloc    = torch.cuda.memory_allocated()  / 1e9
        reserved = torch.cuda.memory_reserved()   / 1e9
        print(f"[gpu] Memory allocated={alloc:.2f}GB | reserved={reserved:.2f}GB")


# ── Training pass ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    One full pass over the training set.

    Uses:
      - non_blocking GPU transfers for faster CPU→GPU data movement
      - Mixed precision (AMP) for 2-3x faster training on RTX GPUs
      - Gradient clipping for training stability

    Returns:
        (avg_loss, avg_top1_acc, avg_top5_acc)
    """
    model.train()
    loss_m  = AverageMeter()
    top1_m  = AverageMeter()
    top5_m  = AverageMeter()

    for images, labels in loader:
        # Async GPU transfer
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        # Mixed precision forward pass
        with autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        # Scaled backward + gradient clip
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        top1_m.update(accuracy(logits, labels), bs)
        top5_m.update(top5_accuracy(logits, labels), bs)

    return loss_m.avg, top1_m.avg, top5_m.avg


# ── Validation pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Evaluate the model on the validation set.
    Runs fully on GPU with no gradient tracking.

    Returns:
        (avg_loss, avg_top1_acc, avg_top5_acc)
    """
    model.eval()
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        top1_m.update(accuracy(logits, labels), bs)
        top5_m.update(top5_accuracy(logits, labels), bs)

    return loss_m.avg, top1_m.avg, top5_m.avg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    set_seed(args.seed)
    device = get_device()

    # Prefetch data on GPU-pinned memory for maximum throughput
    train_loader, val_loader, _ = get_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        val_split   = args.val_split,
        num_workers = 4,
        augment     = not args.no_augment,
        seed        = args.seed,
    )

    # ── Model → GPU ───────────────────────────────────────────────────────────
    model = build_model(num_classes=10, dropout=args.dropout).to(device)

    # ── Mixed precision scaler (RTX 5070 optimised) ───────────────────────────
    scaler = GradScaler(enabled=device.type == "cuda")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    ).to(device)

    # ── Optimiser: SGD + Nesterov momentum ────────────────────────────────────
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr           = args.lr,
        momentum     = args.momentum,
        weight_decay = args.weight_decay,
        nesterov     = True,
    )

    # ── LR schedule: Cosine Annealing ─────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── History dict (saved to JSON after training) ───────────────────────────
    history = {
        "train_loss":  [],
        "train_top1":  [],
        "train_top5":  [],
        "val_loss":    [],
        "val_top1":    [],
        "val_top5":    [],
        "lr":          [],
        "epoch_time":  [],
    }

    best_val_top1 = 0.0
    patience_ctr  = 0

    # ── Print training config ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CIFAR-10 Training — {args.epochs} epochs | device={device}")
    print(f"  Batch size : {args.batch_size} | LR={args.lr} | WD={args.weight_decay}")
    print(f"  Augment    : {not args.no_augment} | Label smoothing={args.label_smoothing}")
    if device.type == "cuda":
        print(f"  AMP (mixed precision) : ENABLED — RTX 5070 optimised")
        log_gpu_memory()
    print(f"{'='*70}\n")
    print(
        f"{'Epoch':>6} | {'Tr Loss':>8} {'Tr Top1':>8} {'Tr Top5':>8} |"
        f" {'Val Loss':>8} {'Val Top1':>8} {'Val Top5':>8} |"
        f" {'LR':>8} {'Time':>6}"
    )
    print("-" * 90)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_top1, tr_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        vl_loss, vl_top1, vl_top5 = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # ── Print row ─────────────────────────────────────────────────────────
        print(
            f"{epoch:>6} | {tr_loss:>8.4f} {tr_top1*100:>7.2f}%"
            f" {tr_top5*100:>7.2f}% |"
            f" {vl_loss:>8.4f} {vl_top1*100:>7.2f}%"
            f" {vl_top5*100:>7.2f}% |"
            f" {lr:>8.1e} {elapsed:>5.1f}s"
        )

        # ── Record history ────────────────────────────────────────────────────
        history["train_loss"].append(tr_loss)
        history["train_top1"].append(tr_top1)
        history["train_top5"].append(tr_top5)
        history["val_loss"].append(vl_loss)
        history["val_top1"].append(vl_top1)
        history["val_top5"].append(vl_top5)
        history["lr"].append(lr)
        history["epoch_time"].append(round(elapsed, 2))

        # ── Checkpoint ────────────────────────────────────────────────────────
        is_best = vl_top1 > best_val_top1
        if is_best:
            best_val_top1 = vl_top1
            patience_ctr  = 0
        else:
            patience_ctr += 1

        save_checkpoint(
            state={
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":         vl_top1,
                "val_top5":        vl_top5,
                "val_loss":        vl_loss,
                "args":            vars(args),
            },
            filepath=os.path.join(
                args.ckpt_dir, f"ckpt_epoch{epoch:03d}.pth"
            ),
            is_best=is_best,
        )

        # ── GPU memory log every 10 epochs ────────────────────────────────────
        if epoch % 10 == 0:
            log_gpu_memory()

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_ctr >= args.patience:
            print(f"\n[early stop] No val improvement for "
                  f"{args.patience} epochs. Stopping at epoch {epoch}.")
            break

    # ── Save training history ─────────────────────────────────────────────────
    history["best_val_top1"]  = best_val_top1
    history["best_val_top1_pct"] = round(best_val_top1 * 100, 3)
    history["epochs_trained"] = epoch
    history["total_time_sec"] = round(sum(history["epoch_time"]), 1)
    history["args"]           = vars(args)

    save_results(
        history,
        os.path.join(args.results_dir, "training_history.json")
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    total_min = sum(history["epoch_time"]) / 60
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Best val Top-1 : {best_val_top1*100:.2f}%")
    print(f"  Epochs trained : {epoch}")
    print(f"  Total time     : {total_min:.1f} minutes")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()