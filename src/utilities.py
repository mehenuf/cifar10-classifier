import os
import json
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Pin all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device():
    """Return GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"[device] Using: {dev}")
    if dev.type == "cuda":
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[device] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return dev


def accuracy(logits, targets):
    """Top-1 accuracy for a batch."""
    return (logits.argmax(1) == targets).float().mean().item()


def save_checkpoint(state, filepath, is_best=False):
    """Save model checkpoint; optionally copy as best_model.pth."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best:
        best = os.path.join(os.path.dirname(filepath), "best_model.pth")
        torch.save(state, best)
        print(f"[ckpt] New best saved → {best}")


def load_checkpoint(filepath, model, optimizer=None, device=None):
    """Load weights (and optionally optimizer state) from a checkpoint."""
    device = device or get_device()
    state  = torch.load(filepath, map_location=device)
    model.load_state_dict(state["model_state"])
    if optimizer and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    print(f"[ckpt] Loaded epoch={state.get('epoch')} "
          f"val_acc={state.get('val_acc'):.4f}")
    return state


def save_results(results, filepath):
    """Persist a dict to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[results] Saved → {filepath}")


class AverageMeter:
    """Tracks running average — used for loss and accuracy per epoch."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count