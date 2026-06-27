import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import get_dataloaders
from src.model      import build_model
from src.utilities      import get_device, load_checkpoint, save_results


class TemperatureScaler(nn.Module):
    """
    Wraps a trained model with a learnable temperature parameter T.
    Only T is optimised — all model weights are frozen.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(x)
        return logits / self.temperature

    def calibrate(self, val_loader, device: torch.device) -> float:
        """
        Find the optimal T by minimising NLL on the validation set.
        Returns the learned temperature value.
        """
        self.to(device)
        self.model.eval()

        # Collect all logits and labels on CPU first
        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                all_logits.append(self.model(imgs).cpu())
                all_labels.append(lbls)

        # Move to same device as temperature parameter
        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)

        # Optimise temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)
        nll = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            scaled = all_logits / self.temperature
            loss   = nll(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        T = self.temperature.item()
        print(f"[calibrate] Optimal temperature T = {T:.4f}")
        return T


def calibrate(self, val_loader, device: torch.device) -> float:
    """
    Find the optimal T by minimising NLL on the validation set.
    Returns the learned temperature value.
    """
    self.to(device)
    self.model.eval()

    # Collect all logits and labels on CPU first
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            all_logits.append(self.model(imgs).cpu())
            all_labels.append(lbls)

    # Move to same device as temperature parameter
    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device)

    # Optimise temperature
    optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def eval_step():
        optimizer.zero_grad()
        scaled = all_logits / self.temperature
        loss   = nll(scaled, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_step)
    T = self.temperature.item()
    print(f"[calibrate] Optimal temperature T = {T:.4f}")
    return T


def compute_ece(logits, labels, n_bins=15):
    """Expected Calibration Error."""
    probs      = F.softmax(logits, dim=1)
    confidences, preds = probs.max(dim=1)
    correct    = preds.eq(labels)
    bin_edges  = torch.linspace(0, 1, n_bins + 1)
    ece        = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].float().mean().item()
        bin_conf = confidences[mask].mean().item()
        ece     += (mask.sum().item() / len(labels)) * abs(bin_acc - bin_conf)
    return ece


def parse_args():
    p = argparse.ArgumentParser(description="Temperature scaling calibration")
    p.add_argument("--checkpoint",  default="./checkpoints/best_model.pth")
    p.add_argument("--data-dir",    default="./data")
    p.add_argument("--results-dir", default="./results")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()

    _, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir, batch_size=256, augment=False
    )

    model = build_model().to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    scaler = TemperatureScaler(model)
    T      = scaler.calibrate(val_loader, device)

    # Compare ECE before and after
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            all_logits.append(model(imgs.to(device)).cpu())
            all_labels.append(lbls)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    ece_before = compute_ece(all_logits, all_labels)
    ece_after  = compute_ece(all_logits / T, all_labels)

    print(f"\n{'='*50}")
    print(f"  Temperature Scaling Results")
    print(f"{'='*50}")
    print(f"  Optimal T     : {T:.4f}")
    print(f"  ECE before    : {ece_before:.4f}")
    print(f"  ECE after     : {ece_after:.4f}")
    print(f"  ECE reduction : {(ece_before-ece_after)/ece_before*100:.1f}%")
    print(f"{'='*50}")

    save_results(
        {"temperature": T, "ece_before": ece_before, "ece_after": ece_after},
        os.path.join(args.results_dir, "calibration_results.json")
    )
    torch.save({"temperature": T}, "./checkpoints/temperature.pt")
    print(f"\n[calibrate] Temperature saved → checkpoints/temperature.pt")
    print("[calibrate] Apply at inference: logits = model(x) / T")


if __name__ == "__main__":
    main()