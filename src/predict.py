
import os
import sys
import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import CLASSES, CIFAR10_MEAN, CIFAR10_STD
from src.model      import build_model
from src.utilities      import get_device, load_checkpoint


# ── Inference transform ───────────────────────────────────────────────────────

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Emoji labels for nicer output
CLASS_EMOJI = {
    "airplane":   "✈️ ",
    "automobile": "🚗",
    "bird":       "🐦",
    "cat":        "🐱",
    "deer":       "🦌",
    "dog":        "🐶",
    "frog":       "🐸",
    "horse":      "🐴",
    "ship":       "🚢",
    "truck":      "🚚",
}


# ── Predictor class ───────────────────────────────────────────────────────────

class CIFAR10Predictor:
    """
    GPU-accelerated predictor wrapper around the trained CIFAR10Net.

    Args:
        checkpoint: Path to best_model.pth
        device:     Torch device — auto-detected if None
    """

    def __init__(self, checkpoint="./checkpoints/best_model.pth", device=None):
        self.device = device or get_device()
        self.model  = build_model().to(self.device)
        load_checkpoint(checkpoint, self.model, device=self.device)
        self.model.eval()
        print(f"[predict] Model ready on {self.device}")

    # ── Core inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_topk(self, tensor: torch.Tensor, k: int = 5):
        """
        Returns top-k predictions for one or more images.

        Args:
            tensor: Shape (C,H,W) or (B,C,H,W) — normalised image tensor
            k:      Number of top predictions to return

        Returns:
            List of lists → each inner list has k (class_name, confidence) tuples
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = self.model(tensor)

        probs   = F.softmax(logits, dim=1).cpu()
        results = []

        for p in probs:
            vals, idxs = torch.topk(p, k=min(k, len(CLASSES)))
            results.append([
                (CLASSES[i.item()], round(v.item(), 6))
                for v, i in zip(vals, idxs)
            ])
        return results

    @torch.no_grad()
    def predict_image(self, image_path: str):
        """
        Classify a single image file.

        Args:
            image_path: Path to any image file (.jpg/.png/.bmp etc.)

        Returns:
            (top1_class, top1_confidence, top5_list, inference_ms)
        """
        img    = Image.open(image_path).convert("RGB")
        tensor = INFER_TRANSFORM(img)

        t0   = time.perf_counter()
        top5 = self.predict_topk(tensor, k=5)[0]
        ms   = (time.perf_counter() - t0) * 1000

        return top5[0][0], top5[0][1], top5, round(ms, 2)

    @torch.no_grad()
    def predict_batch(self, image_paths: list):
        """
        Classify a batch of images in one GPU forward pass — much faster
        than calling predict_image() in a loop.

        Args:
            image_paths: List of image file paths

        Returns:
            List of (class_name, confidence) tuples — one per image
        """
        tensors = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            tensors.append(INFER_TRANSFORM(img))

        batch  = torch.stack(tensors)           # (B, C, H, W)
        t0     = time.perf_counter()
        top1s  = self.predict_topk(batch, k=1)
        ms     = (time.perf_counter() - t0) * 1000

        results = [(r[0][0], r[0][1]) for r in top1s]
        print(f"[predict] Batch of {len(image_paths)} images → "
              f"{ms:.1f}ms total ({ms/len(image_paths):.2f}ms/image)")
        return results

    def predict_folder(self, folder: str):
        """
        Classify all images in a folder using batched GPU inference.

        Args:
            folder: Directory containing image files

        Returns:
            List of dicts: {file, class, confidence, emoji}
        """
        exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        paths  = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in exts
        ]

        if not paths:
            print(f"[predict] No images found in '{folder}'")
            return []

        preds   = self.predict_batch(paths)
        results = []
        for path, (cls, conf) in zip(paths, preds):
            results.append({
                "file":       os.path.basename(path),
                "class":      cls,
                "confidence": conf,
                "emoji":      CLASS_EMOJI.get(cls, ""),
            })
        return results


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise_prediction(image_path, top5, inference_ms, save_path=None):
    """
    Side-by-side visualisation: original image + top-5 confidence bar chart.
    Saves to PNG if save_path is provided.
    """
    img    = Image.open(image_path).convert("RGB").resize((192, 192))
    names  = [cls  for cls, _    in top5]
    confs  = [conf * 100 for _, conf in top5]
    colors = ["#4CAF50" if i == 0 else "#2196F3" for i in range(len(top5))]

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Image panel
    ax_img.imshow(np.array(img))
    emoji = CLASS_EMOJI.get(names[0], "")
    ax_img.set_title(
        f"Prediction: {emoji} {names[0].upper()}\n"
        f"Confidence: {confs[0]:.1f}%  |  "
        f"Inference: {inference_ms:.1f}ms",
        fontsize=11, fontweight="bold",
    )
    ax_img.axis("off")

    # Bar chart panel
    bars = ax_bar.barh(
        [f"{CLASS_EMOJI.get(n,'')} {n}" for n in names[::-1]],
        confs[::-1],
        color=colors[::-1],
        edgecolor="white",
    )
    ax_bar.bar_label(bars, fmt="%.2f%%", fontsize=9, padding=4)
    ax_bar.set_xlim(0, 115)
    ax_bar.set_xlabel("Confidence (%)")
    ax_bar.set_title("Top-5 Class Probabilities", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    out = save_path or "prediction_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[predict] Visualisation saved → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 GPU inference")
    p.add_argument("--checkpoint", default="./checkpoints/best_model.pth")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to a single image")
    group.add_argument("--folder", help="Path to a folder of images")

    p.add_argument("--top-k",     type=int,  default=5)
    p.add_argument("--visualise", action="store_true",
                   help="Save a prediction visualisation PNG")
    p.add_argument("--out-dir",   default="./results")
    return p.parse_args()


def main():
    args      = parse_args()
    predictor = CIFAR10Predictor(args.checkpoint)

    if args.image:
        # ── Single image ──────────────────────────────────────────────────────
        if not os.path.exists(args.image):
            print(f"[predict] Error: file not found → {args.image}")
            return

        cls, conf, top5, ms = predictor.predict_image(args.image)

        print(f"\n{'='*50}")
        print(f"  File      : {os.path.basename(args.image)}")
        print(f"  Top-1     : {CLASS_EMOJI.get(cls,'')} {cls.upper()}")
        print(f"  Confidence: {conf*100:.2f}%")
        print(f"  Inference : {ms:.1f}ms")
        print(f"{'='*50}")
        print(f"\n  Top-{args.top_k} predictions:")
        for rank, (c, p) in enumerate(top5[:args.top_k], 1):
            bar = "█" * int(p * 30)
            print(f"  {rank}. {CLASS_EMOJI.get(c,'')} {c:<12} "
                  f"{p*100:6.2f}%  {bar}")
        print()

        if args.visualise:
            os.makedirs(args.out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.image))[0]
            visualise_prediction(
                args.image, top5, ms,
                save_path=os.path.join(
                    args.out_dir, f"{base}_prediction.png"
                ),
            )

    elif args.folder:
        # ── Folder ────────────────────────────────────────────────────────────
        results = predictor.predict_folder(args.folder)

        if not results:
            return

        print(f"\n{'='*60}")
        print(f"  Results for {len(results)} images in '{args.folder}'")
        print(f"{'='*60}")
        print(f"  {'File':<28} {'Class':<14} Confidence")
        print(f"  {'-'*55}")
        for r in results:
            print(f"  {r['file']:<28} "
                  f"{r['emoji']} {r['class']:<12} "
                  f"{r['confidence']*100:.1f}%")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()