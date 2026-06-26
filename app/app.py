import os
import sys
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict    import CIFAR10Predictor, INFER_TRANSFORM
from src.data_utils import CLASSES, get_dataloaders, denormalise
from src.utilities      import get_device


# ── Constants ─────────────────────────────────────────────────────────────────

DEVICE    = get_device()
PREDICTOR = None   # initialised in main() after arg parsing

CLASS_EMOJI = {
    "airplane":   "✈️",
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

CLASS_DESCRIPTIONS = {
    "airplane":   "A fixed-wing aircraft — commercial, military, or private.",
    "automobile": "A passenger car or small vehicle.",
    "bird":       "Any species of bird, in flight or perched.",
    "cat":        "A domestic cat.",
    "deer":       "A deer or similar hoofed animal.",
    "dog":        "A domestic dog of any breed.",
    "frog":       "A frog or toad.",
    "horse":      "A horse or pony.",
    "ship":       "A large boat, vessel, or ocean liner.",
    "truck":      "A large goods vehicle or heavy truck.",
}

# Colour for each class — used in the confidence bar chart
CLASS_COLORS = [
    "#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#8BC34A", "#795548", "#607D8B", "#F44336",
]


# ── Gradio callback ───────────────────────────────────────────────────────────

def classify_image(image: Image.Image):

    if image is None:
        return {}, "⬆️ Please upload an image to classify."

    # Run inference on GPU
    tensor = INFER_TRANSFORM(image.convert("RGB"))
    top10  = PREDICTOR.predict_topk(tensor, k=10)[0]   # all 10 classes

    top1_cls  = top10[0][0]
    top1_conf = top10[0][1]
    emoji     = CLASS_EMOJI.get(top1_cls, "")
    desc      = CLASS_DESCRIPTIONS.get(top1_cls, "")

    # Gradio Label widget expects {label: float}
    confidences = {cls: float(conf) for cls, conf in top10}

    # Markdown info panel
    conf_bar = "█" * int(top1_conf * 20)
    info = f"""
## {emoji} **{top1_cls.upper()}**
**Confidence:** {top1_conf * 100:.1f}%  `{conf_bar}`

_{desc}_

---
**Top-3 predictions:**
| Rank | Class | Confidence |
|------|-------|------------|
| 🥇 | {CLASS_EMOJI.get(top10[0][0],'')} {top10[0][0]} | {top10[0][1]*100:.1f}% |
| 🥈 | {CLASS_EMOJI.get(top10[1][0],'')} {top10[1][0]} | {top10[1][1]*100:.1f}% |
| 🥉 | {CLASS_EMOJI.get(top10[2][0],'')} {top10[2][0]} | {top10[2][1]*100:.1f}% |
"""
    return confidences, info


# ── Sample image loader ───────────────────────────────────────────────────────

def load_sample_images(data_dir="./data"):

    try:
        _, _, test_loader = get_dataloaders(
            data_dir    = data_dir,
            batch_size  = 64,
            augment     = False,
            num_workers = 2,
        )
        imgs, lbls = next(iter(test_loader))

        samples = {}
        for img_t, lbl in zip(imgs, lbls):
            cls = CLASSES[lbl.item()]
            if cls in samples:
                continue
            arr = (denormalise(img_t).permute(1, 2, 0).numpy() * 255
                   ).astype(np.uint8)
            pil = Image.fromarray(arr).resize((128, 128), Image.NEAREST)
            samples[cls] = pil
            if len(samples) == 10:
                break

        # Return as list ordered by CLASSES
        return [samples[c] for c in CLASSES if c in samples]

    except Exception as e:
        print(f"[app] Could not load sample images: {e}")
        return []


# ── Build Gradio UI ───────────────────────────────────────────────────────────

def build_ui(data_dir="./data") -> gr.Blocks:
    sample_images = load_sample_images(data_dir)

    with gr.Blocks(
        title  = "CIFAR-10 Classifier",
        theme  = gr.themes.Soft(primary_hue="blue"),
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🖼️ CIFAR-10 Image Classifier
            Upload any image and the model will predict which of the
            **10 CIFAR-10 categories** it belongs to.

            > **Model:** Custom ResNet-style CNN (11M params) |
            **Training:** 100 epochs | **Val accuracy: 95.06%** |
            **Device:** GPU (RTX 5070)
            """
        )

        # ── Main row ──────────────────────────────────────────────────────────
        with gr.Row():

            # Left — image upload
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type    = "pil",
                    label   = "Upload Image",
                    height  = 320,
                )
                classify_btn = gr.Button(
                    "🔍  Classify Image",
                    variant = "primary",
                    size    = "lg",
                )
                gr.Markdown(
                    "_Accepts any image format. The model resizes it to "
                    "32×32 internally (CIFAR-10 native resolution)._"
                )

            # Right — results
            with gr.Column(scale=1):
                label_output = gr.Label(
                    num_top_classes = 5,
                    label           = "Top-5 Class Probabilities",
                )
                info_output = gr.Markdown(
                    value = "Upload an image and click **Classify** to see results."
                )

        # ── Button & auto-classify on upload ──────────────────────────────────
        classify_btn.click(
            fn      = classify_image,
            inputs  = image_input,
            outputs = [label_output, info_output],
        )
        image_input.change(
            fn      = classify_image,
            inputs  = image_input,
            outputs = [label_output, info_output],
        )

        # ── Example images ────────────────────────────────────────────────────
        if sample_images:
            gr.Markdown("---\n### 📷 Click an example to classify it")
            gr.Examples(
                examples    = sample_images,
                inputs      = image_input,
                label       = "CIFAR-10 Test Set Samples (one per class)",
            )

        # ── Class reference ───────────────────────────────────────────────────
        gr.Markdown("---\n### 📋 CIFAR-10 Classes")
        with gr.Row():
            for cls in CLASSES:
                gr.Markdown(
                    f"**{CLASS_EMOJI.get(cls,'')}** {cls}"
                )

        # ── Footer ────────────────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            **Model:** Custom ResNet-style CNN &nbsp;|&nbsp;
            **Dataset:** CIFAR-10 (60,000 images, 10 classes, 32×32 px) &nbsp;|&nbsp;
            **Framework:** PyTorch &nbsp;|&nbsp;
            **Inference:** GPU accelerated (CUDA)
            """
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 Gradio web app")
    p.add_argument("--checkpoint", default="./checkpoints/best_model.pth")
    p.add_argument("--data-dir",   default="./data")
    p.add_argument("--port",       type=int, default=7860)
    p.add_argument("--share",      action="store_true",
                   help="Generate a public Gradio share link")
    return p.parse_args()


def main():
    global PREDICTOR
    args      = parse_args()
    PREDICTOR = CIFAR10Predictor(args.checkpoint)

    print(f"\n[app] Starting Gradio app on http://localhost:{args.port}")
    print(f"[app] Model loaded from: {args.checkpoint}")

    demo = build_ui(args.data_dir)
    demo.launch(
        server_port = args.port,
        share       = args.share,
        show_error  = True,
        inbrowser   = True,       
    )


if __name__ == "__main__":
    main()