import sys
import os
sys.path.insert(0, '.')

from src.data_utils import get_dataloaders, denormalise, CLASSES
from PIL import Image
import numpy as np

# Create a folder for test images
os.makedirs("test_images", exist_ok=True)

# Load test set
_, _, test_loader = get_dataloaders(
    data_dir    = "./data",
    batch_size  = 64,
    augment     = False,
)

# Get one batch
imgs, lbls = next(iter(test_loader))

# Save one image per class
saved   = {}
saved_files = []

for i in range(len(imgs)):
    label = CLASSES[lbls[i].item()]

    # Save only one image per class
    if label in saved:
        continue

    # Convert tensor → PIL image → resize for visibility
    img_np = (denormalise(imgs[i]).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil    = Image.fromarray(img_np).resize((128, 128), Image.NEAREST)

    fname  = f"test_images/{label}.png"
    pil.save(fname)
    saved[label] = True
    saved_files.append((fname, label))
    print(f"  Saved: {fname}")

    # Stop once we have all 10 classes
    if len(saved) == 10:
        break

print(f"\nDone! {len(saved_files)} images saved to 'test_images/' folder")
print("\nRun predictions with:")
for fname, label in saved_files:
    print(f"  python src/predict.py --image {fname} --top-k 5 --visualise")