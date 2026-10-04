import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

import design

INPUT_IMAGE = "o-upper-med.jpg"
OUTPUT_IMAGE = "pipeline.png"

THRESHOLD = 128
THICKNESS = 24

steps = [
    ("Grayscale", transforms.Grayscale(num_output_channels=1)),
    (f"Binarized (> {THRESHOLD})", design.BinaryTransform(threshold=THRESHOLD)),
    (f"Dilated ({THICKNESS} passes)", design.DilateTransform(thickness=THICKNESS)),
    ("Final: cropped + centered", design.CropResizeTransform(final_size=28, inner_size=20)),
]

img = Image.open(INPUT_IMAGE)
stages = [("Input", img)]
for name, step in steps:
    img = step(img)
    stages.append((name, img))

fig, axes = plt.subplots(1, len(stages), figsize=(3 * len(stages), 3.5))
for ax, (name, im) in zip(axes, stages):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(f"{name}\n{im.size[0]}x{im.size[1]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=200)
print(f"Saved {OUTPUT_IMAGE}")