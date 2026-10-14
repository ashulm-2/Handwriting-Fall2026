import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import design

# CNNEMNIST.pt was trained on EMNIST bymerge (47 classes)
# so design has to describe that, even if DATASET_TYPE says CUSTOM right now
design.DATASET_TYPE = "EMNIST"
design.EMNIST_SPLIT = "bymerge"

from CNN import CNN

CLEAN_DIR = "custom_writting_clean"

model = CNN()
model.load_state_dict(torch.load("CNNEMNIST.pt", map_location=model.device))
model.eval()

for class_name in sorted(os.listdir(CLEAN_DIR)):

    class_dir = os.path.join(CLEAN_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    for filename in sorted(os.listdir(class_dir)):

        if not filename.lower().endswith(".png"):
            continue

        # Already preprocessed: upright, 28x28, white on black.
        # No transpose here, the training transpose only turned EMNIST upright
        image = Image.open(os.path.join(class_dir, filename)).convert("L")
        x = TF.to_tensor(image).unsqueeze(0).to(model.device)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]

        top = torch.topk(probs, 3)
        guesses = ", ".join(
            f"{design.CToC(i.item())} {p.item():.0%}"
            for p, i in zip(top.values, top.indices)
        )

        print(f"{filename}: should be {class_name} -> {guesses}")