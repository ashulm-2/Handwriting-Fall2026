import os
import numpy as np
from PIL import Image

from preprocess import preprocess_image

RAW_DIR = "custom_writting"
CLEAN_DIR = "custom_writting_clean"

IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp")

saved = 0
skipped = []

for class_name in sorted(os.listdir(RAW_DIR)):

    class_dir = os.path.join(RAW_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    out_dir = os.path.join(CLEAN_DIR, class_name)
    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(class_dir):

        if not filename.lower().endswith(IMAGE_TYPES):
            continue

        path = os.path.join(class_dir, filename)

        try:
            result = preprocess_image(path)
        except Exception as e:
            skipped.append((path, str(e)))
            continue

        # A blank result means the character was lost (bad photo, too faint, etc.)
        if not np.array(result).any():
            skipped.append((path, "blank after preprocessing"))
            continue

        name = os.path.splitext(filename)[0] + ".png"
        result.save(os.path.join(out_dir, name))
        saved += 1

print("Saved:", saved)
print("Skipped:", len(skipped))
for path, reason in skipped:
    print("  ", path, "-", reason)