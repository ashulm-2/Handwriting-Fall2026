import numpy as np
from PIL import Image
from torchvision.datasets import EMNIST

ROOT = "./data" # local download location relative to curr directory
NUM_SAMPLES = 300 # at least 200
ROW = 14 # 14 pixels down

THRESHOLD = 128 # pixel >= this is white, otherwise its a black pixel

def get_O_images(n=NUM_SAMPLES, split="letters", labels=(15,)):
    dataset = EMNIST(ROOT, split=split, train=True, download=True, transform=None)
    images = []
    for img, label in dataset:
        if label in labels:
            images.append(img.transpose(Image.TRANSPOSE)) # images flipped along the diagonal so we need to transpose
            if len(images) >= n:
                break
    return images

def stroke_width_of(img, row=ROW):
    pixels = np.array(img) # 28x28 with values 0-255
    width = pixels.shape[1] # should be 28

    x = 0
    # scan left to right until we hit a white pixel
    while x < width and pixels[row, x] < THRESHOLD:
        x += 1

    # never found a white pixel on this row
    if x >= width:
        return None

    white_hit = x

    # keep going until we hit a black pixel again
    while x < width and pixels[row, x] >= THRESHOLD:
        x += 1

    # didn't find a black pixel
    if x >= width:
        return None

    black_hit = x
    
    return black_hit - white_hit

if __name__ == "__main__":
    images = get_O_images(NUM_SAMPLES)

    widths = []
    for img in images:
        w = stroke_width_of(img)
        if w is not None and w > 0:
            widths.append(w)

    if not widths:
        print("No images were measured")
    else:
        average = sum(widths) / len(widths)
        print(f"Average stroke width: {average:.3f} pixels")

    print(f"Images requested: {NUM_SAMPLES}")
    print(f"Images measured: {len(widths)}")