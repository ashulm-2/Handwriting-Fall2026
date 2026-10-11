import numpy as np
from PIL import Image
from torchvision.datasets import EMNIST
import torchvision.transforms.functional as F

EMNIST_DIR = "./emnist_data"
EMNIST_SPLIT = "byclass"
LETTER_LABEL = 24

NUM_IMAGES = 200
ROW = 14
WHITE_LEVEL = 128

# the thickness number my pipeline uses
MY_THICKNESS = 3


def fix_orientation(img):
    # EMNIST images come out rotated and mirrored from how they actually look.
    # fix for that.
    img = F.rotate(img, -90)
    img = F.hflip(img)
    return img


def get_letter_images(num_needed):
    dataset = EMNIST(EMNIST_DIR, split=EMNIST_SPLIT, train=True, download=True)

    letters = []
    for img, label in dataset:
        if label == LETTER_LABEL:
            img = fix_orientation(img)
            letters.append(np.array(img))
        if len(letters) >= num_needed:
            break

    return letters


def measure_width(img):
    row = img[ROW]
    col = 0

    # move right until hit a white pixel
    while col < len(row) and row[col] < WHITE_LEVEL:
        col = col + 1

    if col >= len(row):
        return None

    start = col

    # moving right while white
    while col < len(row) and row[col] >= WHITE_LEVEL:
        col = col + 1

    end = col
    width = end - start

    return width


def main():
    letters = get_letter_images(NUM_IMAGES)
    print("O images used:", len(letters))

    widths = []
    for img in letters:
        w = measure_width(img)
        if w is not None:
            widths.append(w)

    average_width = sum(widths) / len(widths)

    print("average EMNIST O stroke width:", average_width)
    print("thickness used in my pipeline:", MY_THICKNESS)


main()