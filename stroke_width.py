import numpy as np
from torchvision.datasets import MNIST

MNIST_DIR = "./mnist_data"
NUM_IMAGES = 200
ROW = 14
WHITE_LEVEL = 128

# the thickness number my own pipeline uses (see emnist_style.py convert_image)
MY_THICKNESS = 3


def get_zero_images(num_needed):
    # this downloads MNIST the first time you run it, then reuses the saved copy
    dataset = MNIST(MNIST_DIR, train=True, download=True)

    zeros = []
    for img, label in dataset:
        if label == 0:
            zeros.append(np.array(img))
        if len(zeros) >= num_needed:
            break

    return zeros


def measure_width(img):
    row = img[ROW]
    col = 0

    # move right until we hit a white pixel
    while col < len(row) and row[col] < WHITE_LEVEL:
        col = col + 1

    if col >= len(row):
        return None

    start = col

    # keep moving right while still white
    while col < len(row) and row[col] >= WHITE_LEVEL:
        col = col + 1

    end = col
    width = end - start

    return width


def main():
    zeros = get_zero_images(NUM_IMAGES)
    print("zero images used:", len(zeros))

    widths = []
    for img in zeros:
        w = measure_width(img)
        if w is not None:
            widths.append(w)

    average_width = sum(widths) / len(widths)

    print("average MNIST stroke width:", average_width)
    print("thickness used in my pipeline:", MY_THICKNESS)


main()
