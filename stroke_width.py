from torchvision.datasets import EMNIST
import numpy as np
import cv2

import design
from preprocess import preprocess_image

# =========================================
# Finding the average stroke width in EMNIST
# I am using 0's and measuring them
# =========================================

dataset = EMNIST(
    root=design.EMNISTDir,
    split="bymerge",
    train=True,
    download=True
)


# =========================================
# measure one image
# go to row 14 and scan from left to right
# =========================================

def measure_stroke_width(image):

    image_array = np.array(image)

    # Row 14 is the horizontal middle of a 28x28 image
    row = image_array[14]

    start = None

    for i, pixel in enumerate(row):

        # First non-black pixel means I reached the stroke
        if start is None and pixel >= 128:
            start = i

        # Once I am inside the stroke
        # the next black pixel means the stroke ended
        elif start is not None and pixel < 128:
            return i - start

    return None


# =========================================
# Measure at least 250 zeros to have a little more data
# =========================================

stroke_widths = []

for image, label in dataset:

    if label == 0:

        # torchvisions EMNIST images are stored transposed,
        # so I flip them upright before scanning row 14
        upright = np.array(image).T

        width = measure_stroke_width(upright)

        # I know some handwriting might not cross row 14 cleanly
        # so I need to only keep measurements that worked
        if width is not None:
            stroke_widths.append(width)

            if width > 10:
                print("Large width found:", width)

        if len(stroke_widths) == 250:
            break

# =========================================
# Measure my own images
# =========================================

thin_image = preprocess_image("custom_writting/O/O_Thin.png")
medium_image = preprocess_image("custom_writting/O/O_Medium.png")
thick_image = preprocess_image("custom_writting/O/O_Thick.png")

thin_width = measure_stroke_width(thin_image)
medium_width = measure_stroke_width(medium_image)
thick_width = measure_stroke_width(thick_image)

print("O_thin stroke width:", thin_width)
print("O_medium stroke width:", medium_width)
print("O_thick stroke width:", thick_width)



kernel = np.ones((2, 2), np.uint8)

# Dilating the thin O to make its stroke thicker
thin_dilated = cv2.dilate(
    np.array(thin_image),
    kernel,
    iterations=1
)

# Eroding the thick O to make its stroke thinner
thick_eroded = cv2.erode(
    np.array(thick_image),
    kernel,
    iterations=1
)

print("Dilated O_thin stroke width:", measure_stroke_width(thin_dilated))
print("Eroded O_thick stroke width:", measure_stroke_width(thick_eroded))

# =========================================
# Average stroke width
# =========================================

average_width = np.mean(stroke_widths)

print("Images measured:", len(stroke_widths))
print("Average stroke width:", average_width)
print("Smallest width:", min(stroke_widths))
print("Largest width:", max(stroke_widths))