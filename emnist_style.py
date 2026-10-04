import sys
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image


def load_image(path):
    # just read the image in black and white mode
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def make_binary(img):
    # turn the image into pure black and white pixels
    thresh_value, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = binary_img > 0
    return mask


def crop_to_char(mask):
    # find where the white pixels are
    rows, cols = np.where(mask)
    top = rows.min()
    bottom = rows.max() + 1
    left = cols.min()
    right = cols.max() + 1

    # add a little space around the letter so it's not cut off
    height = bottom - top
    width = right - left
    pad = int(max(height, width) * 0.05)

    top = max(0, top - pad)
    bottom = min(mask.shape[0], bottom + pad)
    left = max(0, left - pad)
    right = min(mask.shape[1], right + pad)

    return mask[top:bottom, left:right]


def get_skeleton(mask):
    # shrink the letter down to a thin line in the middle of the stroke
    img = mask.astype(np.uint8)
    skeleton = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        piece = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, piece)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return skeleton > 0


def fix_thickness(mask, thickness):
    # take the thin line and make it thicker again but even this time
    skeleton = get_skeleton(mask)
    radius = int(thickness / 2)
    if radius < 1:
        radius = 1

    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    thick_mask = cv2.dilate(skeleton.astype(np.uint8), kernel)

    return thick_mask > 0


def resize_char(mask, box_size):
    # shrink the letter down so it fits in a small box, keep the shape the same
    h, w = mask.shape
    scale = box_size / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    if new_h < 1:
        new_h = 1
    if new_w < 1:
        new_w = 1

    img = mask.astype(np.uint8) * 255
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized


def center_char(img, canvas_size):
    # make a black square and put the letter in the middle of it
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    h, w = img.shape
    top = (canvas_size - h) // 2
    left = (canvas_size - w) // 2
    canvas[top:top + h, left:left + w] = img

    # nudge it so the letter is balanced in the middle, not just the box
    cy, cx = ndimage.center_of_mass(canvas)
    shift_y = canvas_size / 2 - cy
    shift_x = canvas_size / 2 - cx

    canvas = ndimage.shift(canvas, (shift_y, shift_x), order=1, mode="constant", cval=0)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    return canvas


def convert_image(path, thickness=3, box_size=20, canvas_size=28):
    # run all the steps one after another
    img = load_image(path)
    mask = make_binary(img)
    mask = crop_to_char(mask)

    # scale the thickness up so it still looks right after shrinking later
    scale = max(mask.shape) / box_size
    real_thickness = thickness * scale
    mask = fix_thickness(mask, real_thickness)
    mask = crop_to_char(mask)

    small = resize_char(mask, box_size)
    final = center_char(small, canvas_size)

    return final


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    result = convert_image(input_path)
    Image.fromarray(result).save(output_path)
    print("saved:", output_path)
