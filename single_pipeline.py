import sys
import numpy as np
from PIL import Image, ImageDraw

from emnist_style import load_image, make_binary, crop_to_char, get_skeleton, fix_thickness, resize_char, center_char

THUMB_SIZE = 120
LABEL_H = 20


def mask_to_img(mask):
    # turn true/false pixels back into 0/255 so we can look at it
    return mask.astype(np.uint8) * 255


def run_pipeline(path, thickness=3, box_size=20, canvas_size=28):
    # run every step and keep a picture from each one
    steps = []

    img = load_image(path)
    steps.append(("original", img))

    mask = make_binary(img)
    steps.append(("binary", mask_to_img(mask)))

    mask = crop_to_char(mask)
    steps.append(("cropped", mask_to_img(mask)))

    skeleton = get_skeleton(mask)
    steps.append(("skeleton", mask_to_img(skeleton)))

    # scale thickness the same way convert_image does
    scale = max(mask.shape) / box_size
    real_thickness = thickness * scale
    mask = fix_thickness(mask, real_thickness)
    steps.append(("even thickness", mask_to_img(mask)))

    mask = crop_to_char(mask)
    steps.append(("cropped again", mask_to_img(mask)))

    small = resize_char(mask, box_size)
    steps.append(("resized 20x20", small))

    final = center_char(small, canvas_size)
    steps.append(("final 28x28", final))

    return steps


def build_strip(steps, output_path):
    num_steps = len(steps)

    grid_w = num_steps * THUMB_SIZE
    grid_h = THUMB_SIZE + LABEL_H

    grid = Image.new("L", (grid_w, grid_h), color=40)
    draw = ImageDraw.Draw(grid)

    for col in range(num_steps):
        label, img_array = steps[col]
        thumb = Image.fromarray(img_array).resize((THUMB_SIZE, THUMB_SIZE), Image.NEAREST)
        grid.paste(thumb, (col * THUMB_SIZE, LABEL_H))
        draw.text((col * THUMB_SIZE + 4, 4), label, fill=255)

    grid.save(output_path)
    print("saved:", output_path)


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    steps = run_pipeline(input_path)
    build_strip(steps, output_path)


main()
