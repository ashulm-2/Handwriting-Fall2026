import os
import sys
import numpy as np
from PIL import Image, ImageDraw

from emnist_style import load_image, make_binary, crop_to_char, get_skeleton, fix_thickness, resize_char, center_char

THUMB_SIZE = 100
LABEL_H = 20
EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


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


def get_image_list(folder):
    names = os.listdir(folder)
    names.sort()
    paths = []
    for name in names:
        if name.lower().endswith(EXTENSIONS):
            paths.append(os.path.join(folder, name))
    return paths


def make_thumb(img_array):
    # make every step the same size so the row lines up
    img = Image.fromarray(img_array)
    img = img.resize((THUMB_SIZE, THUMB_SIZE), Image.NEAREST)
    return img


def build_chart(all_rows, step_labels, output_path):
    num_steps = len(step_labels)
    num_rows = len(all_rows)

    grid_w = num_steps * THUMB_SIZE
    grid_h = LABEL_H + num_rows * (THUMB_SIZE + LABEL_H)

    grid = Image.new("L", (grid_w, grid_h), color=40)
    draw = ImageDraw.Draw(grid)

    # write the step names once across the top
    for col in range(num_steps):
        draw.text((col * THUMB_SIZE + 4, 4), step_labels[col], fill=255)

    y = LABEL_H
    for name, steps in all_rows:
        draw.text((4, y), name, fill=255)
        y = y + LABEL_H
        for col in range(num_steps):
            label, img_array = steps[col]
            thumb = make_thumb(img_array)
            grid.paste(thumb, (col * THUMB_SIZE, y))
        y = y + THUMB_SIZE

    grid.save(output_path)
    print("pipeline chart saved:", output_path)


def main():
    input_folder = sys.argv[1]
    output_path = sys.argv[2]

    # only doing these two characters, not the whole folder
    file_names = ["k.png", "small_a.png"]
    paths = []
    for name in file_names:
        paths.append(os.path.join(input_folder, name))

    all_rows = []
    step_labels = None

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        steps = run_pipeline(path)
        if step_labels is None:
            step_labels = []
            for label, img_array in steps:
                step_labels.append(label)
        all_rows.append((name, steps))
        print("processed:", name)

    build_chart(all_rows, step_labels, output_path)


main()
