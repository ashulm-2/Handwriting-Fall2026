import os
import sys
from PIL import Image, ImageDraw

from emnist_style import convert_image

EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
THUMB_SIZE = 140


def get_image_list(folder):
    # grab every image file in the folder
    names = os.listdir(folder)
    names.sort()
    paths = []
    for name in names:
        if name.lower().endswith(EXTENSIONS):
            paths.append(os.path.join(folder, name))
    return paths


def process_image(path, output_folder):
    # convert one image and save the result
    name = os.path.splitext(os.path.basename(path))[0]
    result = convert_image(path)

    out_path = os.path.join(output_folder, name + "_28x28.png")
    Image.fromarray(result).save(out_path)
    print("done:", name)

    # make small versions for the comparison picture
    original = Image.open(path).convert("L")
    original_thumb = original.resize((THUMB_SIZE, THUMB_SIZE), Image.NEAREST)
    result_thumb = Image.fromarray(result).resize((THUMB_SIZE, THUMB_SIZE), Image.NEAREST)

    return original_thumb, result_thumb, name


def build_grid(rows, output_path):
    # stack all the before/after pictures into one big image
    label_h = 24
    row_h = THUMB_SIZE + label_h
    grid_w = THUMB_SIZE * 2 + 20
    grid_h = row_h * len(rows)

    grid = Image.new("L", (grid_w, grid_h), color=40)
    draw = ImageDraw.Draw(grid)

    y = 0
    for original_thumb, result_thumb, name in rows:
        grid.paste(original_thumb, (0, y + label_h))
        grid.paste(result_thumb, (THUMB_SIZE + 20, y + label_h))
        draw.text((4, y + 4), name + ": original | converted", fill=255)
        y = y + row_h

    grid.save(output_path)
    print("grid saved:", output_path)


def main():
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    paths = get_image_list(input_folder)
    rows = []

    # go through every image one at a time
    for path in paths:
        row = process_image(path, output_folder)
        rows.append(row)

    build_grid(rows, os.path.join(output_folder, "comparison_grid.png"))


main()
